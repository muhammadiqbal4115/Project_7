%%writefile train_patchcore.py

"""
Train PatchCore: build a coreset memory bank from "good" images only.
Run this on Kaggle (GPU) — it matches the feature extraction logic in patchcore_model.py.

Expected dataset structure (MVTec format):
    LSM_1/
      train/
        good/
          img001.png
          img002.png
          ...
      test/
        good/
        defect_type_1/
        defect_type_2/

Output: memory_bank.pt  -> copy this into screen-print-inspector/models/
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import timm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def preprocess(img_bgr, image_size, mean, std):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))
    img_norm = (img_resized / 255.0 - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def extract_patch_features(backbone, tensor, device):
    tensor = tensor.to(device)
    with torch.no_grad():
        feats = backbone(tensor)
    target_size = feats[0].shape[-2:]
    resized = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in feats]
    combined = torch.cat(resized, dim=1)
    b, c, h, w = combined.shape
    patches = combined.permute(0, 2, 3, 1).reshape(-1, c)
    return patches.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to train/good folder, e.g. /kaggle/input/LSM_1/train/good")
    parser.add_argument("--test_good_dir", type=str, default=None,
                         help="Optional: test/good folder, used to calibrate max_score_seen")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--coreset_size", type=int, default=10000)
    parser.add_argument("--output", type=str, default="memory_bank.pt")
    parser.add_argument("--max_patches_per_image", type=int, default=200,
                         help="Randomly subsample patches per image to bound memory usage.")
    parser.add_argument("--kmeans_batch_size", type=int, default=4096)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    backbone = timm.create_model(args.backbone, pretrained=True, features_only=True, out_indices=(1, 2))
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    data_dir = "/kaggle/input/datasets/muhammadiqbal4115/dataset/LSM_1/train/images"  # fill in from previous step
    image_paths = glob.glob(os.path.join(data_dir, "*.png")) + \
                  glob.glob(os.path.join(data_dir, "*.jpg")) + \
                  glob.glob(os.path.join(data_dir, "*.jpeg"))
    print(f"Found {len(image_paths)} good images.")
    assert len(image_paths) > 0, "No images found — check --data_dir path."

    rng = np.random.default_rng(42)

    # Incremental k-means: never hold more than one batch of patches in RAM at a time.
    # This keeps memory bounded regardless of how many "good" images you have.
    kmeans = MiniBatchKMeans(
        n_clusters=args.coreset_size,
        batch_size=args.kmeans_batch_size,
        n_init=3,
        random_state=42,
    )

    buffer = []
    buffer_count = 0
    fitted_once = False

    for path in tqdm(image_paths, desc="Extracting features"):
        img = cv2.imread(path)
        if img is None:
            continue
        tensor = preprocess(img, args.image_size, mean, std)
        patches = extract_patch_features(backbone, tensor, device)

        # Subsample patches per image so a single image can't dominate memory
        if len(patches) > args.max_patches_per_image:
            idx = rng.choice(len(patches), size=args.max_patches_per_image, replace=False)
            patches = patches[idx]

        buffer.append(patches)
        buffer_count += len(patches)

        # Once buffer is big enough for one k-means batch, feed it and clear
        if buffer_count >= args.kmeans_batch_size:
            batch = np.concatenate(buffer, axis=0)
            if len(batch) >= args.coreset_size:
                kmeans.partial_fit(batch)
                fitted_once = True
            buffer = []
            buffer_count = 0

    # Flush any remaining patches
    if buffer:
        batch = np.concatenate(buffer, axis=0)
        if len(batch) >= args.coreset_size and fitted_once:
            kmeans.partial_fit(batch)
        elif not fitted_once:
            # Not enough images/patches to fill even one batch — fall back to direct fit
            kmeans = MiniBatchKMeans(n_clusters=min(args.coreset_size, len(batch)),
                                      batch_size=args.kmeans_batch_size, n_init=3, random_state=42)
            kmeans.fit(batch)
            fitted_once = True

    assert fitted_once, "Not enough patch data collected to build memory bank — check dataset size."

    memory_bank = kmeans.cluster_centers_
    print(f"Final memory bank size: {memory_bank.shape}")

    # Calibrate max_score_seen using held-out good images (for score normalization in the app)
    max_score_seen = 10.0
    if args.test_good_dir and os.path.isdir(args.test_good_dir):
        nn_index = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(memory_bank)
        test_paths = sorted(glob.glob(os.path.join(args.test_good_dir, "*.png")) +
                             glob.glob(os.path.join(args.test_good_dir, "*.jpg")))
        scores = []
        for path in tqdm(test_paths, desc="Calibrating threshold on held-out good images"):
            img = cv2.imread(path)
            if img is None:
                continue
            tensor = preprocess(img, args.image_size, mean, std)
            patches = extract_patch_features(backbone, tensor, device)
            distances, _ = nn_index.kneighbors(patches)
            scores.append(distances.max())
        if scores:
            max_score_seen = float(np.percentile(scores, 99))  # robust to outliers
            print(f"Calibrated max_score_seen (99th percentile of good scores): {max_score_seen:.4f}")

    torch.save({
        "memory_bank": torch.from_numpy(memory_bank).float(),
        "backbone_name": args.backbone,
        "layer_names": ["layer2", "layer3"],
        "image_size": args.image_size,
        "max_score_seen": max_score_seen,
    }, args.output)
    print(f"Saved memory bank to {args.output}")


if __name__ == "__main__":
    main()




!python train_patchcore.py \
  --data_dir /kaggle/input/datasets/muhammadiqbal4115/dataset/LSM_1/train/images \
  --test_good_dir /kaggle/input/datasets/muhammadiqbal4115/dataset/LSM_1/test/images \
  --coreset_size 10000 \
  --output /kaggle/working/memory_bank.pt