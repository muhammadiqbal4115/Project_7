"""
PatchCore inference wrapper.

Expects a memory bank file (built during training, see train_patchcore.py)
containing:
    {
        "memory_bank": torch.Tensor [N, C]   # coreset of patch features from "good" images
        "backbone_name": str
        "layer_names": list[str]
        "image_size": int
        "max_score_seen": float   # for normalizing score to ~0-1 range (optional)
    }
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import timm
from sklearn.neighbors import NearestNeighbors


class PatchCoreModel:
    def __init__(self, memory_bank_path: str, backbone_name: str = "wide_resnet50_2",
                 device: str = "cpu", image_size: int = 224):
        self.device = device
        self.image_size = image_size

        # --- Load backbone (pretrained feature extractor) ---
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True,
            out_indices=(1, 2)  # mid-level layers, standard for PatchCore
        )
        self.backbone.eval().to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # --- Load memory bank ---
        try:
            checkpoint = torch.load(memory_bank_path, map_location=device)
            self.memory_bank = checkpoint["memory_bank"].numpy()
            self.max_score_seen = checkpoint.get("max_score_seen", 10.0)
            self.nn_index = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
            self.nn_index.fit(self.memory_bank)
            self.ready = True
        except FileNotFoundError:
            # Allows the app to boot before training is done; infer() will return dummy values
            self.ready = False
            self.max_score_seen = 10.0

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size))
        img_norm = (img_resized / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self.device)

    def _extract_features(self, img_bgr: np.ndarray):
        tensor = self._preprocess(img_bgr)
        with torch.no_grad():
            feats = self.backbone(tensor)  # list of feature maps at chosen layers

        # Resize all layers to same spatial size, concat along channel dim
        target_size = feats[0].shape[-2:]
        resized = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in feats]
        combined = torch.cat(resized, dim=1)  # [1, C, H, W]

        b, c, h, w = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(-1, c)  # [H*W, C]
        return patches.cpu().numpy(), (h, w)

    def infer(self, img_bgr: np.ndarray, threshold: float = 0.5):
        """
        Returns:
            is_defective (bool)
            score (float, roughly 0-1 normalized)
            heatmap (np.array, same H,W,3 as input, colorized anomaly map)
        """
        h_orig, w_orig = img_bgr.shape[:2]

        if not self.ready:
            # No memory bank yet — safe fallback so the app doesn't crash pre-training
            heatmap = np.zeros_like(img_bgr)
            return False, 0.0, heatmap

        patches, (h, w) = self._extract_features(img_bgr)
        distances, _ = self.nn_index.kneighbors(patches)  # [H*W, 1]
        distances = distances.reshape(h, w)

        raw_score = float(distances.max())
        norm_score = min(raw_score / self.max_score_seen, 1.0)

        # Build heatmap
        dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
        dist_resized = cv2.resize(dist_norm, (w_orig, h_orig))
        heatmap_color = cv2.applyColorMap((dist_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        is_defective = norm_score > threshold
        return is_defective, norm_score, heatmap_color
