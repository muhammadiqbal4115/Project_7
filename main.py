"""
Screen Printing Defect Inspector
PatchCore (anomaly detection) + YOLOv8 (defect localization) + streamlit-webrtc (live video)
"""

import streamlit as st
import cv2
import av
import os
import glob
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from patchcore_model import PatchCoreModel
from yolo_model import YOLOModel

# ---------- Page config + theme (dark) ----------
st.set_page_config(page_title="Screen Print Inspector", page_icon="🖨️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0F1117; }
    h1, h2, h3 { color: #93C5FD; }
    .stButton>button { background-color: #1E3A8A; color: white; }
    div[data-testid="stMetricValue"] { color: #93C5FD; }
</style>
""", unsafe_allow_html=True)

st.title("🖨️ Screen Printing Defect Inspector")
st.caption("PatchCore anomaly detection + YOLOv8 defect localization")

STATIC_DIR = "static"


# ---------- Load models (cached, once per session) ----------
@st.cache_resource
def load_models():
    print("=" * 60)

    print("STEP 1: Starting PatchCore")
    t = time.time()

    patchcore = PatchCoreModel(
        memory_bank_path="models/memory_bank.pt",
        backbone_name="wide_resnet50_2",
        device="cpu",
    )

    print(f"STEP 2: PatchCore loaded in {time.time()-t:.2f}s")

    print("STEP 3: Starting YOLO")
    t = time.time()

    yolo = YOLOModel(weights_path="models/yolo_best.pt")

    print(f"STEP 4: YOLO loaded in {time.time()-t:.2f}s")

    print("STEP 5: Finished")

    return patchcore, yolo


with st.spinner("Loading models..."):
    patchcore_model, yolo_model = load_models()


@st.cache_data
def list_sample_images():
    if not os.path.isdir(STATIC_DIR):
        return []
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(STATIC_DIR, ext)))
    return sorted(paths)


# ---------- Sidebar controls ----------
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Input mode", ["Sample Images", "Upload Image", "Live Camera"])
threshold = st.sidebar.slider("Anomaly threshold", 0.0, 1.0, 0.6, 0.01)
run_yolo_on_fail = st.sidebar.checkbox("Localize defect with YOLO on FAIL", value=True)
yolo_conf = st.sidebar.slider("YOLO confidence", 0.1, 0.9, 0.1, 0.05)
frame_skip = st.sidebar.slider("Process every Nth frame (live mode)", 1, 5, 2)

st.sidebar.markdown("---")
st.sidebar.caption("PatchCore flags PASS/FAIL. YOLO only runs on FAIL frames to save compute.")


# ---------- Shared inference + drawing logic ----------
def process_frame(img_bgr: np.ndarray):
    is_defective, score, heatmap = patchcore_model.infer(img_bgr, threshold=threshold)

    overlay = img_bgr.copy()
    boxes = []

    if is_defective:
        overlay = cv2.addWeighted(overlay, 0.6, heatmap, 0.4, 0)
        label, color = f"FAIL  score={score:.3f}", (0, 0, 255)

        if run_yolo_on_fail:
            boxes = yolo_model.infer(img_bgr, conf_thresh=yolo_conf)
            for (x1, y1, x2, y2, cls_name, conf) in boxes:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(overlay, f"{cls_name} {conf:.2f}", (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    else:
        label, color = f"PASS  score={score:.3f}", (0, 200, 0)

    cv2.rectangle(overlay, (0, 0), (330, 45), (0, 0, 0), -1)
    cv2.putText(overlay, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return overlay, is_defective, score, boxes


def show_result(img_bgr: np.ndarray, overlay: np.ndarray, is_defective: bool,
                 score: float, boxes: list, download_name: str):
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
    with col2:
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Result", use_column_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Verdict", "FAIL" if is_defective else "PASS")
    m2.metric("Anomaly score", f"{score:.3f}")
    m3.metric("Defects found", len(boxes))

    if boxes:
        st.write("**Detected defects:**")
        for (x1, y1, x2, y2, cls_name, conf) in boxes:
            st.write(f"- {cls_name} (confidence {conf:.2f}) at [{x1},{y1},{x2},{y2}]")

    # Encode result image for download
    success, buffer = cv2.imencode(".png", overlay)
    if success:
        st.download_button(
            label="⬇️ Download result image",
            data=buffer.tobytes(),
            file_name=download_name,
            mime="image/png",
        )


# ---------- Sample Images mode ----------
if mode == "Sample Images":
    st.subheader("🗂️ Sample Images")
    sample_paths = list_sample_images()

    if not sample_paths:
        st.warning(f"No images found in `{STATIC_DIR}/`. Add .png/.jpg files there.")
    else:
        labels = [os.path.basename(p) for p in sample_paths]
        selected_label = st.selectbox("Choose a sample image", labels)
        selected_path = sample_paths[labels.index(selected_label)]

        # Thumbnail strip so users can see all samples at a glance
        with st.expander("Preview all sample images"):
            thumb_cols = st.columns(5)
            for i, p in enumerate(sample_paths):
                with thumb_cols[i % 5]:
                    st.image(p, caption=os.path.basename(p), use_column_width=True)

        img_bgr = cv2.imread(selected_path)
        if img_bgr is None:
            st.error(f"Could not read {selected_path}")
        else:
            overlay, is_defective, score, boxes = process_frame(img_bgr)
            result_name = f"result_{os.path.splitext(selected_label)[0]}.png"
            show_result(img_bgr, overlay, is_defective, score, boxes, result_name)

# ---------- Upload mode ----------
elif mode == "Upload Image":
    st.subheader("🖼️ Upload Image")
    uploaded = st.file_uploader("Upload a screen-printed sample", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        overlay, is_defective, score, boxes = process_frame(img_bgr)
        result_name = f"result_{os.path.splitext(uploaded.name)[0]}.png"
        show_result(img_bgr, overlay, is_defective, score, boxes, result_name)
    else:
        st.info("Upload an image to run inspection.")

# ---------- Live camera mode ----------
else:
    st.subheader("📹 Live Feed")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class DefectProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
            self.last_overlay = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            if self.frame_count % frame_skip == 0 or self.last_overlay is None:
                overlay, _, _, _ = process_frame(img)
                self.last_overlay = overlay
            else:
                overlay = self.last_overlay

            return av.VideoFrame.from_ndarray(overlay, format="bgr24")

    webrtc_streamer(
        key="defect-detection",
        video_processor_factory=DefectProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.info("Grant camera permission in your browser.")
