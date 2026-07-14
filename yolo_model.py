"""
YOLOv8 inference wrapper for defect localization.
Only runs on frames PatchCore has already flagged as defective (saves compute).
"""

from ultralytics import YOLO
import os


class YOLOModel:
    def __init__(self, weights_path: str):
        self.ready = os.path.exists(weights_path)
        if self.ready:
            self.model = YOLO(weights_path)
        else:
            self.model = None  # allows app to boot before YOLO is trained

    def infer(self, img_bgr, conf_thresh: float = 0.4):
        """
        Returns list of (x1, y1, x2, y2, class_name, confidence)
        """
        if not self.ready:
            return []

        results = self.model(img_bgr, conf=conf_thresh, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = results.names[int(box.cls[0])]
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, cls_name, conf))
        return boxes
