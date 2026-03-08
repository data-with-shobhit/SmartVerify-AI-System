import numpy as np
from PIL import Image
from ultralytics import RTDETR
import cv2
from core.logger import get_logger
from core.config import Config

logger = get_logger("detect_manager")

class DetectorManager:
    def __init__(self, weights_path: str = None):
        self.weights_path = weights_path or Config.RTDETR_WEIGHTS
        self.model = None
        self._load_model()

    def _load_model(self):
        """Lazy load the RT-DETR model."""
        if self.model is None:
            logger.info(f"Initializing RT-DETR with: {self.weights_path}")
            self.model = RTDETR(self.weights_path)

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)

    def detect_and_crop(self, image: Image.Image, conf_threshold: float = None) -> list:
        """Run RT-DETR, apply NMS, and return crops."""
        conf_threshold = conf_threshold if conf_threshold is not None else Config.DEFAULT_DET_CONF
        img_np = np.array(image.convert("RGB"))
        
        results = self.model(img_np, conf=conf_threshold, verbose=False)
        raw_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None: continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_label = self.model.names.get(cls_id, str(cls_id))
                raw_detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_label": cls_label,
                    "cls_id": cls_id,
                    "det_confidence": conf,
                })

        # Logic: Deduplicate overlapping boxes (NMS style)
        raw_detections.sort(key=lambda x: x["det_confidence"], reverse=True)
        final_detections = []
        for i, det in enumerate(raw_detections):
            keep = True
            for best in final_detections:
                if det["cls_id"] == best["cls_id"]:
                    iou = self._calculate_iou(det["bbox"], best["bbox"])
                    if iou > 0.45:
                        keep = False
                        logger.info(f"   -> [NMS] Removing overlapping {det['class_label']} (IoU: {iou:.2f})")
                        break
            if keep:
                final_detections.append(det)

        # Generate crops
        for det in final_detections:
            x1, y1, x2, y2 = det["bbox"]
            pad = 10
            h, w = img_np.shape[:2]
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
            det["crop"] = image.crop((x1p, y1p, x2p, y2p))
            logger.info(f"   -> [DET] Box: {det['class_label']} Conf: {det['det_confidence']:.4f}")

        return final_detections

    def draw_boxes(self, image: Image.Image, detections: list) -> Image.Image:
        """Draw bounding boxes for visual verification."""
        img_np = np.array(image).copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det.get("verified_as", det.get("class_label", "Unknown"))
            conf = det.get("final_score", det["det_confidence"])
            status = det.get("status", "PENDING")
            color = {
                "VERIFIED": (0, 200, 0),
                "REVIEW": (255, 165, 0),
                "MISMATCH": (200, 0, 0),
                "PENDING": (150, 150, 150),
            }.get(status, (150, 150, 150))
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_np, f"{label} {conf:.2f}", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return Image.fromarray(img_np)
