from PIL import Image
from core.logger import get_logger

logger = get_logger("pipeline")

class VerificationPipeline:
    """
    Orchestrates the verification process: Detection -> Verification -> Reconciliation.
    Extracts all logic from the frontend UI into this dedicated backend controller.
    """
    def __init__(self, detector, verifier, reconciler):
        self.detector = detector
        self.verifier = verifier
        self.reconciler = reconciler
        self.rt_map = {0: "sku_004", 1: "sku_001", 2: "sku_002", 3: "sku_003"}

    def process_order(self, images_info: list, expected_order: list, config: dict, ui_callback=None) -> dict:
        """
        Executes the full pipeline on a batch of images and compares against the expected order.
        
        images_info: List of tuples (filename, PIL.Image)
        expected_order: List of order items from the bill
        config: Dictionary of thresholds and feature flags
        ui_callback: Optional callable for logging progress to a UI (e.g. st.write)
        """
        def log(msg):
            if ui_callback: ui_callback(msg)
            logger.info(msg)

        all_verifications = []
        annotated_frames = []
        
        for filename, img in images_info:
            log(f"🖼️ Checking `{filename}`...")
            
            # Step 1: Detect & Crop
            dets = self.detector.detect_and_crop(img, conf_threshold=config.get("conf_threshold"))
            
            frame_res = []
            for det in dets:
                # Step 2: Verify Each Item using Ensemble
                res = self.verifier.verify_crop(
                    crop=det["crop"],
                    det_confidence=det["det_confidence"],
                    det_class_label=det["class_label"],
                    det_sku_id=self.rt_map.get(det["cls_id"], "unknown"),
                    use_detection=config.get("use_detection", True),
                    use_clip=config.get("use_clip", True),
                    use_ocr=config.get("use_ocr", True),
                    verified_threshold=config.get("verified_thresh"),
                    review_threshold=config.get("review_thresh")
                )
                res["bbox"] = det["bbox"]
                res["frame_id"] = filename
                frame_res.append(res)
                all_verifications.append(res)
            
            # Step 3: Draw Annotations
            annotated_frames.append(self.detector.draw_boxes(img, frame_res))
            
        # Step 4: Reconcile Physical Items against Expected Order
        reconciliation = self.reconciler.reconcile(expected_order, all_verifications)
        
        return {
            "all_verifications": all_verifications,
            "annotated_frames": annotated_frames,
            "reconciliation": reconciliation
        }
