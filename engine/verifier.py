from PIL import Image
from core.logger import get_logger
from core.config import Config

logger = get_logger("verification_engine")

class VerificationEngine:
    def __init__(self, clip_matcher=None, ocr_manager=None):
        self.clip_matcher = clip_matcher
        self.ocr_manager = ocr_manager

    def verify_crop(
        self,
        crop: Image.Image,
        det_confidence: float,
        det_class_label: str,
        det_sku_id: str,
        use_detection: bool = True,
        use_clip: bool = True,
        use_ocr: bool = True,
        verified_threshold: float = None,
        review_threshold: float = None,
    ) -> dict:
        """
        Ensemble verification logic. 
        Uses RT-DETR (Detection), CLIP (Visual), and OCR (Text) to reach a verdict.
        """
        verified_threshold = verified_threshold if verified_threshold is not None else Config.VERIFIED_THRESHOLD
        review_threshold = review_threshold if review_threshold is not None else Config.REVIEW_THRESHOLD

        clip_result = None
        ocr_result  = None
        
        logger.info(f"Verifying crop with DET confidence: {det_confidence:.2f} (Label: {det_class_label})")

        # ── Execution ──────────────────────────────────────────────────────────
        if use_clip and self.clip_matcher:
            clip_result = self.clip_matcher.match(crop)
        
        if use_ocr and self.ocr_manager:
            ocr_result = self.ocr_manager.match(crop)

        # ── Math ───────────────────────────────────────────────────────────────
        # Apply the user-approved 20/40/40 weights
        weights = {
            "detection": Config.WEIGHT_DETECTION if use_detection else 0.0,
            "clip":      Config.WEIGHT_CLIP      if use_clip else 0.0,
            "ocr":       Config.WEIGHT_OCR       if use_ocr else 0.0,
        }
        total_w = sum(weights.values()) or 1.0
        norm_weights = {k: v / total_w for k, v in weights.items()}

        det_score  = det_confidence
        clip_score = clip_result["clip_score"] if clip_result else 0.0
        ocr_score  = ocr_result["ocr_score"]   if ocr_result  else 0.0

        final_score = (
            norm_weights["detection"] * det_score +
            norm_weights["clip"]      * clip_score +
            norm_weights["ocr"]       * ocr_score
        )

        # ── SKU Resolution (Master Override Priority: CLIP > OCR > Detection) ───
        verified_sku_id = "unknown"
        verified_name = "Unknown"
        
        if use_clip and clip_result and clip_result["sku_id"] != "unknown":
            verified_sku_id = clip_result["sku_id"]
            verified_name = clip_result["product_name"]
        elif use_ocr and ocr_result and ocr_result["sku_id"] != "unknown":
            verified_sku_id = ocr_result["sku_id"]
            verified_name = ocr_result["product_name"]
        elif use_detection and det_sku_id != "unknown":
            verified_sku_id = det_sku_id
            verified_name = det_class_label

        # ── Verdict Logic ──────────────────────────────────────────────────────
        det_clip_avg = (det_score + clip_score) / 2 if (use_detection and use_clip) else 0.0
        status = "MISMATCH"

        if use_detection and det_confidence >= 0.95:
            status = "VERIFIED"
        elif use_clip and clip_score >= 0.80:
            status = "VERIFIED"
        elif det_clip_avg >= Config.HYBRID_VERIFY_AVG:
            status = "VERIFIED"
        elif final_score >= verified_threshold:
            status = "VERIFIED"
        elif final_score >= review_threshold:
            status = "REVIEW"

        logger.info(f"Verification Verdict: {status} | Score: {final_score:.2f} | Match: {verified_name}")

        return {
            "status": status,
            "verified_sku_id": verified_sku_id,
            "verified_as": verified_name,
            "det_confidence": round(det_score, 4),
            "clip_score": round(clip_score, 4),
            "ocr_score": round(ocr_score, 4),
            "final_score": round(final_score, 4),
            "extracted_text": ocr_result["extracted_text"] if ocr_result else "",
            "crop": crop,
        }
