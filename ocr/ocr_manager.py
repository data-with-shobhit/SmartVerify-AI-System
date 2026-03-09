import json
import os
import re
import numpy as np
import difflib
import easyocr
from PIL import Image
from core.logger import get_logger
from core.config import Config

logger = get_logger("ocr_manager")

class OCRManager:
    def __init__(self, catalog_json: str = None):
        self.catalog_json = catalog_json or Config.CATALOG_JSON
        self.reader = None
        self.catalog = None
        
        self._load_ocr()
        self._load_catalog()

    def _load_ocr(self):
        if self.reader is None:
            logger.info("Initializing EasyOCR (en, hi)...")
            self.reader = easyocr.Reader(['en', 'hi'])

    def _load_catalog(self):
        if self.catalog is None:
            with open(self.catalog_json, encoding="utf-8") as f:
                self.catalog = json.load(f)

    def _extract_text(self, crop: Image.Image) -> str:
        """Run OCR with automatic upscaling for better accuracy."""
        w, h = crop.size
        # Simple heuristic for upscaling small crops
        if w < 300 or h < 300:
            factor = 2 if (w > 150) else 3
            crop = crop.resize((w * factor, h * factor), Image.Resampling.LANCZOS)
        
        img_np = np.array(crop.convert("RGB"))
        result = self.reader.readtext(img_np)

        texts = []
        for line in result:
            if len(line) >= 2:
                text = line[1]
                # Keep alphanumeric and Devanagari script
                clean_text = re.sub(r'[^a-zA-Z0-9\s\u0900-\u097F]', '', str(text))
                texts.append(clean_text.lower())
        return " ".join(texts)

    def _keyword_score(self, extracted_text: str, keywords: list) -> float:
        """Fuzzy keyword matching score."""
        if not keywords or not extracted_text: return 0.0
        
        text = " ".join(extracted_text.split())
        hits = 0.0
        for kw in keywords:
            kw_l = kw.lower()
            if kw_l in text:
                hits += 1.0
                continue
            
            # Fuzzy fallback
            words = text.split()
            best_ratio = 0.0
            for word in words:
                ratio = difflib.SequenceMatcher(None, kw_l, word).ratio()
                if ratio > best_ratio: best_ratio = ratio
            
            if best_ratio >= 0.8: hits += 0.8
            elif best_ratio >= 0.7: hits += 0.4
                
        # Require at least 2.5 logical keyword matches for 100% score
        return min(1.0, hits / 2.5)

    def match(self, crop: Image.Image) -> dict:
        """Analyze crop text and match against catalog keywords."""
        extracted_text = self._extract_text(crop)
        logger.info(f"OCR Manager read: '{extracted_text}'")

        scores = []
        for sku in self.catalog:
            score = self._keyword_score(extracted_text, sku["ocr_keywords"])
            scores.append((sku, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_sku, best_score = scores[0]

        top_matches = [
            {"sku_id": s["sku_id"], "product_name": s["product_name"], "score": round(sc, 4)}
            for s, sc in scores[:3]
        ]

        return {
            "sku_id": best_sku["sku_id"],
            "product_name": best_sku["product_name"],
            "ocr_score": round(best_score, 4),
            "extracted_text": extracted_text,
            "top_matches": top_matches,
        }
