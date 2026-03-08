import os
import json
import numpy as np
import torch
import clip
import faiss
from PIL import Image
from collections import defaultdict
from core.logger import get_logger
from core.config import Config

logger = get_logger("clip_matcher")

class CLIPMatcher:
    def __init__(self, model_path: str = None, catalog_dir: str = None):
        self.model_path = model_path or Config.CLIP_MODEL_PATH
        self.catalog_dir = catalog_dir or Config.CATALOG_DIR
        self.model = None
        self.preprocess = None
        self.index = None
        self.metadata = None
        
        self._load_clip()
        self._load_catalog()

    def _load_clip(self):
        if self.model is None:
            logger.info(f"Loading CLIP model: {self.model_path}")
            self.model, self.preprocess = clip.load(self.model_path, device="cpu")
            self.model.eval()

    def _load_catalog(self):
        if self.index is None:
            faiss_path = os.path.join(self.catalog_dir, "catalog.faiss")
            meta_path = os.path.join(self.catalog_dir, "catalog_meta.json")
            logger.info(f"Loading FAISS index from: {faiss_path}")
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, encoding="utf-8") as f:
                self.metadata = json.load(f)

    def match(self, crop: Image.Image, top_k: int = 3) -> dict:
        """Match crop against the visual catalog."""
        tensor = self.preprocess(crop.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            emb = self.model.encode_image(tensor).numpy().astype("float32")
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

        distances, indices = self.index.search(emb, min(top_k * 10, self.index.ntotal))

        sku_scores = defaultdict(list)
        sku_meta_map = {}

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0: continue
            meta = self.metadata[idx]
            sku_id = meta["sku_id"]
            sku_scores[sku_id].append(float(dist))
            sku_meta_map[sku_id] = meta

        sku_avg = {
            sku_id: np.mean(sorted(scores, reverse=True)[:3])
            for sku_id, scores in sku_scores.items()
        }

        ranked = sorted(sku_avg.items(), key=lambda x: x[1], reverse=True)
        if not ranked:
            return {"sku_id": "unknown", "product_name": "Unknown", "clip_score": 0.0, "top_matches": []}

        best_sku_id, best_score = ranked[0]
        best_meta = sku_meta_map[best_sku_id]
        
        logger.info(f"CLIP match: {best_meta['product_name']} (Score: {best_score:.4f})")

        top_matches = [
            {
                "sku_id": sid,
                "product_name": sku_meta_map[sid]["product_name"],
                "score": round(score, 4),
            }
            for sid, score in ranked[:top_k]
        ]

        return {
            "sku_id": best_meta["sku_id"],
            "product_name": best_meta["product_name"],
            "brand": best_meta["brand"],
            "variant": best_meta["variant"],
            "clip_score": round(best_score, 4),
            "top_matches": top_matches,
        }
