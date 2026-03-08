import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    # Model Weights
    RTDETR_WEIGHTS = os.getenv("RTDETR_WEIGHTS", "models/best.pt")
    CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "ViT-B/32")
    
    # Path settings
    CATALOG_DIR = os.getenv("CATALOG_DIR", "catalog")
    CATALOG_JSON = os.path.join(CATALOG_DIR, "catalog.json")
    
    # Thresholds (Boss-level Defaults)
    DEFAULT_DET_CONF = float(os.getenv("DEFAULT_DET_CONF", "0.65"))
    VERIFIED_THRESHOLD = float(os.getenv("VERIFIED_THRESHOLD", "0.85"))
    REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.60"))
    
    # Ensemble Weights (20/60/20 Split)
    WEIGHT_DETECTION = float(os.getenv("WEIGHT_DETECTION", "0.2"))
    WEIGHT_CLIP = float(os.getenv("WEIGHT_CLIP", "0.6"))
    WEIGHT_OCR = float(os.getenv("WEIGHT_OCR", "0.2"))
    
    # Hybrid Rule Thresholds
    HYBRID_VERIFY_AVG = float(os.getenv("HYBRID_VERIFY_AVG", "0.65"))
