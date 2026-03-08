"""
build_catalog.py
----------------
Reads your 4 SKU videos, extracts frames, generates CLIP embeddings,
and saves a FAISS index + metadata file.

Expected videos (place in sku_videos/ folder):
    amul_masti_butter_milk.mp4
    maggi_cuppa_noddles.mp4
    madhur_sugar.mp4
    tata_sampan_toor _daal.mp4   (space in name is fine)

Run:
    python catalog/build_catalog.py --source sku_videos --output catalog/
    python catalog/build_catalog.py --source sku_videos --output catalog/ --model_path C:/Users/shobh/.cache/clip/ViT-B-32.pt
"""

import os
import json
import argparse
import numpy as np
import torch
import clip
import faiss
import cv2
from PIL import Image
from tqdm import tqdm
from collections import Counter


CATALOG_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "catalog", "catalog.json")

# Map every reasonable filename variation -> sku_id
# Keys are lowercased, spaces+dashes+underscores collapsed
FILENAME_MAP = {
    "amul_masti_butter_milk":   "sku_001",
    "amul_masti_buttermilk":    "sku_001",
    "amulmastibuttermilk":      "sku_001",
    "maggi_cuppa_noddles":      "sku_002",   # note: typo in your filename
    "maggi_cuppa_noodles":      "sku_002",
    "maggicuppanoodles":        "sku_002",
    "madhur_sugar":             "sku_003",
    "madhursugar":              "sku_003",
    "tata_sampan_toor_daal":    "sku_004",   # space in filename handled below
    "tata_sampann_toor_daal":   "sku_004",
    "tatasampantoor_daal":      "sku_004",
    "tatasampantoordaal":       "sku_004",
}


def normalize_filename(fname: str) -> str:
    """Lowercase, strip extension, collapse spaces/dashes/underscores."""
    stem = os.path.splitext(fname)[0]
    stem = stem.lower().strip()
    # collapse all whitespace and separators into single underscore
    import re
    stem = re.sub(r'[\s\-_]+', '_', stem)
    return stem


def extract_frames(video_path: str, every_n: int = 15) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames = []
    idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  -> Video: {total} frames at {fps:.1f} fps = {total/fps:.1f}s")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        idx += 1
    cap.release()
    return frames


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / (norm + 1e-8)


def build_catalog(source_dir: str, output_dir: str,
                  every_n: int = 10, model_path: str = "ViT-B/32"):

    with open(CATALOG_JSON, encoding="utf-8") as f:
        catalog = json.load(f)
    sku_lookup = {s["sku_id"]: s for s in catalog}

    print("[INFO] Loading CLIP model on CPU...")
    model, preprocess = clip.load(model_path, device="cpu")
    model.eval()

    video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    video_files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith(video_exts)
    ]

    if not video_files:
        raise ValueError(f"No video files found in '{source_dir}'")

    print(f"\n[INFO] Found {len(video_files)} videos: {video_files}\n")

    all_embeddings = []
    all_metadata = []

    for fname in video_files:
        norm = normalize_filename(fname)
        sku_id = FILENAME_MAP.get(norm)

        if not sku_id:
            print(f"[WARN] '{fname}' (normalized: '{norm}') not in FILENAME_MAP — skipping")
            print(f"       Known keys: {list(FILENAME_MAP.keys())}")
            continue

        sku = sku_lookup[sku_id]
        print(f"[INFO] '{fname}' -> {sku['product_name']}")

        video_path = os.path.join(source_dir, fname)
        try:
            frames = extract_frames(video_path, every_n)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        print(f"  -> Encoding {len(frames)} frames...")
        for img in tqdm(frames, desc=sku["product_name"]):
            try:
                tensor = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    emb = model.encode_image(tensor).numpy().astype("float32")
                emb = normalize_vec(emb)
                all_embeddings.append(emb)
                all_metadata.append({
                    "sku_id": sku["sku_id"],
                    "product_name": sku["product_name"],
                    "brand": sku["brand"],
                    "variant": sku["variant"],
                    "weight": sku["weight"],
                    "ocr_keywords": sku["ocr_keywords"],
                    "source": fname,
                })
            except Exception as e:
                print(f"  [WARN] Frame encoding error: {e}")

    if not all_embeddings:
        raise ValueError("No embeddings generated. Check video filenames match FILENAME_MAP.")

    # Build FAISS index (Inner Product on L2-normalized = cosine similarity)
    dim = 512
    index = faiss.IndexFlatIP(dim)
    matrix = np.vstack(all_embeddings)
    index.add(matrix)

    os.makedirs(output_dir, exist_ok=True)
    faiss_path = os.path.join(output_dir, "catalog.faiss")
    meta_path  = os.path.join(output_dir, "catalog_meta.json")

    faiss.write_index(index, faiss_path)
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"CATALOG BUILT SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"  Total embeddings : {len(all_embeddings)}")
    print(f"  FAISS index      : {faiss_path}")
    print(f"  Metadata         : {meta_path}")
    print(f"\nPer-SKU breakdown:")
    counts = Counter(m["product_name"] for m in all_metadata)
    for name, count in counts.items():
        print(f"  {name:<35} {count} embeddings")
    print(f"\nReady. Now run: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="catalog_builder/sku_videos",
                        help="Folder containing your 4 SKU videos")
    parser.add_argument("--output", default="catalog",
                        help="Output folder for FAISS index")
    parser.add_argument("--every_n", type=int, default=2,
                        help="Extract 1 frame every N frames (default 6, ~100 frames per 20s video at 30fps)")
    parser.add_argument("--model_path", default="ViT-B/32",
                        help="CLIP model name or path to local .pt file")
    args = parser.parse_args()

    build_catalog(args.source, args.output, args.every_n, args.model_path)
