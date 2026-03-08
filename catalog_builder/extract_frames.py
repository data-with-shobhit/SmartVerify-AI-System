"""
pipeline/extract_frames.py
--------------------------
Extract frames from a video OR copy images from a folder.
Returns a list of (frame_index, PIL.Image) tuples.
"""

import cv2
import os
from PIL import Image
import numpy as np


def extract_from_video(video_path: str, every_n: int = 10) -> list:
    """
    Extract 1 frame every N frames.
    Returns list of (frame_idx, PIL.Image)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, Image.fromarray(rgb)))
        idx += 1
    cap.release()
    print(f"[extract_frames] {video_path}: extracted {len(frames)} frames from {idx} total")
    return frames


def extract_from_images(image_dir: str) -> list:
    """
    Load all images from a folder.
    Returns list of (filename, PIL.Image)
    """
    supported = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    files = sorted([f for f in os.listdir(image_dir)
                    if f.lower().endswith(supported)])
    frames = []
    for fname in files:
        try:
            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            frames.append((fname, img))
        except Exception as e:
            print(f"[WARN] Could not open {fname}: {e}")
    print(f"[extract_frames] {image_dir}: loaded {len(frames)} images")
    return frames


def extract(source: str, every_n: int = 10) -> list:
    """
    Auto-detect if source is a video file or image folder.
    Returns list of (id, PIL.Image)
    """
    video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    if os.path.isfile(source) and source.lower().endswith(video_exts):
        return extract_from_video(source, every_n)
    elif os.path.isdir(source):
        return extract_from_images(source)
    else:
        raise ValueError(f"Source must be a video file or image folder: {source}")
