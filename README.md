# 📦 SmartVerify AI: Quick Commerce Order Reconciliation

A state-of-the-art multimodal AI proof-of-concept (POC) designed to automatically verify packaged items in quick commerce orders (e.g., Blinkit, Zepto, Swiggy Instamart) to ensure perfect order accuracy before delivery.

**🎥 Watch the Demo:**  
[Click here to view the demo video](https://www.youtube.com/watch?v=WLC1klk0KcA)

*(See it in action: Fast, AI-driven order reconciliation)*

---

## 🚨 The Problem Statement

Quick commerce platforms operate under extreme time constraints, packing orders in minutes. This speed inevitably leads to:
* **Missing Items:** Customers receive fewer products than they paid for.
* **Variant Mismatches:** E.g., receiving *Kurkure Solid Masti* instead of *Kurkure Masala Munch*.
* **Wrong Products:** Visually similar items getting swapped during rapid picking.

**The Cost:** Manual barcode scanning is slow, and human verification is prone to visual fatigue. Every incorrect delivery results in refund costs, logistics overhead for returns, and severe damage to customer trust and retention.

---

## 💼 Business Impact & Value Proposition

Implementing SmartVerify AI at the packing station solves these issues instantly:

1. **Massive Cost Reduction:** Eliminates the logistics cost of reverse pickups and refunds associated with missing or wrong items.
2. **Time Savings (Agility):** The AI verifies a complete basket in 1-3 seconds, far faster than manual human checking or individual barcode scanning.
3. **Data Security & Privacy:** The entire pipeline is designed to run locally / on-edge. No sensitive store feeds or order data need to cross the internet.
4. **Agile Model Iteration:** Because the system leverages CLIP for zero-shot catalog matching, new SKUs can be added to the database by simply dropping in a video, without needing to retrain heavy neural networks.

---

## ⚙️ How It Works: The Triple-Layer Engine

The system uses a robust ensemble of three distinct AI models to aggressively cross-verify products:

1. **RT-DETR (Object Detection):** Identifies the presence and boundaries of packages in the frame, applying IoU-based Non-Maximum Suppression (NMS) to prevent duplicate counting.
2. **OpenAI CLIP (Visual Matching):** Extracts dense visual embeddings from the cropped package and compares it against a FAISS vector catalog of known products. (Weight: 60%)
3. **EasyOCR (Text Matching):** Reads flavor text, brand names, and weights directly off the packaging to disambiguate visually identical variants. (Weight: 20%)

*Note: If CLIP is highly confident (>85%), it can override weak OCR/Detection scores to prevent false negatives on blurry images.*

---

## 🧠 Architecture Deep-Dive: Built as a Visual RAG

The SmartVerify AI engine operates fundamentally as a **Visual Retrieval-Augmented Generation (RAG)** system optimized for high-speed edge environments. 

### 1. Zero-Shot Cataloging with OpenAI CLIP
Instead of retraining a heavy neural network every time an online grocery store adds a new product, we leverage **CLIP (`ViT-B/32`)**. 
- During setup, `build_catalog.py` reads short videos of products and automatically extracts frames. 
- It passes these frames through the CLIP Vision Transformer, translating each image into a dense, universal **512-dimensional floating-point embedding**.

### 2. High-Speed Retrieval with FAISS
- All 512D embeddings are indexed into **Facebook AI Similarity Search (FAISS)**.
- Specifically, we use `IndexFlatIP` (Inner Product) on L2-normalized vectors. Because the vectors are L2-normalized natively by CLIP, an inner product directly yields the **Cosine Similarity** scale (0.0 to 1.0).
- When a new image from the packing station arrives, the system crops it, generates a 512D vector on the fly, and queries the FAISS database to identify the nearest catalog neighbor in mere milliseconds.

### 3. Localization with RT-DETR
- Before CLIP can match a product, the system must find where the products are hiding on the loud packing table. We employ **RT-DETR (Real-Time DEtection TRansformer)** for this task.
- Unlike traditional YOLO models, RT-DETR provides higher bounding-box fidelity crucial for dense overlaps. We implemented custom highly-aggressive **IoU-based Non-Maximum Suppression (NMS)** directly into `detectors/rtdetr_detector.py` to guarantee that overlapping boxes don't falsely duplicate quantities on the UI receipt.

---

## 📂 Project File Structure

The project has been refactored into a scalable, enterprise-grade modular architecture:

```text
order_verification/
├── app.py                      # Main Streamlit UI Presentation Layer
├── requirements.txt            # Python dependencies
├── .env                        # AI Model Thresholds & Weights Configuration
│
├── core/
│   ├── config.py               # Centralized config manager (loads .env)
│   └── logger.py               # UTF-8 Safe Console & File Logger
│
├── catalog/
│   ├── catalog.json            # Master SKU metadata definition
│   ├── catalog.faiss           # Compiled FAISS vector index 
│   └── catalog_meta.json       # Compiled metadata pointers
│
├── catalog_builder/
│   ├── build_catalog.py        # Script to convert videos into FAISS catalog
│   ├── extract_frames.py       # Frame extraction utility
│   └── sku_videos/             # Directory for raw training videos
│
├── detectors/
│   └── rtdetr_detector.py      # RT-DETR model loading & NMS logic
├── matchers/
│   └── clip_matcher.py         # OpenAI CLIP & Vector DB matching
├── ocr/
│   └── ocr_manager.py          # EasyOCR text extraction & keyword scoring
├── engine/
│   ├── pipeline.py             # Main execution orchestrator (batch processing)
│   ├── verifier.py             # 3-layer ensemble math & decision logic
│   └── reconciler.py           # Compares detections vs customer bill
│
└── models/
    └── best.pt                 # Compiled RT-DETR detection weights
```

---

## 🚀 Setup & Usage

### 1. Installation
```bash
# Install required dependencies
pip install -r requirements.txt

# Download CLIP weights manually if standard network download fails:
# Save to: C:\Users\<you>\.cache\clip\ViT-B-32.pt
```

### 2. Build the Product Catalog
Place a short video of the products you wish to track into the `catalog_builder/sku_videos/` directory. Ensure the filename matches the `product_name` in `catalog.json` (lowercase with underscores).
```bash
python catalog_builder/build_catalog.py
```

### 3. Run the Application
Start the Streamlit UI to process orders:
```bash
streamlit run app.py
```

*All thresholds (Detection confidence, weights, ensemble thresholds) can be tuned live in the UI sidebar or permanently in the `.env` file.*

---

## 🔭 Future Scope & Roadmap

### 1. Layer 2: CLIP — Label Correction / Niche Annotation
Currently, the system uses CLIP for general product similarity. In the future, a secondary, highly-specialized CLIP model (or contrastive learning layer) will be introduced specifically to focus on **micro-differences in packaging** (e.g., isolating the "100g" vs "250g" labels, or noticing the "Sugar Free" badge on an otherwise identical package). This niche annotation will force the model to mathematically separate highly confusing SKUs.

### 2. Temporal Video Aggregation
Shifting from static frame analysis to temporal streaming, where the engine confirms an item only after it appears securely across 10+ consecutive video frames, vastly reducing single-frame motion blur errors.

### 3. Edge Deployment Optimization
Quantizing the RT-DETR and CLIP models using ONNX or TensorRT to achieve 30+ FPS directly on warehouse edge devices (like Jetson Nano) without requiring cloud GPU compute.

---

## 🛠️ Tech Stack & AI Frameworks

* **Computer Vision:** OpenAI CLIP (ViT-B/32), Ultralytics RT-DETR, EasyOCR, Pillow, OpenCV
* **Vector Database:** FAISS (Facebook AI Similarity Search) FlatIP Inner-Product indexing
* **Backend logic & Pipeline:** Python 3.12, PyTorch
* **Frontend GUI:** Streamlit

---

## 🤝 Open for Collaboration

I am actively looking for **feedback, suggestions, and collaboration** from AI engineers, quick commerce professionals, and anyone passionate about computer vision! 

Whether you want to discuss scaling this system, tweaking the RAG mechanics, or deploying it to edge hardware, feel free to reach out.

**Contact Me:**
- 💼 LinkedIn: [linkedin.com/in/shobhit-mohadikar](https://www.linkedin.com/in/shobhit-mohadikar/)
- 📧 Email: [shobhitmohadikar@gmail.com](mailto:shobhitmohadikar@gmail.com)
---

## 💡 Credits
This POC was proudly designed and architected with the assistance of advanced AI coding tools and models:
* **Antigravity**
* **ChatGPT**
* **Claude**
