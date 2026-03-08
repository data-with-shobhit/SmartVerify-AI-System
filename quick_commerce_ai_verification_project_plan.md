# AI-Based Order Verification System for Quick Commerce (POC)

## 1. Project Overview

This project builds a multi-layer AI system to verify delivered items in
quick commerce orders using computer vision and multimodal AI. The goal
is to detect missing items, wrong products, and wrong variants before or
during delivery.

Target companies: Blinkit, Zepto, Swiggy Instamart

------------------------------------------------------------------------

## 2. Problem Statement

Quick commerce platforms frequently face issues such as: - Missing
items - Wrong product variant - Wrong SKU - Packing errors

Manual verification and barcode scanning often fail to detect
variant-level mistakes.

This system uses: - Computer Vision - Multimodal Embeddings - Reasoning
Layer - Human Override

to achieve high accuracy verification.

------------------------------------------------------------------------

## 3. POC Scope

Initial dataset will include 5 SKUs:

1.  Kurkure Masala Munch
2.  Kurkure Solid Masti
3.  Maggi 2 Minute Noodles
4.  Amul Taaza Milk
5.  Britannia Bread

The system should detect: - Correct order - Missing item - Wrong item -
Wrong variant

------------------------------------------------------------------------

## 4. System Architecture

Mobile Upload → Frame Extraction → Object Detection → OCR → CLIP
Matching → Verification Engine → Optional LLM → Result

------------------------------------------------------------------------

## 5. Dataset Requirements

For POC:

5 SKUs × 20 images each

Total dataset size: \~100 images

Image variations required: - Different lighting - Different angles -
Partial occlusion - Front packaging

Annotation format: Bounding boxes for each product.

Tools: - LabelImg - Roboflow - CVAT

------------------------------------------------------------------------

## 6. Model Components

### Object Detection

Model: YOLOv8

Purpose: Detect product categories inside the image.

Example Output: - Kurkure packet - Milk packet - Bread loaf

------------------------------------------------------------------------

### OCR Layer

Libraries: - PaddleOCR - EasyOCR

Purpose: Extract text from packaging such as: - Brand name - Flavor -
Weight

Example Output: "Kurkure Masala Munch 90g"

------------------------------------------------------------------------

### CLIP Matching

Model: OpenAI CLIP

Steps: 1. Generate embeddings for catalog images 2. Store embeddings in
vector database 3. Convert uploaded image to embedding 4. Compare
similarity

Vector DB for POC: FAISS

Example Result: Masala Munch → 0.93 similarity

------------------------------------------------------------------------

### Verification Engine

Combine multiple signals:

Final Score = 0.4 Detection Score + 0.3 OCR Score + 0.3 CLIP Similarity

Decision thresholds:

> 0.85 → Verified 0.6 -- 0.85 → Send to LLM reasoning \< 0.6 → Mismatch

------------------------------------------------------------------------

### LLM Reasoning Layer (Optional)

Models: - Phi-3 Mini - Llama 3.2 3B

Used only when confidence is low.

Purpose: Provide logical reasoning for mismatch cases.

------------------------------------------------------------------------

## 7. Catalog System

Catalog Schema:

-   sku_id
-   product_name
-   brand
-   variant
-   weight
-   image_embedding
-   text_embedding

Example:

sku_001 Kurkure Masala Munch Brand: Kurkure Weight: 90g

Embeddings stored in FAISS.

------------------------------------------------------------------------

## 8. API Design

POST /verify-order

Input: - order_id - order_items - image or video

Output: { "status": "verified", "confidence": 0.91, "detected_items":
\[\] }

Additional APIs:

POST /add-sku GET /catalog

------------------------------------------------------------------------

## 9. Tech Stack

Backend: Python FastAPI

Models: YOLOv8 CLIP PaddleOCR

Vector Database: FAISS

Storage: PostgreSQL

Optional UI: React

------------------------------------------------------------------------

## 10. Hardware Requirements

For POC:

GPU: RTX 3060 (recommended) RAM: 16GB Storage: 20GB

Inference can run on CPU if optimized.

------------------------------------------------------------------------

## 11. POC Development Phases

### Phase 1 -- Dataset Collection

Collect 100 images and annotate bounding boxes.

Estimated time: 1 day

### Phase 2 -- Detection Model Training

Train YOLOv8 model.

Estimated time: 1 day

### Phase 3 -- CLIP Embeddings

Generate embeddings for product catalog.

Estimated time: 4--5 hours

### Phase 4 -- OCR Integration

Extract packaging text and match with catalog.

Estimated time: 4 hours

### Phase 5 -- Verification Engine

Combine model outputs and implement scoring logic.

Estimated time: 1 day

### Phase 6 -- API & Demo

Create API endpoint and demo pipeline.

Estimated time: 1 day

------------------------------------------------------------------------

## 12. Evaluation Metrics

Measure system performance using:

-   Accuracy
-   Precision
-   Recall
-   False positives
-   False negatives

Target accuracy for POC: \>90%

------------------------------------------------------------------------

## 13. Demo Scenarios

Test cases:

1.  Correct order
2.  Missing item
3.  Wrong variant
4.  Extra item

Example:

Order: Kurkure Masala Munch Maggi

Delivered: Kurkure Solid Masti Maggi

Expected Output: Variant mismatch detected.

------------------------------------------------------------------------

## 14. Future Improvements

-   Video frame aggregation for higher accuracy
-   Larger SKU catalog (10k+)
-   Warehouse packing station cameras
-   Real-time driver verification
-   Active learning for continuous model improvement

------------------------------------------------------------------------

## 15. Expected Outcome

A working AI pipeline capable of: - Detecting grocery products -
Matching SKU variants - Identifying wrong deliveries - Providing
confidence scores

This POC can demonstrate the feasibility of an AI-powered verification
system for quick commerce platforms.
