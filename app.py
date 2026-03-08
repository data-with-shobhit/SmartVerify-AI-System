import os
import json
import streamlit as st
from PIL import Image
from core.logger import get_logger
from core.config import Config
from detectors.rtdetr_detector import DetectorManager
from matchers.clip_matcher import CLIPMatcher
from ocr.ocr_manager import OCRManager
from engine.verifier import VerificationEngine
from engine.reconciler import OrderReconciler
from engine.pipeline import VerificationPipeline

logger = get_logger("app")

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartVerify AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Managers Initialization (Cached) ──────────────────────────────────────────
@st.cache_resource
def get_managers(weights_path, model_path, catalog_dir, _v=1):
    detector = DetectorManager(weights_path)
    matcher  = CLIPMatcher(model_path, catalog_dir)
    ocr      = OCRManager(os.path.join(catalog_dir, "catalog.json"))
    verifier = VerificationEngine(matcher, ocr)
    reconciler = OrderReconciler()
    pipeline = VerificationPipeline(detector, verifier, reconciler)
    return pipeline

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d0f14; color: #f0f2f6; }
h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; color: #ffffff; }
.stButton>button { border-radius: 8px; font-weight: 600; }
.bill-card { background: #161a23; border: 1px solid #2d333f; border-radius: 12px; padding: 24px; margin-bottom: 12px; }
.order-line { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #2d333f; font-size: 14px; }
.status-verified  { color: #10b981; font-weight: 700; }
.status-review    { color: #f59e0b; font-weight: 700; }
.status-missing   { color: #ef4444; font-weight: 700; }
.status-pending   { color: #6b7280; font-weight: 500; }
.score-box { background: #1e2533; border: 1px solid #2d333f; border-radius: 10px; padding: 14px; margin: 8px 0; }
.score-verified { border-left: 5px solid #10b981; }
.score-review   { border-left: 5px solid #f59e0b; }
.score-mismatch { border-left: 5px solid #ef4444; }
.summary-box { background: #161a23; border: 1px solid #2d333f; border-radius: 12px; padding: 15px; text-align: center; }
.big-number { font-size: 28px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.confirmed-banner { background: linear-gradient(135deg, #065f46, #10b981); color: white; font-weight: 700; font-size: 16px; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 15px; }
.issue-banner { background: linear-gradient(135deg, #7f1d1d, #ef4444); color: white; font-weight: 700; font-size: 16px; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────
STATUS_ICON = {"VERIFIED": "✅", "REVIEW": "⚠️", "MISSING": "❌", "PARTIAL": "🔶", "PENDING": "⏳"}
STATUS_CLASS = {"VERIFIED": "status-verified", "REVIEW": "status-review", "MISSING": "status-missing", 
                "PARTIAL": "status-review", "PENDING": "status-pending"}

def render_bill(order_items, reconciliation=None):
    lines_html = ""
    for item in order_items:
        sku_id, name, qty = item["sku_id"], item["product_name"], item.get("qty", 1)
        status = "PENDING"
        if reconciliation:
            line = next((l for l in reconciliation["order_lines"] if l["sku_id"] == sku_id), None)
            if line: status = line["status"]
        icon = STATUS_ICON.get(status, "⏳")
        css_class = STATUS_CLASS.get(status, "status-pending")
        lines_html += (f'<div class="order-line"><span>{name} <b>× {qty}</b></span>'
                       f'<span class="{css_class}">{icon} {status}</span></div>')
    st.markdown(f'<div class="bill-card">{lines_html}</div>', unsafe_allow_html=True)

def render_summary(reconciliation):
    s, overall = reconciliation["summary"], reconciliation["overall_status"]
    if overall == "VERIFIED":
        st.markdown('<div class="confirmed-banner">✨ ORDER VERIFIED — Matches Expected Bill</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="issue-banner">🚨 DISCREPANCY DETECTED — Correction Needed</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, key, label, color in zip([c1, c2, c3, c4], ['verified', 'review', 'missing', 'mismatch'], 
                                     ['Verified', 'Review', 'Missing', 'Error'], ['#10b981', '#f59e0b', '#ef4444', '#ef4444']):
        with col:
            st.markdown(f"""<div class="summary-box"><div class="big-number" style="color:{color}">{s[key]}</div>
                <div style="font-size:10px; color:#9ca3af; text-transform:uppercase;">{label}</div></div>""", unsafe_allow_html=True)

def score_card(det):
    status = det.get("status", "PENDING")
    css = {"VERIFIED": "score-verified", "REVIEW": "score-review"}.get(status, "score-mismatch")
    html = (
        f'<div class="score-box {css}">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">'
        f'<b style="font-size:14px;">{STATUS_ICON.get(status,"")} {det.get("verified_as", "Unknown")}</b>'
        f'<span style="font-size:10px; color:#888;">{det.get("frame_id", "Pack")}</span></div>'
        f'<div style="display:flex; gap:8px; margin-bottom: 8px;">'
        f'<div><div style="font-size:8px; color:#888;">DET</div><b>{det.get("det_confidence",0):.2f}</b></div>'
        f'<div><div style="font-size:8px; color:#888;">CLIP</div><b>{det.get("clip_score",0):.2f}</b></div>'
        f'<div><div style="font-size:8px; color:#888;">OCR</div><b>{det.get("ocr_score",0):.2f}</b></div>'
        f'<div><div style="font-size:8px; color:#888;">FINAL</div><b style="color:#10b981;">{det.get("final_score",0):.2f}</b></div></div>'
        f'<div style="background: rgba(0,0,0,0.2); padding: 6px; border-radius: 4px; color:#9ca3af; font-size:11px;">'
        f'Read: <i>{det.get("extracted_text", "")[:50] or "(No text read)"}</i>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Build Configuration")
    model_path     = st.text_input("CLIP Model Path", value=Config.CLIP_MODEL_PATH)
    rtdetr_weights = st.text_input("RT-DETR Model Weights", value=Config.RTDETR_WEIGHTS)
    catalog_dir    = st.text_input("Catalog Directory", value=Config.CATALOG_DIR)
    conf_threshold = st.slider("Detection confidence", 0.1, 0.9, Config.DEFAULT_DET_CONF, 0.05)

    st.markdown("---")
    st.markdown("### 🎛️ Processing Mode")
    PRESETS = {
        "🧠 Intelligent Mode (RT-DETR + CLIP + OCR)": {"det": True, "clip": True, "ocr": True},
        "⚙️ Custom Mode": {"det": None, "clip": None, "ocr": None},
    }
    
    preset = st.selectbox("Preset mode", list(PRESETS.keys()))
    chosen = PRESETS[preset]
    
    if chosen["det"] is None:
        st.markdown("**Active layers (Select max 2):**")
        use_detection = st.checkbox("Detection", value=True)
        use_clip      = st.checkbox("CLIP",      value=True)
        use_ocr       = st.checkbox("OCR",       value=False)
        if use_detection and use_clip and use_ocr:
            st.error("⚠️ Please select only 2 layers for Custom Mode.")
    else:
        use_detection, use_clip, use_ocr = True, True, True
        st.info("Full Ensemble Enabled: RT-DETR + CLIP + OCR")
    
    st.markdown("### 🎚️ Thresholds")
    verified_thresh = st.slider("Verified", 0.7, 1.0, Config.VERIFIED_THRESHOLD, 0.01)
    review_thresh   = st.slider("Review",   0.4, 0.8, Config.REVIEW_THRESHOLD, 0.01)

# ── Initialize Modular Engine ──────────────────────────────────────────────────
pipeline = get_managers(rtdetr_weights, model_path, catalog_dir, 1)

# ── main UI ─────────────────────────────────────────────────────────────────────
st.markdown("# 📦 SmartVerify AI: Order Reconciliation")
st.markdown("""
**Triple-Layer Verification Engine:** Ensemble pipeline combining **RT-DETR**, **CLIP**, and **EasyOCR** 
to automate multi-product order reconciliation.
""")
st.markdown("---")

t_input, t_output = st.columns([1, 1], gap="large")

with t_input:
    st.markdown("### 🛒 1. Expected Order")
    with open(Config.CATALOG_JSON, encoding="utf-8") as f:
        catalog_data = json.load(f)
    sku_options = {s["product_name"]: s["sku_id"] for s in catalog_data}
    if "order_items" not in st.session_state: st.session_state.order_items = []
    
    col_s, col_q = st.columns([3, 1])
    sel_name = col_s.selectbox("Sku", list(sku_options.keys()), label_visibility="collapsed")
    sel_qty  = col_q.number_input("Qty", min_value=1, value=1, label_visibility="collapsed")
    if st.button("➕ Add to Order", use_container_width=True):
        st.session_state.order_items.append({"sku_id": sku_options[sel_name], "product_name": sel_name, "qty": sel_qty})
    
    render_bill(st.session_state.order_items, st.session_state.get("reconciliation"))
    if st.button("🗑️ Reset All", width="stretch"):
        for k in ["order_items", "reconciliation", "all_verifications", "annotated_frames"]:
            if k in st.session_state: del st.session_state[k]
        st.rerun()

with t_output:
    st.markdown("### 📸 2. Verification")
    uploaded = st.file_uploader("Evidence", type=["jpg", "png"], accept_multiple_files=True, label_visibility="collapsed")
    if st.button("🚀 Verify Order", type="primary", width="stretch", disabled=not (st.session_state.order_items and uploaded)):
        with st.status("🔍 Analyzing Physical Items...") as status:
            images_info = [(f.name, Image.open(f).convert("RGB")) for f in uploaded]
            config_dict = {
                "conf_threshold": conf_threshold,
                "use_detection": use_detection,
                "use_clip": use_clip,
                "use_ocr": use_ocr,
                "verified_thresh": verified_thresh,
                "review_thresh": review_thresh
            }
            results = pipeline.process_order(
                images_info=images_info,
                expected_order=st.session_state.order_items,
                config=config_dict,
                ui_callback=status.write
            )
            status.update(label="✅ Verification Done!", state="complete")
        
        st.session_state.reconciliation = results["reconciliation"]
        st.session_state.all_verifications = results["all_verifications"]
        st.session_state.annotated_frames = results["annotated_frames"]
        st.rerun()

st.markdown("---")
res_l, res_r = st.columns([1.2, 1.8], gap="large")
with res_l:
    st.markdown("### 🎥 Annotated Feed")
    if st.session_state.get("annotated_frames"):
        for frame in st.session_state.annotated_frames: st.image(frame)
    else: st.info("Analysis feed will appear here.")
with res_r:
    st.markdown("### 📊 Dashboard")
    if st.session_state.get("reconciliation"):
        render_summary(st.session_state.reconciliation)
        t1, t2 = st.tabs(["📑 Logs", "✋ Actions"])
        with t1:
            for v in reversed(st.session_state.all_verifications): score_card(v)
    else: st.info("Results pending.")