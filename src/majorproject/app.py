# app.py - Impressive UI for Leaf Health AI
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from load_data import calculate_indices

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="LeafHealth AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1f0d 50%, #0a0f1e 100%);
    }

    /* Hide default header */
    header[data-testid="stHeader"] { background: transparent; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d2010 0%, #0a1a0a 100%);
        border-right: 1px solid #1a4a1a;
    }
    [data-testid="stSidebar"] * { color: #a8d8a8 !important; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0d4a1a 0%, #1a6e2e 40%, #0d4a1a 100%);
        border: 1px solid #2d8a3e;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 0 40px rgba(45, 138, 62, 0.3);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        text-shadow: 0 0 20px rgba(100, 255, 100, 0.5);
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #a8d8a8;
        margin-top: 10px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(45, 138, 62, 0.3);
        border: 1px solid #2d8a3e;
        border-radius: 20px;
        padding: 5px 15px;
        font-size: 0.8rem;
        color: #64ff64;
        margin: 5px;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0d2a10 0%, #0a1f0a 100%);
        border: 1px solid #2d5a1e;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #64ff64;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #7ab87a;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Status cards */
    .status-healthy {
        background: linear-gradient(135deg, #0d3a1a, #1a5c2a);
        border: 2px solid #2d8a3e;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
    }
    .status-moderate {
        background: linear-gradient(135deg, #3a2a00, #5c4400);
        border: 2px solid #d4a017;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
    }
    .status-critical {
        background: linear-gradient(135deg, #3a0a0a, #5c1010);
        border: 2px solid #d43f3f;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
    }
    .status-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }

    /* Index cards */
    .index-card {
        background: linear-gradient(135deg, #0a1a2e, #0d2040);
        border: 1px solid #1a3a6e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .index-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #64b4ff;
    }
    .index-name {
        font-size: 0.9rem;
        color: #7ab4d8;
        font-weight: 600;
    }
    .index-desc {
        font-size: 0.75rem;
        color: #5a8aaa;
        margin-top: 5px;
    }

    /* Recommendation box */
    .rec-box {
        background: linear-gradient(135deg, #1a1a0d, #2a2a10);
        border-left: 4px solid #d4d417;
        border-radius: 0 12px 12px 0;
        padding: 20px 25px;
        margin: 10px 0;
    }
    .rec-title {
        font-size: 0.8rem;
        color: #d4d417;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    .rec-text {
        font-size: 1rem;
        color: #e8e8c8;
        line-height: 1.6;
    }

    /* Progress bar custom */
    .efficiency-bar-container {
        background: #0a1a0a;
        border-radius: 50px;
        height: 25px;
        border: 1px solid #2d5a1e;
        overflow: hidden;
        margin: 10px 0;
    }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #0d2010, #0a1a0a);
        border: 2px dashed #2d8a3e;
        border-radius: 15px;
        padding: 20px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #64ff64;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid #2d5a1e;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Info pills */
    .pill {
        display: inline-block;
        background: rgba(45, 138, 62, 0.2);
        border: 1px solid #2d8a3e;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        color: #64ff64;
        margin: 3px;
    }

    /* All text color fix */
    .stMarkdown, p, span, div { color: #c8e8c8; }
    h1, h2, h3 { color: #ffffff !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0f1e; }
    ::-webkit-scrollbar-thumb { background: #2d8a3e; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Load model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "final_model.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()


# ── Helpers ──────────────────────────────────────────────
def preprocess(img_rgb):
    img = cv2.resize(img_rgb, (128, 128))
    img = img.astype(np.float32) / 255.0
    indices = calculate_indices(img)
    img_6ch = np.concatenate([img, indices], axis=-1)
    return np.expand_dims(img_6ch, axis=0)

def get_health_info(eff_pct):
    if eff_pct >= 80:
        return {
            "status": "Healthy",
            "emoji": "🌿",
            "color": "healthy",
            "observation_en": "Plant is in optimal condition with strong chlorophyll activity.",
            "action_en": "Maintain current irrigation and fertilizer schedule. Monitor weekly.",
            "observation_hi": "पौधा उत्तम स्थिति में है, क्लोरोफिल गतिविधि मजबूत है।",
            "action_hi": "वर्तमान सिंचाई और उर्वरक कार्यक्रम बनाए रखें। साप्ताहिक निगरानी करें।",
            "bar_color": "#2d8a3e"
        }
    elif eff_pct >= 50:
        return {
            "status": "Moderate Stress",
            "emoji": "⚠️",
            "color": "moderate",
            "observation_en": "Early stress detected — 7 to 14 days before visible symptoms appear.",
            "action_en": "Check soil moisture. Apply balanced NPK fertilizer. Increase watering by 20%.",
            "observation_hi": "प्रारंभिक तनाव पाया गया — दृश्य लक्षणों से 7-14 दिन पहले।",
            "action_hi": "मिट्टी की नमी जांचें। संतुलित NPK उर्वरक लगाएं। पानी 20% बढ़ाएं।",
            "bar_color": "#d4a017"
        }
    else:
        return {
            "status": "Critical",
            "emoji": "🚨",
            "color": "critical",
            "observation_en": "Severe stress or disease detected. Immediate intervention required.",
            "action_en": "Consult agronomist immediately. Apply targeted treatment. Check for pest/disease.",
            "observation_hi": "गंभीर तनाव या रोग पाया गया। तत्काल हस्तक्षेप आवश्यक है।",
            "action_hi": "तुरंत कृषि विशेषज्ञ से संपर्क करें। लक्षित उपचार करें। कीट/रोग की जांच करें।",
            "bar_color": "#d43f3f"
        }


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 LeafHealth AI")
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    Deep learning system for estimating **photosynthetic efficiency** from leaf images.
    """)
    st.markdown("---")
    st.markdown("### 🔬 Indices Used")
    st.markdown("🟢 **VARI** — Chlorophyll proxy")
    st.markdown("🟡 **ExG** — Greenness measure")
    st.markdown("🔵 **MGRVI** — Stress indicator")
    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    st.markdown("- Architecture: **Custom CNN**")
    st.markdown("- Input: **128×128×6**")
    st.markdown("- Dataset: **PlantVillage + OLID**")
    st.markdown("- R² Score: **0.62**")
    st.markdown("---")
    st.markdown("### 🎯 Health Scale")
    st.markdown("🌿 **80-100%** → Healthy")
    st.markdown("⚠️ **50-79%** → Moderate")
    st.markdown("🚨 **0-49%** → Critical")
    st.markdown("---")

    # Language toggle
    lang = st.radio("🌐 Language / भाषा",
                    ["English", "हिंदी (Hindi)"],
                    index=0)
    st.markdown("---")
    st.markdown("<small style='color:#4a7a4a'>ST. Vincent Pallotti COE&T<br>Dept. of CS (Data Science)<br>2026-27</small>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🌿 LeafHealth AI</p>
    <p class="hero-subtitle">Smartphone RGB-based Photosynthetic Efficiency Prediction System</p>
    <span class="hero-badge">🧠 Deep Learning</span>
    <span class="hero-badge">🔬 Vegetation Indices</span>
    <span class="hero-badge">🌾 Precision Agriculture</span>
    <span class="hero-badge">📱 RGB-Based</span>
</div>
""", unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────
st.markdown('<p class="section-header">📤 Upload Leaf Image</p>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & drop or click to upload a leaf image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    # Landing info
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">6CH</div>
            <div class="metric-label">Input Channels</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">128²</div>
            <div class="metric-label">Image Size</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">0.62</div>
            <div class="metric-label">R² Score</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">25ms</div>
            <div class="metric-label">Per Leaf</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; padding: 40px; color: #4a7a4a;'>
        <p style='font-size:3rem'>🍃</p>
        <p style='font-size:1.1rem; color:#5a8a5a'>Upload a leaf image above to begin analysis</p>
        <p style='font-size:0.85rem; color:#3a6a3a'>Supports JPG, JPEG, PNG formats</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Process image ─────────────────────────────────────
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Animated progress
    with st.spinner(""):
        progress = st.progress(0, text="🔬 Preprocessing image...")
        time.sleep(0.3)
        progress.progress(30, text="🧪 Computing vegetation indices...")
        processed = preprocess(img_array)
        time.sleep(0.3)
        progress.progress(60, text="🧠 Running CNN model...")
        prediction = model.predict(processed, verbose=0)[0][0]
        time.sleep(0.3)
        progress.progress(90, text="📊 Generating report...")
        efficiency_pct = round(float(prediction) * 100, 1)
        info = get_health_info(efficiency_pct)
        time.sleep(0.2)
        progress.progress(100, text="✅ Analysis complete!")
        time.sleep(0.3)
        progress.empty()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top Results Row ───────────────────────────────────
    left, right = st.columns([1, 1.2])

    with left:
        st.markdown('<p class="section-header">📷 Leaf Image</p>',
                    unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with right:
        st.markdown('<p class="section-header">📊 Analysis Results</p>',
                    unsafe_allow_html=True)

        # Big efficiency display
        st.markdown(f"""
        <div style='text-align:center; padding: 20px; background: linear-gradient(135deg, #0d2a10, #0a1f0a);
                    border-radius: 15px; border: 1px solid {info["bar_color"]}; margin-bottom: 15px;'>
            <p style='font-size:0.85rem; color:#7ab87a; text-transform:uppercase;
                      letter-spacing:2px; margin:0'>Photosynthetic Efficiency</p>
            <p style='font-size:4rem; font-weight:700; color:{info["bar_color"]};
                      margin:5px 0; text-shadow: 0 0 20px {info["bar_color"]}88'>{efficiency_pct}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Efficiency bar
        bar_width = efficiency_pct
        st.markdown(f"""
        <div class="efficiency-bar-container">
            <div style='height:100%; width:{bar_width}%;
                        background: linear-gradient(90deg, {info["bar_color"]}88, {info["bar_color"]});
                        border-radius: 50px; transition: width 1s ease;
                        display:flex; align-items:center; justify-content:center;'>
                <span style='color:white; font-size:0.8rem; font-weight:600'>{efficiency_pct}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Status badge
        st.markdown(f"""
        <div class="status-{info['color']}" style='margin-top:15px'>
            <p class="status-title">{info['emoji']} {info['status']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Vegetation Indices ────────────────────────────────
    st.markdown('<p class="section-header">🧪 Vegetation Indices</p>',
                unsafe_allow_html=True)

    img_norm = cv2.resize(img_array, (128, 128)).astype(np.float32) / 255.0
    R = img_norm[:,:,0]; G = img_norm[:,:,1]; B = img_norm[:,:,2]

    vari  = float(np.mean(np.clip((G-R)/(G+R-B+1e-5), -1, 1)))
    exg   = float(np.mean(np.clip(2*G-R-B, 0, 1)))
    mgrvi = float(np.mean(np.clip((G**2-R**2)/(G**2+R**2+1e-5), -1, 1)))

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(f"""<div class="index-card">
            <div class="index-name">VARI</div>
            <div class="index-value">{vari:.4f}</div>
            <div class="index-desc">Visible Atmospherically<br>Resistant Index</div>
            <div class="index-desc" style='color:#4a7a9a; margin-top:5px'>
                {'🟢 Good' if vari > 0.1 else '🟡 Moderate' if vari > 0 else '🔴 Low'}</div>
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown(f"""<div class="index-card">
            <div class="index-name">ExG</div>
            <div class="index-value">{exg:.4f}</div>
            <div class="index-desc">Excess Green<br>Index</div>
            <div class="index-desc" style='color:#4a7a9a; margin-top:5px'>
                {'🟢 High' if exg > 0.15 else '🟡 Medium' if exg > 0.05 else '🔴 Low'}</div>
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown(f"""<div class="index-card">
            <div class="index-name">MGRVI</div>
            <div class="index-value">{mgrvi:.4f}</div>
            <div class="index-desc">Modified Green Red<br>Vegetation Index</div>
            <div class="index-desc" style='color:#4a7a9a; margin-top:5px'>
                {'🟢 Positive' if mgrvi > 0.1 else '🟡 Low' if mgrvi > 0 else '🔴 Negative'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────
    st.markdown('<p class="section-header">💡 Recommendations</p>',
                unsafe_allow_html=True)

    if lang == "English":
        st.markdown(f"""
        <div class="rec-box">
            <div class="rec-title">🔍 Observation</div>
            <div class="rec-text">{info['observation_en']}</div>
        </div>
        <div class="rec-box" style='border-left-color:#64ff64'>
            <div class="rec-title" style='color:#64ff64'>✅ Recommended Action</div>
            <div class="rec-text">{info['action_en']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="rec-box">
            <div class="rec-title">🔍 अवलोकन</div>
            <div class="rec-text">{info['observation_hi']}</div>
        </div>
        <div class="rec-box" style='border-left-color:#64ff64'>
            <div class="rec-title" style='color:#64ff64'>✅ अनुशंसित कार्रवाई</div>
            <div class="rec-text">{info['action_hi']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Training Graphs ───────────────────────────────────
    st.markdown('<p class="section-header">📈 Model Training Graphs</p>',
                unsafe_allow_html=True)

    base_dir = os.path.dirname(__file__)
    g1, g2 = st.columns(2)
    train_graph = os.path.join(base_dir, "training_graph.png")
    pred_graph  = os.path.join(base_dir, "prediction_graph.png")

    if os.path.exists(train_graph):
        g1.image(train_graph, caption="Training & Validation Loss",
                 use_container_width=True)
    if os.path.exists(pred_graph):
        g2.image(pred_graph, caption="Actual vs Predicted Efficiency",
                 use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Footer summary ────────────────────────────────────
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #0d2a10, #0a1f0a);
                border: 1px solid #2d5a1e; border-radius: 15px;
                padding: 20px; text-align: center;'>
        <span class="pill">Efficiency: {efficiency_pct}%</span>
        <span class="pill">Status: {info['status']}</span>
        <span class="pill">VARI: {vari:.3f}</span>
        <span class="pill">ExG: {exg:.3f}</span>
        <span class="pill">MGRVI: {mgrvi:.3f}</span>
    </div>
    """, unsafe_allow_html=True)