"""
Image Enhancement & Noise Reduction
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from skimage.metrics import structural_similarity as _ssim_metric

from utils import load_image, add_gaussian_noise, add_salt_pepper_noise, save_image, normalize_image, denormalize_image, estimate_noise_level
from filters import (average_filter, gaussian_blur, median_filter_cv,
                     bilateral_filter, sharpening_filter,
                     morphological_opening, morphological_closing)
from enhancement import (histogram_equalization, clahe_enhancement,
                         adaptive_histogram_equalization, contrast_stretching, gamma_correction)
from cnn_denoise import load_keras_cnn_model, denoise_image_keras_cnn

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"


def compute_psnr(original, denoised):
    original = np.asarray(original, dtype=np.float32)
    denoised = np.asarray(denoised, dtype=np.float32)

    if original.shape != denoised.shape:
        if original.ndim != denoised.ndim:
            if denoised.ndim == 2 and original.ndim == 3:
                denoised = cv2.cvtColor(np.uint8(denoised), cv2.COLOR_GRAY2BGR)
                denoised = denoised[:, :, 0]
            elif original.ndim == 2 and denoised.ndim == 3:
                original = original[:, :, 0]
        if original.shape[:2] != denoised.shape[:2]:
            denoised = cv2.resize(denoised, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)

    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(original, denoised):
    """Structural Similarity Index (SSIM) between two grayscale images.

    Returns a float in [0, 1]; higher means more structurally similar.
    """
    original = np.asarray(original, dtype=np.uint8)
    denoised = np.asarray(denoised, dtype=np.uint8)

    # Align shapes the same way compute_psnr does.
    if original.shape != denoised.shape:
        if original.ndim != denoised.ndim:
            if original.ndim == 3:
                original = original[:, :, 0]
            else:
                denoised = denoised[:, :, 0]
        if original.shape[:2] != denoised.shape[:2]:
            denoised = cv2.resize(denoised, (original.shape[1], original.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

    try:
        return float(_ssim_metric(original, denoised, data_range=255))
    except Exception:
        # e.g. images too small for the default SSIM window
        return float('nan')


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
  page_title="Image Enhancement",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --bg:        #f4f6fa;
  --surface:   #ffffff;
  --border:    #e2e8f0;
  --text:      #0f172a;
  --muted:     #475569;
  --faint:     #94a3b8;
  --accent:    #1d4ed8;
  --accent-dark:#1e40af;
  --accent-soft:#eff6ff;
  --teal:      #0f766e;
  --teal-soft: #f0fdfa;
  --slate:     #334155;
  --slate-soft:#f8fafc;
  --ff:        'Inter', -apple-system, 'Segoe UI', sans-serif;
  --mono:      'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
  --r8:8px; --r12:12px; --r16:16px;
}

*{ font-family: var(--ff); }
html, body, [class*="css"]{ background: var(--bg) !important; }
.main{ background: var(--bg) !important; }
.block-container{ padding: 2.5rem 2.5rem 5rem !important; max-width: 1200px !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"]{ background: var(--surface) !important; border-right: 1px solid var(--border) !important; min-width: 280px !important; }
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] > div { padding: 0 !important; }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] small { color: var(--muted) !important; font-size: 0.82rem !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 { color: var(--text) !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div { background: var(--surface) !important; border-color: #cbd5e1 !important; color: var(--text) !important; border-radius: var(--r8) !important; font-size: 0.82rem !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"]{ background: var(--slate-soft) !important; border: 1px dashed #cbd5e1 !important; border-radius: var(--r12) !important; padding: 12px !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * { color: var(--faint) !important; }

/* ── Buttons ── */
.stButton > button{
  font-family: var(--ff) !important; font-weight: 600 !important; font-size: 0.82rem !important;
  border: none !important; border-radius: var(--r8) !important; padding: 10px 20px !important;
  background: var(--accent) !important; color: #fff !important;
  transition: background 0.15s ease !important; cursor: pointer !important;
}
.stButton > button:hover{ background: var(--accent-dark) !important; }
section[data-testid="stSidebar"] .stButton > button{ width: 100% !important; }
.main .stButton > button{ padding: 11px 28px !important; }

/* ── Hero ── */
.pf-hero{
  background: var(--surface);
  border: 1px solid var(--border); border-left: 4px solid var(--accent);
  border-radius: var(--r16);
  padding: 40px 44px; margin-bottom: 36px;
  display: flex; align-items: center; justify-content: space-between; gap: 40px;
  box-shadow: 0 1px 3px rgba(15,23,42,.06);
}
.pf-hero-left{ }
.pf-hero-tag{
  display: inline-block;
  background: var(--accent-soft); border: 1px solid #bfdbfe;
  border-radius: 100px; padding: 4px 12px;
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--accent-dark); margin-bottom: 14px;
}
.pf-hero h1{ font-size: 2rem; font-weight: 800; color: var(--text); margin: 0 0 10px; line-height: 1.2; letter-spacing: -0.5px; }
.pf-hero h1 em{ font-style: normal; color: var(--accent); }
.pf-hero p{ font-size: 0.95rem; font-weight: 400; color: var(--muted); margin: 0; max-width: 520px; line-height: 1.65; }
.pf-hero-right{ display: flex; gap: 28px; }
.pf-stat{ text-align: right; min-width: 80px; }
.pf-stat-num{ font-size: 1.9rem; font-weight: 800; color: var(--text); letter-spacing: -1px; line-height: 1; }
.pf-stat-label{ font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--faint); margin-top: 6px; }

/* ── Section Header ── */
.pf-section{ display: flex; align-items: center; gap: 12px; margin: 40px 0 20px; }
.pf-section-pill{
  background: var(--accent); color: #fff;
  font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  padding: 4px 10px; border-radius: 6px; white-space: nowrap;
}
.pf-section h2{ font-size: 1.1rem; font-weight: 700; color: var(--text); margin: 0; letter-spacing: -0.2px; white-space: nowrap; }
.pf-section-rule{ flex: 1; height: 1px; background: var(--border); }

/* ── Image Card ── */
.pf-card{
  background: var(--surface); border-radius: var(--r12); padding: 16px;
  border: 1px solid var(--border);
  box-shadow: 0 1px 3px rgba(15,23,42,.06);
}
.pf-card-header{ display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.pf-card-dot{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.pf-card-title{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }

/* ── Metric Cards ── */
.pf-metrics{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin: 24px 0 8px; }
.pf-metric{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--r12); padding: 22px 24px; box-shadow: 0 1px 3px rgba(15,23,42,.06); border-top: 3px solid var(--accent); }
.pf-metric.blue{ border-top-color: var(--accent); }
.pf-metric.teal{ border-top-color: var(--teal); }
.pf-metric.violet{ border-top-color: #6d28d9; }
.pf-metric::after{ display: none; }
.pf-metric-label{ font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--faint); margin-bottom: 8px; }
.pf-metric-value{ font-size: 2.2rem; font-weight: 800; color: var(--text); line-height: 1; letter-spacing: -1px; }
.pf-metric-unit{ font-size: 0.95rem; font-weight: 500; color: var(--faint); letter-spacing: 0; }
.pf-metric-sub{ font-size: 0.78rem; color: var(--faint); margin-top: 6px; }

/* ── Alerts ── */
.pf-alert{ border-radius: var(--r8); padding: 14px 18px; margin: 12px 0; display: flex; gap: 10px; align-items: flex-start; font-size: 0.85rem; line-height: 1.6; }
.pf-alert.info{ background: var(--accent-soft); border: 1px solid #bfdbfe; color: #1e40af; }
.pf-alert.success{ background: var(--teal-soft); border: 1px solid #99f6e4; color: #115e59; }
.pf-alert strong{ font-weight: 700; display: block; margin-bottom: 2px; }

/* ── About ── */
.pf-about{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin: 24px 0; }
.pf-about-card{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--r12); padding: 24px; box-shadow: 0 1px 3px rgba(15,23,42,.06); }
.pf-about-icon{ width: 40px; height: 40px; border-radius: var(--r8); display: flex; align-items: center; justify-content: center; font-size: 1.2rem; margin-bottom: 14px; }
.pf-about-icon.blue   { background: var(--accent-soft); }
.pf-about-icon.teal   { background: var(--teal-soft); }
.pf-about-icon.violet { background: #f5f3ff; }
.pf-about-card h3{ font-size: 0.92rem; font-weight: 700; color: var(--text); margin: 0 0 8px; }
.pf-about-card p{ font-size: 0.82rem; color: var(--muted); line-height: 1.7; margin: 0; }
.pf-tags{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
.pf-tag{
  background: var(--slate-soft); border: 1px solid var(--border);
  color: var(--muted); font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
  padding: 4px 10px; border-radius: 100px; font-family: var(--mono);
}

/* ── Save ── */
.pf-save-item{ display: flex; align-items: center; gap: 10px; padding: 9px 0; border-bottom: 1px solid var(--border); font-size: 0.82rem; color: var(--muted); font-family: var(--mono); }
.pf-save-item:last-child{ border-bottom: none; }
.pf-save-check{ background: var(--teal-soft); color: var(--teal); font-weight: 700; font-size: 0.7rem; width: 20px; height: 20px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; }

/* ── Sidebar structure ── */
.sb-brand{ padding: 24px 20px 18px; border-bottom: 1px solid var(--border); margin-bottom: 8px; }
.sb-brand-name{ font-size: 1.05rem; font-weight: 800; color: var(--text); letter-spacing: -0.3px; }
.sb-brand-sub{ font-size: 0.7rem; color: var(--faint); font-weight: 500; letter-spacing: 0.04em; margin-top: 2px; }
.sb-label{ font-size: 0.62rem !important; font-weight: 700 !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; color: var(--faint) !important; padding: 16px 20px 8px !important; display: block; border-top: 1px solid var(--border); margin-top: 4px; }

/* ── Streamlit native elements ── */
[data-testid="stMetric"]{ background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--r8) !important; padding: 12px 16px !important; }
[data-testid="stMetric"] [data-testid="stMetricLabel"], [data-testid="stMetricLabel"]{ color: var(--faint) !important; }
[data-testid="stMetric"] [data-testid="stMetricLabel"] *{ color: var(--faint) !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"], [data-testid="stMetricValue"]{ color: var(--text) !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] *{ color: var(--text) !important; }
[data-testid="stDataFrame"]{ border: 1px solid var(--border); border-radius: var(--r8); overflow: hidden; background: var(--surface); }
[data-testid="stCaptionContainer"]{ color: var(--faint) !important; }
[data-testid="stAlert"]{ border-radius: var(--r8) !important; }
[data-testid="stImage"] img{ border-radius: var(--r8) !important; border: 1px solid var(--border); }

/* ── Global ── */
#MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}
[data-testid="column"]{padding: 0 8px !important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for k, v in [
    ('original_image', None), ('noisy_image', None),
    ('filtered_images', {}),  ('cnn_denoised', None),
    ('keras_cnn_model', None),
    ('noise_params', None),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
sb = st.sidebar

sb.markdown("""
<div class="sb-brand">
  <div class="sb-brand-name">🔬 Image Enhancement</div>
  <div class="sb-brand-sub">Image Enhancement Studio</div>
</div>
""", unsafe_allow_html=True)

workflow_mode = sb.radio(
    "Workflow",
    ["Synthetic Noise Pipeline", "Upload Noisy Image Directly"],
    index=0,
    horizontal=False,
)

if workflow_mode == "Synthetic Noise Pipeline":
    sb.markdown('<span class="sb-label">01 · Upload Image</span>', unsafe_allow_html=True)
    uploaded_file = sb.file_uploader("Choose JPG or PNG", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded_file:
        try:
            st.session_state.original_image = load_image(uploaded_file)
            sb.success("Image loaded successfully.")
        except Exception as e:
            sb.error(f"Error: {e}")

    sb.markdown('<span class="sb-label">02 · Add Noise</span>', unsafe_allow_html=True)
    noise_type = sb.selectbox("Type", ["Gaussian Noise", "Salt & Pepper Noise"], label_visibility="collapsed")
    if noise_type == "Gaussian Noise":
        noise_std = sb.slider("Std Dev", 5, 50, 25)
        noise_params = {"type": "gaussian", "std": noise_std}
    else:
        salt_prob   = sb.slider("Salt %",   0.01, 0.10, 0.05, 0.01)
        pepper_prob = sb.slider("Pepper %", 0.01, 0.10, 0.05, 0.01)
        noise_params = {"type": "salt_pepper", "salt_prob": salt_prob, "pepper_prob": pepper_prob}

    if sb.button("Generate Noisy Image", key="gen_noise"):
        if st.session_state.original_image is not None:
            st.session_state.noise_params = noise_params
            if noise_params["type"] == "gaussian":
                st.session_state.noisy_image = add_gaussian_noise(st.session_state.original_image, std=noise_params["std"])
            else:
                st.session_state.noisy_image = add_salt_pepper_noise(st.session_state.original_image,
                    salt_prob=noise_params["salt_prob"], pepper_prob=noise_params["pepper_prob"])
            sb.success("Noisy image generated.")
        else:
            sb.warning("Upload an image first.")
    noise_type_label = noise_type
else:
    sb.markdown('<span class="sb-label">01 · Upload Noisy Image</span>', unsafe_allow_html=True)
    direct_noisy_file = sb.file_uploader("Choose JPG or PNG", type=["jpg","jpeg","png"], key="direct_noisy_file", label_visibility="collapsed")
    if direct_noisy_file:
        try:
            st.session_state.noisy_image = load_image(direct_noisy_file)
            st.session_state.original_image = None
            st.session_state.noise_params = {"type": "direct", "note": "Uploaded noisy image"}
            sb.success("Noisy image loaded successfully.")
        except Exception as e:
            sb.error(f"Error: {e}")

    sb.markdown('<span class="sb-label">02 · Clean Reference (Required for PSNR)</span>', unsafe_allow_html=True)
    reference_file = sb.file_uploader("Upload clean/original image for PSNR", type=["jpg","jpeg","png"], key="direct_reference_file", label_visibility="collapsed")
    if reference_file:
        try:
            st.session_state.original_image = load_image(reference_file)
            sb.success("Reference image loaded for PSNR comparison.")
        except Exception as e:
            sb.error(f"Error: {e}")

    if sb.button("Use Direct Noisy Image", key="use_direct_noisy"):
        if st.session_state.noisy_image is not None:
            # Auto-detect the noise type/level so the matching CNN model is used.
            # End users can't be expected to know the noise parameters of an upload.
            try:
                ntype, nparam = estimate_noise_level(st.session_state.noisy_image)
                if ntype == "gaussian":
                    st.session_state.noise_params = {"type": "gaussian", "std": float(nparam), "estimated": True}
                    sb.info(f"Detected Gaussian noise (σ ≈ {nparam:.0f}) — matching CNN model will be used.")
                else:
                    st.session_state.noise_params = {
                        "type": "salt_pepper",
                        "salt_prob": float(nparam) / 2, "pepper_prob": float(nparam) / 2,
                        "estimated": True,
                    }
                    sb.info("Detected Salt & Pepper noise — matching CNN model will be used.")
            except Exception as e:
                st.session_state.noise_params = {"type": "direct", "note": "Uploaded noisy image"}
                sb.warning(f"Could not estimate noise ({e}); using the default model.")
            sb.success("Direct noisy image is ready for processing.")
        else:
            sb.warning("Upload a noisy image first.")
    _ni = st.session_state.noise_params
    if _ni and _ni.get("estimated"):
        noise_type_label = (f"Auto-detected noise (σ≈{_ni['std']:.0f})" if _ni["type"] == "gaussian"
                            else "Auto-detected Salt & Pepper noise")
    else:
        noise_type_label = "Uploaded noisy image"

sb.markdown('<span class="sb-label">03 · Digital Filters</span>', unsafe_allow_html=True)
filter_options = {
    "Average Filter":        "average",
    "Gaussian Blur":         "gaussian",
    "Median Filter":         "median",
    "Bilateral Filter":      "bilateral",
    "Sharpening":            "sharpening",
    "Morphological Opening": "morph_open",
    "Morphological Closing": "morph_close",
}
selected_filters = sb.multiselect("Filters", list(filter_options.keys()),
    default=["Gaussian Blur","Median Filter"], label_visibility="collapsed")

sb.markdown('<span class="sb-label">04 · Enhancement</span>', unsafe_allow_html=True)
enhancement_options = {
    "Histogram Equalization": "hist_eq",
    "CLAHE":                  "clahe",
    "Adaptive Hist. Equal.":  "ahe",
    "Contrast Stretching":    "contrast",
    "Gamma Correction":       "gamma",
}
selected_enhancements = sb.multiselect("Methods", list(enhancement_options.keys()),
    default=["CLAHE"], label_visibility="collapsed")

sb.markdown('<span class="sb-label">05 · Deep Learning</span>', unsafe_allow_html=True)
use_cnn = sb.checkbox("Enable CNN Denoising", value=True)

sb.markdown('<span class="sb-label" style="padding-bottom:14px;"></span>', unsafe_allow_html=True)

if sb.button("⚡  Run Full Pipeline", key="apply_all", use_container_width=True):
    if st.session_state.noisy_image is not None:
        st.session_state.filtered_images = {}
        for fname, fkey in filter_options.items():
            if fname in selected_filters:
                try:
                    m = {"average": average_filter, "gaussian": gaussian_blur, "median": median_filter_cv,
                         "bilateral": bilateral_filter, "sharpening": sharpening_filter,
                         "morph_open": morphological_opening, "morph_close": morphological_closing}
                    st.session_state.filtered_images[fname] = m[fkey](st.session_state.noisy_image)
                except Exception as e:
                    sb.warning(f"{fname}: {e}")

        base = st.session_state.filtered_images.get(
            selected_filters[0] if selected_filters else None,
            st.session_state.noisy_image)
        for ename, ekey in enhancement_options.items():
            if ename in selected_enhancements:
                try:
                    em = {"hist_eq": histogram_equalization, "clahe": clahe_enhancement,
                          "ahe": adaptive_histogram_equalization,
                          "contrast": contrast_stretching, "gamma": lambda i: gamma_correction(i, gamma=0.8)}
                    st.session_state.filtered_images[ename] = em[ekey](base)
                except Exception as e:
                    sb.warning(f"{ename}: {e}")

        if use_cnn:
            try:
                ni = st.session_state.noise_params
                if ni and ni["type"] == "gaussian":
                    s = ni["std"]
                    mp = MODEL_DIR / ('cnn_denoiser_sigma15.h5' if s <= 20 else ('cnn_denoiser_sigma25.h5' if s <= 30 else 'cnn_denoiser_sigma35.h5'))
                elif ni and ni["type"] == "direct":
                    mp = MODEL_DIR / 'cnn_denoiser_sigma25.h5'
                else:
                    mp = MODEL_DIR / 'cnn_denoiser_saltpepper.h5'
                st.session_state.keras_cnn_model = load_keras_cnn_model(str(mp))
                st.session_state.cnn_denoised = denoise_image_keras_cnn(
                    st.session_state.noisy_image, model=st.session_state.keras_cnn_model)
            except Exception as e:
                sb.warning(f"CNN: {e}")
        sb.success("Pipeline complete.")
    else:
        sb.warning("Generate a noisy image first.")

sb.markdown("""
<div style="padding:20px;text-align:center;margin-top:8px;">
  <span style="font-size:0.68rem;color:#1e2d40;">Image Enhancement and Noise Reduction</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────

# Hero
st.markdown("""
<div class="pf-hero">
  <div class="pf-hero-left">
    <div class="pf-hero-tag">Research Pipeline</div>
    <h1>Image Enhancement<br>&amp; <em>Noise Reduction</em></h1>
    <p>Classical digital filters, histogram techniques, and deep learning denoising combined in a single evaluation pipeline.</p>
  </div>
  <div class="pf-hero-right">
    <div class="pf-stat"><div class="pf-stat-num">7</div><div class="pf-stat-label">Filters</div></div>
    <div class="pf-stat"><div class="pf-stat-num">5</div><div class="pf-stat-label">Enhance</div></div>
    <div class="pf-stat"><div class="pf-stat-num">4</div><div class="pf-stat-label">CNN Models</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# Section 01 — Source & Noisy
if st.session_state.original_image is not None:
    st.markdown("""
    <div class="pf-section">
      <span class="pf-section-pill">Step 01</span>
      <h2>Original &amp; Noisy Images</h2>
      <div class="pf-section-rule"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="pf-card"><div class="pf-card-header"><div class="pf-card-dot" style="background:#1a56db;"></div><span class="pf-card-title">Original / Clean</span></div>', unsafe_allow_html=True)
        st.image(st.session_state.original_image, use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        if st.session_state.noisy_image is not None:
            st.markdown(f'<div class="pf-card"><div class="pf-card-header"><div class="pf-card-dot" style="background:#e11d48;"></div><span class="pf-card-title">Degraded — {noise_type_label}</span></div>', unsafe_allow_html=True)
            st.image(st.session_state.noisy_image, use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""<div class="pf-alert info"><span style="font-size:1.1rem;flex-shrink:0;">ℹ️</span><div><strong>Next step</strong>Configure noise in the sidebar, then click Generate Noisy Image or upload a noisy image directly.</div></div>""", unsafe_allow_html=True)

    if st.session_state.noisy_image is not None:
        noisy_psnr = compute_psnr(st.session_state.original_image, st.session_state.noisy_image)
        noisy_ssim = compute_ssim(st.session_state.original_image, st.session_state.noisy_image)
        m1, m2 = st.columns(2)
        m1.metric("Noisy Input PSNR", f"{noisy_psnr:.2f} dB")
        m2.metric("Noisy Input SSIM", f"{noisy_ssim:.3f}")
elif st.session_state.noisy_image is not None:
    st.info("True PSNR requires a clean reference image. Upload the matching clean/original image in the sidebar, or switch to Synthetic Noise Pipeline.")

# Section 02 — Filters & Enhancements
if st.session_state.filtered_images:
    st.markdown("""
    <div class="pf-section">
      <span class="pf-section-pill">Step 02</span>
      <h2>Filters &amp; Enhancement Results</h2>
      <div class="pf-section-rule"></div>
    </div>
    """, unsafe_allow_html=True)

    dot_colors = ["#1a56db","#0e9f8a","#7c3aed","#d97706","#e11d48","#0ea5e9","#f59e0b"]
    items = list(st.session_state.filtered_images.items())
    for row_start in range(0, len(items), 3):
        cols = st.columns(3, gap="large")
        for j, (name, img) in enumerate(items[row_start:row_start+3]):
            dc = dot_colors[j % len(dot_colors)]
            with cols[j]:
                st.markdown(f'<div class="pf-card"><div class="pf-card-header"><div class="pf-card-dot" style="background:{dc};"></div><span class="pf-card-title">{name}</span></div>', unsafe_allow_html=True)
                st.image(img, use_container_width=True, clamp=True)
                st.markdown('</div>', unsafe_allow_html=True)

    reference_image = (
        st.session_state.original_image
        if st.session_state.original_image is not None
        else st.session_state.noisy_image
    )
    if reference_image is not None:
        has_clean_reference = st.session_state.original_image is not None
        metric_rows = []
        if has_clean_reference:
            metric_rows.append({
                "Output": "Noisy Input",
                "PSNR (dB)": compute_psnr(reference_image, st.session_state.noisy_image),
                "SSIM": compute_ssim(reference_image, st.session_state.noisy_image),
            })
        metric_rows.extend(
            {"Output": name,
             "PSNR (dB)": compute_psnr(reference_image, img),
             "SSIM": compute_ssim(reference_image, img)}
            for name, img in items
        )
        if st.session_state.cnn_denoised is not None:
            metric_rows.append({
                "Output": "CNN Denoised",
                "PSNR (dB)": compute_psnr(reference_image, st.session_state.cnn_denoised),
                "SSIM": compute_ssim(reference_image, st.session_state.cnn_denoised),
            })
        metric_table = pd.DataFrame(metric_rows).sort_values("PSNR (dB)", ascending=False)
        st.markdown("""
        <div class="pf-section" style="margin-top:32px;">
          <span class="pf-section-pill">PSNR · SSIM</span>
          <h2>Quality Comparison</h2>
          <div class="pf-section-rule"></div>
        </div>
        """, unsafe_allow_html=True)
        if has_clean_reference:
            st.caption("Higher PSNR and SSIM indicate a result closer to the clean reference image.")
        else:
            st.info("No clean reference image was uploaded. These are diagnostic PSNR/SSIM values against the noisy input, not ground-truth quality scores.")
        st.dataframe(
            metric_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "PSNR (dB)": st.column_config.NumberColumn(format="%.2f dB"),
                "SSIM": st.column_config.NumberColumn(format="%.3f"),
            },
        )

# Section 03 — CNN Denoising
if st.session_state.cnn_denoised is not None:
    st.markdown("""
    <div class="pf-section">
      <span class="pf-section-pill">Step 03</span>
      <h2>Deep Learning Denoising</h2>
      <div class="pf-section-rule"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown('<div class="pf-card"><div class="pf-card-header"><div class="pf-card-dot" style="background:#e11d48;"></div><span class="pf-card-title">Noisy Input</span></div>', unsafe_allow_html=True)
        st.image(st.session_state.noisy_image, use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="pf-card"><div class="pf-card-header"><div class="pf-card-dot" style="background:#7c3aed;"></div><span class="pf-card-title">CNN Denoised</span></div>', unsafe_allow_html=True)
        st.image(st.session_state.cnn_denoised, use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        if "Gaussian Blur" in st.session_state.filtered_images:
            st.markdown('<div class="pf-card"><div class="pf-card-header"><div class="pf-card-dot" style="background:#0e9f8a;"></div><span class="pf-card-title">Gaussian Blur (Ref.)</span></div>', unsafe_allow_html=True)
            st.image(st.session_state.filtered_images["Gaussian Blur"], use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pf-alert info"><span style="font-size:1.1rem;flex-shrink:0;">💡</span><div>Add <strong>Gaussian Blur</strong> for side-by-side comparison.</div></div>', unsafe_allow_html=True)

    if st.session_state.original_image is not None:
        psnr_cnn   = compute_psnr(st.session_state.original_image, st.session_state.cnn_denoised)
        psnr_noisy = compute_psnr(st.session_state.original_image, st.session_state.noisy_image)
        delta      = psnr_cnn - psnr_noisy
        sign       = "+" if delta >= 0 else ""
        st.markdown(f"""
        <div class="pf-metrics">
          <div class="pf-metric blue">
            <div class="pf-metric-label">CNN Output PSNR</div>
            <div class="pf-metric-value">{psnr_cnn:.1f}<span class="pf-metric-unit"> dB</span></div>
            <div class="pf-metric-sub">Peak signal-to-noise ratio</div>
          </div>
          <div class="pf-metric teal">
            <div class="pf-metric-label">Noisy Input PSNR</div>
            <div class="pf-metric-value">{psnr_noisy:.1f}<span class="pf-metric-unit"> dB</span></div>
            <div class="pf-metric-sub">Before denoising</div>
          </div>
          <div class="pf-metric violet">
            <div class="pf-metric-label">PSNR Improvement</div>
            <div class="pf-metric-value">{sign}{delta:.1f}<span class="pf-metric-unit"> dB</span></div>
            <div class="pf-metric-sub">Gain from CNN</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        ssim_cnn   = compute_ssim(st.session_state.original_image, st.session_state.cnn_denoised)
        ssim_noisy = compute_ssim(st.session_state.original_image, st.session_state.noisy_image)
        ssim_delta = ssim_cnn - ssim_noisy
        ssim_sign  = "+" if ssim_delta >= 0 else ""
        st.markdown(f"""
        <div class="pf-metrics">
          <div class="pf-metric blue">
            <div class="pf-metric-label">CNN Output SSIM</div>
            <div class="pf-metric-value">{ssim_cnn:.3f}</div>
            <div class="pf-metric-sub">Structural similarity index</div>
          </div>
          <div class="pf-metric teal">
            <div class="pf-metric-label">Noisy Input SSIM</div>
            <div class="pf-metric-value">{ssim_noisy:.3f}</div>
            <div class="pf-metric-sub">Before denoising</div>
          </div>
          <div class="pf-metric violet">
            <div class="pf-metric-label">SSIM Improvement</div>
            <div class="pf-metric-value">{ssim_sign}{ssim_delta:.3f}</div>
            <div class="pf-metric-sub">Gain from CNN</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# Section 04 — Export
st.markdown("""
<div class="pf-section">
  <span class="pf-section-pill">Step 04</span>
  <h2>Export Results</h2>
  <div class="pf-section-rule"></div>
</div>
""", unsafe_allow_html=True)

if st.button("Save All Images to outputs/", key="save_all"):
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    saved = []
    if st.session_state.original_image is not None:
        save_image(st.session_state.original_image, "01_original.png"); saved.append("01_original.png")
    if st.session_state.noisy_image is not None:
        save_image(st.session_state.noisy_image, "02_noisy.png"); saved.append("02_noisy.png")
    for i,(n,img) in enumerate(st.session_state.filtered_images.items()):
        fn = f"03_{i:02d}_{n.lower().replace(' ','_')}.png"
        save_image(img, fn); saved.append(fn)
    if st.session_state.cnn_denoised is not None:
        save_image(st.session_state.cnn_denoised, "04_cnn_denoised.png"); saved.append("04_cnn_denoised.png")

    rows = "".join([f'<div class="pf-save-item"><span class="pf-save-check">✓</span>{f}</div>' for f in saved])
    st.markdown(f"""
    <div class="pf-alert success" style="flex-direction:column;align-items:stretch;gap:0;">
      <div style="display:flex;gap:10px;align-items:center;padding-bottom:12px;margin-bottom:4px;border-bottom:1px solid #a7f3d0;">
        <span style="font-size:1.1rem;">✓</span>
        <strong>{len(saved)} file(s) saved to outputs/</strong>
      </div>
      {rows}
    </div>""", unsafe_allow_html=True)

# About
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="pf-section">
  <span class="pf-section-pill">Project</span>
  <h2>About This Application</h2>
  <div class="pf-section-rule"></div>
</div>

<div class="pf-about">
  <div class="pf-about-card">
    <div class="pf-about-icon blue">🎯</div>
    <h3>Part A — Noise Simulation</h3>
    <p>Upload images and apply synthetic Gaussian or Salt & Pepper noise with fully configurable intensity to replicate real-world degradation for benchmarking.</p>
  </div>
  <div class="pf-about-card">
    <div class="pf-about-icon teal">🎛️</div>
    <h3>Part B — Digital Filters</h3>
    <p>Apply and compare seven classical filters alongside five histogram-based enhancement techniques — histogram equalisation, CLAHE, adaptive histogram equalisation, contrast stretching, and gamma correction.</p>
  </div>
  <div class="pf-about-card">
    <div class="pf-about-icon violet">🤖</div>
    <h3>Part C — CNN Denoising</h3>
    <p>Trained convolutional models automatically selected based on noise type and intensity. PSNR and SSIM metrics quantify improvement over classical methods.</p>
  </div>
</div>

<div style="background:#fff;border:1px solid #e4e9f0;border-radius:16px;padding:28px 32px;box-shadow:0 1px 4px rgba(0,0,0,.04);">
  <p style="font-size:0.65rem;font-weight:800;letter-spacing:0.12em;text-transform:uppercase;color:#94a3b8;margin:0 0 14px;">Technology Stack</p>
  <div class="pf-tags">
    <span class="pf-tag">Python</span><span class="pf-tag">OpenCV</span>
    <span class="pf-tag">NumPy</span><span class="pf-tag">TensorFlow</span>
    <span class="pf-tag">Keras</span>
    <span class="pf-tag">Streamlit</span><span class="pf-tag">Pillow</span>
  </div>
  <p style="font-size:0.75rem;color:#cbd5e1;margin:20px 0 0;text-align:right;">Image Enhancement and Noise Reduction Project</p>
</div>
<br><br>
""", unsafe_allow_html=True)