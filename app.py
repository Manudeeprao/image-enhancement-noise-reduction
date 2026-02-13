"""
Streamlit application for Image Enhancement and Noise Reduction.
Capstone Project: Image Enhancement and Noise Reduction Using Digital Filters
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import pandas as pd

# Import custom modules
from utils import load_image, add_gaussian_noise, add_salt_pepper_noise, save_image, normalize_image, denormalize_image
from filters import (average_filter, gaussian_blur, median_filter_cv, 
                    bilateral_filter, sharpening_filter, 
                    morphological_opening, morphological_closing)
from enhancement import (histogram_equalization, clahe_enhancement, 
                        adaptive_histogram_equalization, contrast_stretching, gamma_correction)
from cnn_denoise import load_dncnn_model, denoise_image_dncnn, load_keras_cnn_model, denoise_image_keras_cnn


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_psnr(original, denoised):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        original: Original clean image (numpy array, 0-255)
        denoised: Denoised image (numpy array, 0-255)
        
    Returns:
        PSNR value in dB
    """
    import numpy as np
    
    # Ensure images are float for computation
    original = original.astype(np.float32)
    denoised = denoised.astype(np.float32)
    
    # Calculate MSE
    mse = np.mean((original - denoised) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 10 * np.log10(255.0 ** 2 / mse)
    
    return psnr
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Image Enhancement & Noise Reduction",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Professional CSS with Modern Design
st.markdown("""
    <style>
    /* Root Variables */
    :root {
        --primary-color: #0052A3;
        --secondary-color: #FF6B6B;
        --accent-color: #4ECDC4;
        --dark-bg: #1a1a1a;
        --light-bg: #f8f9fa;
        --text-dark: #2c3e50;
        --border-color: #e0e0e0;
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 0 !important;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #0052A3 0%, #004494 100%);
        color: white;
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
        padding: 40px 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 82, 163, 0.3);
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 1.2em;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Subheader Sections */
    .subheader {
        font-size: 1.5em;
        font-weight: 700;
        color: white;
        background: linear-gradient(135deg, #0052A3 0%, #004494 100%);
        padding: 15px 25px;
        margin-top: 40px;
        margin-bottom: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Section divider */
    .section-divider {
        margin: 40px 0;
        border-top: 3px solid #0052A3;
        border-radius: 2px;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin: 15px 0;
        border-left: 5px solid #0052A3;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0052A3 0%, #004494 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 82, 163, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #004494 0%, #003d7a 100%);
        box-shadow: 0 6px 20px rgba(0, 82, 163, 0.4);
        transform: translateY(-2px);
    }
    
    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 153, 204, 0.3);
    }
    
    /* Success message */
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    
    /* Info message */
    .info-box {
        background: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
    }
    
    /* Warning message */
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: white;
    }
    
    /* Image container */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        background: white;
        padding: 10px;
    }
    
    /* Grid layout for images */
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    /* Text styling */
    h1, h2, h3 {
        color: #0052A3;
        font-weight: 700;
    }
    
    /* Links */
    a {
        color: #0052A3;
        text-decoration: none;
        font-weight: 600;
    }
    
    a:hover {
        color: #FF6B6B;
        text-decoration: underline;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #0052A3;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #004494;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'noisy_image' not in st.session_state:
    st.session_state.noisy_image = None
if 'filtered_images' not in st.session_state:
    st.session_state.filtered_images = {}
if 'cnn_denoised' not in st.session_state:
    st.session_state.cnn_denoised = None
if 'dncnn_model' not in st.session_state:
    st.session_state.dncnn_model = None
if 'keras_cnn_model' not in st.session_state:
    st.session_state.keras_cnn_model = None
if 'noise_params' not in st.session_state:
    st.session_state.noise_params = None


# ============================================================================
# SIDEBAR - CONTROLS
# ============================================================================

st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #0052A3 0%, #004494 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;'>
<h2 style='color: white; margin: 0;'>🎮 Control Panel</h2>
</div>
""", unsafe_allow_html=True)

# Image Upload
st.sidebar.markdown("""
<div style='background: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 4px solid #0052A3; margin-bottom: 15px;'>
<h4 style='color: #0052A3; margin-top: 0;'>📸 Upload Image</h4>
</div>
""", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader(
    "Select image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        st.session_state.original_image = load_image(uploaded_file)
        st.sidebar.success("✅ Image loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading image: {e}")

# Noise Parameters
st.sidebar.markdown("""
<div style='background: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 4px solid #FF6B6B; margin-top: 20px; margin-bottom: 15px;'>
<h4 style='color: #0052A3; margin-top: 0;'>🌪️ Noise Configuration</h4>
</div>
""", unsafe_allow_html=True)
noise_type = st.sidebar.selectbox(
    "Noise Type",
    ["Gaussian Noise", "Salt & Pepper Noise"]
)

if noise_type == "Gaussian Noise":
    noise_std = st.sidebar.slider("Gaussian Std Dev", 5, 50, 25, step=1)
    noise_params = {"type": "gaussian", "std": noise_std}
else:
    salt_prob = st.sidebar.slider("Salt Probability", 0.01, 0.1, 0.05, step=0.01)
    pepper_prob = st.sidebar.slider("Pepper Probability", 0.01, 0.1, 0.05, step=0.01)
    noise_params = {"type": "salt_pepper", "salt_prob": salt_prob, "pepper_prob": pepper_prob}

# Generate Noisy Image
if st.sidebar.button("🔧 Generate Noisy Image", key="generate_noise"):
    if st.session_state.original_image is not None:
        st.session_state.noise_params = noise_params
        if noise_params["type"] == "gaussian":
            st.session_state.noisy_image = add_gaussian_noise(
                st.session_state.original_image,
                std=noise_params["std"]
            )
        else:
            st.session_state.noisy_image = add_salt_pepper_noise(
                st.session_state.original_image,
                salt_prob=noise_params["salt_prob"],
                pepper_prob=noise_params["pepper_prob"]
            )
        st.sidebar.success("✅ Noisy image generated!")
    else:
        st.sidebar.warning("⚠️ Please upload an image first!")

st.sidebar.markdown("""
---
<div style='background: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 4px solid #4ECDC4; margin-top: 20px; margin-bottom: 15px;'>
<h4 style='color: #0052A3; margin-top: 0;'>🎛️ Digital Filters</h4>
</div>
""", unsafe_allow_html=True)

filter_options = {
    "Average Filter": "average",
    "Gaussian Blur": "gaussian",
    "Median Filter": "median",
    "Bilateral Filter": "bilateral",
    "Sharpening": "sharpening",
    "Morphological Opening": "morph_open",
    "Morphological Closing": "morph_close"
}

selected_filters = st.sidebar.multiselect(
    "Select Filters",
    list(filter_options.keys()),
    default=["Gaussian Blur", "Median Filter"]
)

# Enhancement Methods
st.sidebar.markdown("""
<div style='background: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 4px solid #FFA502; margin-top: 20px; margin-bottom: 15px;'>
<h4 style='color: #0052A3; margin-top: 0;'>✨ Enhancement Methods</h4>
</div>
""", unsafe_allow_html=True)

enhancement_options = {
    "Histogram Equalization": "hist_eq",
    "CLAHE": "clahe",
    "Contrast Stretching": "contrast",
    "Gamma Correction": "gamma"
}

selected_enhancements = st.sidebar.multiselect(
    "Select Methods",
    list(enhancement_options.keys()),
    default=["CLAHE"]
)

# CNN Denoising
st.sidebar.markdown("""
<div style='background: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 4px solid #9B59B6; margin-top: 20px; margin-bottom: 15px;'>
<h4 style='color: #0052A3; margin-top: 0;'>🤖 Deep Learning Denoising</h4>
</div>
""", unsafe_allow_html=True)
use_cnn = st.sidebar.checkbox("Enable CNN Denoising", value=True)

st.sidebar.markdown("---")

# Apply Filters Button
st.sidebar.markdown("""
<div style='text-align: center; margin: 20px 0;'>
""", unsafe_allow_html=True)

if st.sidebar.button("⚡ APPLY ALL PROCESSING ⚡", key="apply_filters", use_container_width=True):
    if st.session_state.noisy_image is not None:
        st.session_state.filtered_images = {}
        
        # Apply selected filters
        for filter_name, filter_key in filter_options.items():
            if filter_name in selected_filters:
                try:
                    if filter_key == "average":
                        st.session_state.filtered_images[filter_name] = average_filter(st.session_state.noisy_image)
                    elif filter_key == "gaussian":
                        st.session_state.filtered_images[filter_name] = gaussian_blur(st.session_state.noisy_image)
                    elif filter_key == "median":
                        st.session_state.filtered_images[filter_name] = median_filter_cv(st.session_state.noisy_image)
                    elif filter_key == "bilateral":
                        st.session_state.filtered_images[filter_name] = bilateral_filter(st.session_state.noisy_image)
                    elif filter_key == "sharpening":
                        st.session_state.filtered_images[filter_name] = sharpening_filter(st.session_state.noisy_image)
                    elif filter_key == "morph_open":
                        st.session_state.filtered_images[filter_name] = morphological_opening(st.session_state.noisy_image)
                    elif filter_key == "morph_close":
                        st.session_state.filtered_images[filter_name] = morphological_closing(st.session_state.noisy_image)
                except Exception as e:
                    st.warning(f"Error applying {filter_name}: {e}")
        
        # Apply selected enhancements
        base_image = st.session_state.filtered_images.get(selected_filters[0] if selected_filters else "Gaussian Blur", 
                                                          st.session_state.noisy_image)
        
        for enh_name, enh_key in enhancement_options.items():
            if enh_name in selected_enhancements:
                try:
                    if enh_key == "hist_eq":
                        st.session_state.filtered_images[enh_name] = histogram_equalization(base_image)
                    elif enh_key == "clahe":
                        st.session_state.filtered_images[enh_name] = clahe_enhancement(base_image)
                    elif enh_key == "contrast":
                        st.session_state.filtered_images[enh_name] = contrast_stretching(base_image)
                    elif enh_key == "gamma":
                        st.session_state.filtered_images[enh_name] = gamma_correction(base_image, gamma=0.8)
                except Exception as e:
                    st.warning(f"Error applying {enh_name}: {e}")
        
        # Apply CNN denoising if enabled
        if use_cnn:
            try:
                # Automatic CNN model selection based on noise type and intensity
                if st.session_state.noise_params is not None:
                    noise_info = st.session_state.noise_params
                    
                    if noise_info["type"] == "gaussian":
                        # Select model based on Gaussian noise sigma
                        sigma = noise_info["std"]
                        if sigma <= 20:
                            model_path = './models/cnn_denoiser_sigma15.h5'
                        elif sigma <= 30:
                            model_path = './models/cnn_denoiser_sigma25.h5'
                        else:
                            model_path = './models/cnn_denoiser_sigma35.h5'
                    else:
                        # Salt & Pepper noise
                        model_path = './models/cnn_denoiser_saltpepper.h5'
                    
                    # Load selected model
                    st.session_state.keras_cnn_model = load_keras_cnn_model(model_path)
                    
                    # Run CNN denoising
                    st.session_state.cnn_denoised = denoise_image_keras_cnn(
                        st.session_state.noisy_image,
                        model=st.session_state.keras_cnn_model
                    )
            except Exception as e:
                st.warning(f"Error in CNN denoising: {e}")
    else:
        st.warning("⚠️ Please generate a noisy image first!")

st.sidebar.markdown("---")
st.sidebar.markdown("#### About")
st.sidebar.markdown("""
**Capstone Project**
Image Enhancement and Noise Reduction Using Digital Filters

**Technologies**
- Python, OpenCV, NumPy
- TensorFlow/Keras
- Streamlit
""")


# ============================================================================
# MAIN CONTENT - DISPLAY RESULTS
# ============================================================================

# Header
st.markdown("""
<div style='background: linear-gradient(135deg, #0052A3 0%, #004494 100%); padding: 40px 20px; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 8px 32px rgba(0, 82, 163, 0.3);'>
<h1 style='color: white; text-align: center; margin: 0; font-size: 3em;'>🖼️ IMAGE ENHANCEMENT & NOISE REDUCTION</h1>
<p style='color: rgba(255,255,255,0.9); text-align: center; margin: 10px 0 0 0; font-size: 1.1em;'>BTech Capstone Project: Digital Filters & Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# Original and Noisy Images
if st.session_state.original_image is not None:
    st.markdown('<div class="subheader">📸 Original & Noisy Images</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        """, unsafe_allow_html=True)
        st.image(st.session_state.original_image, caption="📸 Original Image", use_container_width=True, clamp=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if st.session_state.noisy_image is not None:
            st.markdown("""
            <div style='background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            """, unsafe_allow_html=True)
            st.image(st.session_state.noisy_image, caption=f"🌪️ Noisy Image ({noise_type})", use_container_width=True, clamp=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("📋 Generate noisy image from the sidebar")

# Digital Filters and Enhancements
if st.session_state.filtered_images:
    st.markdown('<div class="subheader">✨ Filters & Enhancements</div>', unsafe_allow_html=True)
    
    num_images = len(st.session_state.filtered_images)
    cols_per_row = 3
    rows = (num_images + cols_per_row - 1) // cols_per_row
    
    for row in range(rows):
        cols = st.columns(cols_per_row)
        
        for col_idx in range(cols_per_row):
            img_idx = row * cols_per_row + col_idx
            
            if img_idx < num_images:
                filter_name = list(st.session_state.filtered_images.keys())[img_idx]
                filtered_img = st.session_state.filtered_images[filter_name]
                
                with cols[col_idx]:
                    st.markdown("""
                    <div style='background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                    """, unsafe_allow_html=True)
                    st.image(filtered_img, caption=f"✨ {filter_name}", use_container_width=True, clamp=True)
                    st.markdown("</div>", unsafe_allow_html=True)

# CNN-Based Denoising
if st.session_state.cnn_denoised is not None:
    st.markdown('<div class="subheader">🤖 Deep Learning Denoising</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        """, unsafe_allow_html=True)
        st.image(st.session_state.noisy_image, caption="🌪️ Noisy Input", use_container_width=True, clamp=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        """, unsafe_allow_html=True)
        st.image(st.session_state.cnn_denoised, caption="🤖 CNN Denoised", use_container_width=True, clamp=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        if "Gaussian Blur" in st.session_state.filtered_images:
            st.markdown("""
            <div style='background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            """, unsafe_allow_html=True)
            st.image(st.session_state.filtered_images["Gaussian Blur"], 
                    caption="🔍 Gaussian Blur", use_container_width=True, clamp=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: #d1ecf1; padding: 20px; border-radius: 10px; border-left: 4px solid #17a2b8; text-align: center;'>
            <p style='color: #0c5460; margin: 0;'>📋 Apply Gaussian Blur for comparison</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display CNN Evaluation Metrics
    st.markdown("---")
    if st.session_state.original_image is not None:
        psnr = compute_psnr(st.session_state.original_image, st.session_state.cnn_denoised)
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 15px rgba(0,153,204,0.3);'>
            <h4 style='margin: 0; color: white;'>🎯 PSNR</h4>
            <p style='font-size: 2em; margin: 10px 0 0 0; font-weight: bold;'>{psnr:.2f} dB</p>
            <small style='color: rgba(255,255,255,0.8);'>Peak Signal-to-Noise Ratio</small>
            </div>
            """, unsafe_allow_html=True)

# Save Results
st.markdown('<div class="subheader">💾 Save Results</div>', unsafe_allow_html=True)

if st.button("💾 SAVE ALL IMAGES", key="save_all", use_container_width=True):
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    saved_files = []
    
    # Save original and noisy
    if st.session_state.original_image is not None:
        save_image(st.session_state.original_image, "01_original.png")
        saved_files.append("✅ original.png")
    
    if st.session_state.noisy_image is not None:
        save_image(st.session_state.noisy_image, "02_noisy.png")
        saved_files.append("✅ noisy.png")
    
    # Save filtered images
    for idx, (name, img) in enumerate(st.session_state.filtered_images.items()):
        filename = f"03_filtered_{idx:02d}_{name.lower().replace(' ', '_')}.png"
        save_image(img, filename)
        saved_files.append(f"✅ {filename}")
    
    # Save CNN denoised
    if st.session_state.cnn_denoised is not None:
        save_image(st.session_state.cnn_denoised, "04_dncnn_denoised.png")
        saved_files.append("✅ dncnn_denoised.png")
    
    st.markdown("""
    <div style='background: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; margin: 20px 0;'>
    <h4 style='margin-top: 0; color: #155724;'>✅ All images saved successfully!</h4>
    <p style='margin: 0;'><strong>Saved files:</strong></p>
    </div>
    """, unsafe_allow_html=True)
    for file in saved_files:
        st.markdown(f"- {file}")

st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 30px; border-radius: 12px; border-left: 5px solid #0052A3;'>
<h3 style='color: #0052A3; margin-top: 0;'>📚 About This Application</h3>

<div style='background: white; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #4ECDC4;'>
<h4 style='color: #0052A3; margin-top: 0;'>🎯 PART A: Image Upload & Noise Generation</h4>
<p style='margin: 0;'>Upload an image and add synthetic noise (Gaussian or Salt & Pepper) with configurable parameters</p>
</div>

<div style='background: white; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #FF6B6B;'>
<h4 style='color: #0052A3; margin-top: 0;'>🎛️ PART B: Digital Filters & Enhancements</h4>
<p style='margin: 0;'>Apply 7+ digital filters and 5+ enhancement techniques (Histogram Equalization, CLAHE, Contrast Stretching, Gamma Correction)</p>
</div>

<div style='background: white; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #9B59B6;'>
<h4 style='color: #0052A3; margin-top: 0;'>🤖 PART C: Deep Learning Denoising</h4>
<p style='margin: 0;'>Use pretrained DnCNN models for advanced AI-powered denoising with automatic model selection</p>
</div>

<div style='margin-top: 20px; padding: 15px; background: #f0f2f6; border-radius: 8px;'>
<p style='margin: 0; color: #555;'><strong>🛠️ Technologies:</strong> Python • OpenCV • TensorFlow/Keras • Streamlit • Deep Learning</p>
<p style='margin: 10px 0 0 0; color: #555;'><strong>📍 Dataset:</strong> Custom Dataset (300 medical images with data augmentation)</p>
</div>
</div>
""", unsafe_allow_html=True)
