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

# Professional Custom CSS
st.markdown("""
    <style>
    /* Main Header */
    .main-header {
        font-size: 2.5em;
        font-weight: 700;
        color: #0052A3;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: 0.5px;
    }
    
    /* Subheader */
    .subheader {
        font-size: 1.3em;
        font-weight: 600;
        color: #0052A3;
        margin-top: 25px;
        margin-bottom: 15px;
        padding-top: 10px;
        border-top: 2px solid #E8E8E8;
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 1.1em;
        color: #666666;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 20px;
        font-weight: 500;
    }
    
    /* Section divider */
    .section-divider {
        margin: 30px 0;
        border-top: 2px solid #E8E8E8;
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

st.sidebar.markdown("### Control Panel")
st.sidebar.markdown("---")

# Image Upload
st.sidebar.markdown("#### Upload Image")
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
st.sidebar.markdown("#### Noise Configuration")
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

st.sidebar.markdown("---")

# Filter Selection
st.sidebar.markdown("#### Digital Filters")

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
st.sidebar.markdown("#### Enhancement Methods")

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
st.sidebar.markdown("#### Deep Learning Denoising")
use_cnn = st.sidebar.checkbox("Enable CNN Denoising", value=True)

st.sidebar.markdown("---")

# Apply Filters Button
if st.sidebar.button("Apply All Processing", key="apply_filters", use_container_width=True):
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

st.markdown('<div class="main-header">Image Enhancement & Noise Reduction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Capstone Project: Image Enhancement and Noise Reduction Using Digital Filters</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Original and Noisy Images
if st.session_state.original_image is not None:
    st.markdown('<div class="subheader">Original & Noisy Images</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(st.session_state.original_image, caption="📸 Original Image", use_container_width=True, clamp=True)
    
    with col2:
        if st.session_state.noisy_image is not None:
            st.image(st.session_state.noisy_image, caption=f"🌪️ Noisy Image ({noise_type})", use_container_width=True, clamp=True)
        else:
            st.info("Generate noisy image from the sidebar")

# Digital Filters and Enhancements
if st.session_state.filtered_images:
    st.markdown('<div class="subheader">Filters & Enhancements</div>', unsafe_allow_html=True)
    
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
                    st.image(filtered_img, caption=f"✨ {filter_name}", use_container_width=True, clamp=True)

# CNN-Based Denoising
if st.session_state.cnn_denoised is not None:
    st.markdown('<div class="subheader">Deep Learning Denoising</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(st.session_state.noisy_image, caption="🌪️ Noisy Input", use_container_width=True, clamp=True)
    
    with col2:
        st.image(st.session_state.cnn_denoised, caption="🤖 DnCNN Denoised", use_container_width=True, clamp=True)
    
    with col3:
        if "Gaussian Blur" in st.session_state.filtered_images:
            st.image(st.session_state.filtered_images["Gaussian Blur"], 
                    caption="🔍 Gaussian Blur (Comparison)", use_container_width=True, clamp=True)
        else:
            st.info("Apply Gaussian Blur for comparison")
    
    # Display CNN Evaluation Metrics
    st.markdown("---")
    if st.session_state.original_image is not None:
        psnr = compute_psnr(st.session_state.original_image, st.session_state.cnn_denoised)
        st.metric(
            label="🎯 CNN PSNR",
            value=f"{psnr:.2f} dB",
            help="Peak Signal-to-Noise Ratio: Higher is better."
        )

# Save Results
st.markdown('<div class="subheader">Save Results</div>', unsafe_allow_html=True)

if st.button("Save All Images", key="save_all", use_container_width=True):
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
    
    st.success("✅ All images saved!")
    st.info("\n".join(saved_files))

st.markdown("---")
st.markdown("""
### 📚 About This Application
- **PART A:** Upload an image and add synthetic noise (Gaussian or Salt & Pepper)
- **PART B:** Apply digital filters (Average, Gaussian, Median, Bilateral, etc.) and enhancement methods (Histogram Equalization, CLAHE)
- **PART C:** Use deep learning (pretrained DnCNN) for advanced denoising
""")
