"""
Utility functions for image loading, noise addition, and preprocessing.
"""

import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def load_image(image_file):
    """
    Load image from uploaded file and convert to grayscale.
    
    Args:
        image_file: Uploaded file object from Streamlit
        
    Returns:
        grayscale image (numpy array)
    """
    # Read image from uploaded file
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return gray_image


def add_gaussian_noise(image, mean=0, std=25):
    """
    Add Gaussian noise to image.
    
    Args:
        image: Input image (grayscale)
        mean: Mean of Gaussian noise
        std: Standard deviation of Gaussian noise
        
    Returns:
        Noisy image
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image.astype(float) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image


def add_salt_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    """
    Add salt & pepper noise to image.
    
    Args:
        image: Input image (grayscale)
        salt_prob: Probability of salt noise
        pepper_prob: Probability of pepper noise
        
    Returns:
        Noisy image
    """
    noisy_image = image.copy().astype(float)
    
    # Add salt (white)
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy_image[salt_mask] = 255
    
    # Add pepper (black)
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy_image[pepper_mask] = 0
    
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def save_image(image, filename):
    """
    Save image to outputs folder.
    
    Args:
        image: Image to save
        filename: Name of the file
    """
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    filepath = OUTPUT_DIR / filename if not str(filename).startswith(str(BASE_DIR)) else Path(filename)
    cv2.imwrite(str(filepath), image)

    return str(filepath)


def normalize_image(image):
    """
    Normalize image to 0-1 range.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    return image.astype(float) / 255.0


def denormalize_image(image):
    """
    Denormalize image from 0-1 range to 0-255.
    
    Args:
        image: Input image in 0-1 range
        
    Returns:
        Denormalized image in 0-255 range
    """
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def estimate_noise_level(image):
    """
    Estimate the noise type and level of a grayscale image, so the app can
    automatically pick the matching CNN denoising model for uploaded images
    whose noise parameters are unknown.

    - Gaussian noise: Immerkaer's (1996) Laplacian-based estimator,
      sigma ~= sqrt(pi/2) * mean(|Laplacian|) / 6.
    - Salt & pepper noise: counts isolated extreme pixels (exactly 0 or 255
      that disagree strongly with their 3x3 neighbourhood median). This avoids
      mistaking naturally dark/bright regions (e.g. X-ray backgrounds) for noise.

    Args:
        image: Grayscale image (numpy array, uint8)

    Returns:
        (noise_type, param): ("gaussian", sigma) with sigma in [0, 255] units,
        or ("salt_pepper", probability) with probability in [0, 1].
    """
    image = np.asarray(image, dtype=np.uint8)

    # Salt & pepper check first: isolated extreme pixels.
    median3 = cv2.medianBlur(image, 3)
    residual = np.abs(image.astype(np.int16) - median3.astype(np.int16))
    isolated_extreme = ((image == 0) | (image == 255)) & (residual > 128)
    sp_frac = float(np.mean(isolated_extreme))
    if sp_frac > 0.01:
        return "salt_pepper", float(min(sp_frac, 1.0))

    # Gaussian noise level via Laplacian estimator.
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sigma = float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(laplacian)) / 6.0)
    return "gaussian", max(sigma, 0.0)
