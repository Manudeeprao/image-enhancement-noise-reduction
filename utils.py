"""
Utility functions for image loading, noise addition, and preprocessing.
"""

import cv2
import numpy as np
from pathlib import Path


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
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
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
