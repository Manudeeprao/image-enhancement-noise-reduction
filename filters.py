"""
Digital filters implementation for image enhancement.
"""

import cv2
import numpy as np
from scipy.ndimage import median_filter, binary_dilation, binary_erosion


def average_filter(image, kernel_size=5):
    """
    Apply average (box) filter to image.
    
    Args:
        image: Input image (grayscale)
        kernel_size: Size of the filter kernel (must be odd)
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    filtered = cv2.blur(image, (kernel_size, kernel_size))
    return filtered


def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur filter to image.
    
    Args:
        image: Input image (grayscale)
        kernel_size: Size of the filter kernel (must be odd)
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return filtered


def median_filter_cv(image, kernel_size=5):
    """
    Apply median filter to image.
    
    Args:
        image: Input image (grayscale)
        kernel_size: Size of the filter kernel (must be odd)
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    filtered = cv2.medianBlur(image, kernel_size)
    return filtered


def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter to image (edge-preserving smoothing).
    
    Args:
        image: Input image (grayscale)
        diameter: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Filtered image
    """
    # Bilateral filter works best on 3-channel images
    # Convert grayscale to 3-channel, apply filter, convert back
    bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    filtered_bgr = cv2.bilateralFilter(bgr_image, diameter, sigma_color, sigma_space)
    filtered = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2GRAY)
    
    return filtered


def sharpening_filter(image, strength=1.5):
    """
    Apply sharpening filter to image.
    
    Args:
        image: Input image (grayscale)
        strength: Strength of sharpening (higher = more sharp)
        
    Returns:
        Sharpened image
    """
    # Create Laplacian kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    # Apply filter
    sharpened = cv2.filter2D(image.astype(np.float32), -1, kernel * strength)
    
    # Normalize and clip
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


def morphological_opening(image, kernel_size=5):
    """
    Apply morphological opening (erosion followed by dilation).
    
    Args:
        image: Input image (grayscale)
        kernel_size: Size of the morphological kernel
        
    Returns:
        Filtered image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    filtered = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    return filtered


def morphological_closing(image, kernel_size=5):
    """
    Apply morphological closing (dilation followed by erosion).
    
    Args:
        image: Input image (grayscale)
        kernel_size: Size of the morphological kernel
        
    Returns:
        Filtered image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    filtered = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    return filtered
