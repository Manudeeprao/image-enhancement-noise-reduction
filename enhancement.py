"""
Image enhancement techniques: Histogram Equalization and CLAHE.
"""

import cv2
import numpy as np


def histogram_equalization(image):
    """
    Apply histogram equalization to enhance image contrast.
    
    Args:
        image: Input image (grayscale)
        
    Returns:
        Enhanced image
    """
    enhanced = cv2.equalizeHist(image)
    return enhanced
def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image: Input image (grayscale)
        clip_limit: Threshold for contrast limiting (1.0-4.0)
        tile_grid_size: Size of grid for adaptive histogram equalization
        
    Returns:
        Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    
    return enhanced


def adaptive_histogram_equalization(image, window_size=25):
    """
    Apply Adaptive Histogram Equalization (AHE).
    Similar to CLAHE but without contrast limiting.
    
    Args:
        image: Input image (grayscale)
        window_size: Size of the local window for computation
        
    Returns:
        Enhanced image
    """
    # Use CLAHE with high clip limit for AHE-like behavior
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(window_size, window_size))
    enhanced = clahe.apply(image)
    
    return enhanced


def contrast_stretching(image):
    """
    Apply contrast stretching to expand dynamic range.
    
    Args:
        image: Input image (grayscale)
        
    Returns:
        Enhanced image
    """
    # Get min and max pixel values
    p_min = np.min(image)
    p_max = np.max(image)
    
    # Stretch contrast
    if p_max > p_min:
        enhanced = ((image - p_min) / (p_max - p_min) * 255).astype(np.uint8)
    else:
        enhanced = image.copy()
    
    return enhanced


def gamma_correction(image, gamma=1.0):
    """
    Apply gamma correction for brightness adjustment.
    
    Args:
        image: Input image (grayscale)
        gamma: Gamma value (< 1.0 brightens, > 1.0 darkens)
        
    Returns:
        Gamma-corrected image
    """
    # Build a lookup table mapping pixel values [0, 255] to adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    
    # Apply gamma correction using the lookup table
    corrected = cv2.LUT(image, table)
    
    return corrected
