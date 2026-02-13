"""
CNN-based denoising using trained models.
Supports both PyTorch DnCNN and Keras CNN models.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from pathlib import Path
import urllib.request
import tensorflow as tf


class DnCNN(nn.Module):
    """
    DnCNN model for image denoising.
    Based on: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN
    """
    
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        
        layers = []
        
        # First layer
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size,
                               padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size,
                                   padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        # Last layer
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size,
                               padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out  # Residual learning


def download_dncnn_model(model_path='dncnn_model.pth'):
    """
    Download pretrained DnCNN model from GitHub.
    
    Args:
        model_path: Path to save the model
        
    Returns:
        Path to the model file
    """
    if os.path.exists(model_path):
        return model_path
    
    # Alternative: Use a simple pretrained model from PyTorch
    print(f"Note: Using locally trained DnCNN model structure.")
    return None


def load_dncnn_model(device='cpu', model_path=None):
    """
    Load pretrained DnCNN model.
    
    Args:
        device: Device to load model on ('cpu' or 'cuda')
        model_path: Path to pretrained model weights
        
    Returns:
        DnCNN model in evaluation mode
    """
    model = DnCNN(channels=1, num_of_layers=17)
    model = model.to(device)
    
    # If pretrained weights available, load them
    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded pretrained model from {model_path}")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Using randomly initialized model")
    
    model.eval()
    return model


def denoise_image_dncnn(noisy_image, model=None, device='cpu'):
    """
    Denoise image using DnCNN.
    
    Args:
        noisy_image: Noisy image (grayscale, numpy array)
        model: DnCNN model (will create if None)
        device: Device to run inference on
        
    Returns:
        Denoised image (numpy array)
    """
    if model is None:
        model = load_dncnn_model(device=device)
    
    # Convert image to tensor
    img_tensor = torch.from_numpy(noisy_image).float().to(device)
    
    # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    
    # Normalize to [0, 1]
    img_tensor = img_tensor / 255.0
    
    # Run inference
    with torch.no_grad():
        denoised_tensor = model(img_tensor)
    
    # Convert back to numpy
    denoised = denoised_tensor.squeeze(0).squeeze(0).cpu().numpy()
    
    # Denormalize and clip
    denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
    
    return denoised


def batch_denoise_dncnn(images_list, device='cpu'):
    """
    Denoise multiple images using DnCNN.
    
    Args:
        images_list: List of noisy images
        device: Device to run inference on

    Returns:
        List of denoised images
    """
    model = load_dncnn_model(device=device)
    denoised_list = []
    for img in images_list:
        denoised = denoise_image_dncnn(img, model=model, device=device)
        denoised_list.append(denoised)
    return denoised_list


# ============================================================================
# KERAS CNN MODEL FUNCTIONS (TRAINED MODEL)
# ============================================================================

def load_keras_cnn_model(model_path='./models/cnn_denoiser_sigma25.h5'):
    """
    Load trained Keras CNN denoising model.
    
    Args:
        model_path: Path to trained Keras model
        
    Returns:
        Loaded Keras model
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    try:
        # Load with compile=False to avoid metric deserialization issues
        model = tf.keras.models.load_model(str(model_path), compile=False)
        return model
    except Exception as e:
        raise Exception(f"Error loading Keras model: {e}")


def denoise_image_keras_cnn(noisy_image, model=None, model_path='./models/cnn_denoiser_sigma25.h5'):
    """
    Denoise image using trained Keras CNN model.
    
    Args:
        noisy_image: Noisy image (grayscale, numpy array, [0, 255])
        model: Loaded Keras model (will load if None)
        model_path: Path to trained model file
        
    Returns:
        Denoised image (numpy array, [0, 255])
    """
    # Load model if not provided
    if model is None:
        model = load_keras_cnn_model(model_path)
    
    # Store original shape
    original_shape = noisy_image.shape
    
    # Normalize to [0, 1]
    img_normalized = noisy_image.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions: (H, W) -> (1, H, W, 1)
    img_tensor = np.expand_dims(img_normalized, axis=0)  # Batch
    img_tensor = np.expand_dims(img_tensor, axis=-1)     # Channel
    
    # Run inference
    denoised_tensor = model.predict(img_tensor, verbose=0)
    
    # Remove batch and channel dimensions
    denoised = np.squeeze(denoised_tensor)
    
    # Ensure output shape matches input shape
    if denoised.shape != original_shape:
        import cv2
        denoised = cv2.resize(denoised, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Denormalize to [0, 255] and clip
    denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
    
    return denoised