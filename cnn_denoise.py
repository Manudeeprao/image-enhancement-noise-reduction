"""
CNN-based denoising using the project's trained Keras models.
"""

import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf


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
        denoised = cv2.resize(denoised, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)

    # Denormalize to [0, 255] and clip
    denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)

    return denoised
