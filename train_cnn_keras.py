"""
CNN-based Denoising Model Training using TensorFlow/Keras
Trains a simple convolutional autoencoder on the dataset for Gaussian noise removal.
This script:
- Loads clean images from ./Dataset/NEWDATASET/ directory (300 images)
- Extracts 40x40 patches
- Adds Gaussian noise (sigma=25)
- Builds a simple CNN autoencoder
- Trains for 8 epochs with batch size 16
- Saves trained model to ./models/cnn_denoiser_sigma25.h5
"""

import numpy as np
import cv2
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

# ============================================================================
# Load and Prepare Dataset
# ============================================================================

def load_dataset_patches(dataset_path, patch_size=40, noise_sigma=25):
    """
    Load dataset and extract patches.
    
    Args:
        dataset_path: Path to dataset directory
        patch_size: Size of patches to extract (default: 40x40)
        noise_sigma: Standard deviation of Gaussian noise (default: 25)
        
    Returns:
        noisy_patches: Array of noisy patches [N, 40, 40, 1]
        clean_patches: Array of clean patches [N, 40, 40, 1]
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Get all PNG files
    image_files = sorted(list(dataset_path.glob('*.png')))
    
    if len(image_files) == 0:
        raise ValueError(f"No PNG files found in {dataset_path}")
    
    print(f"Found {len(image_files)} images in dataset")
    
    clean_patches = []
    noisy_patches = []
    
    print("Extracting patches and adding noise...")
    
    for img_path in tqdm(image_files, desc="Loading images"):
        # Read image in grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Extract patches
        h, w = img.shape
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                clean_patch = img[i:i+patch_size, j:j+patch_size]
                
                # Add Gaussian noise
                noise = np.random.normal(0, noise_sigma / 255.0, clean_patch.shape)
                noisy_patch = np.clip(clean_patch + noise, 0, 1)
                
                # Store patches
                clean_patches.append(clean_patch)
                noisy_patches.append(noisy_patch)
    
    # Convert to numpy arrays and add channel dimension
    clean_patches = np.array(clean_patches, dtype=np.float32)
    noisy_patches = np.array(noisy_patches, dtype=np.float32)
    
    # Add channel dimension if needed
    if len(clean_patches.shape) == 3:
        clean_patches = np.expand_dims(clean_patches, axis=-1)
        noisy_patches = np.expand_dims(noisy_patches, axis=-1)
    
    print(f"Extracted {len(clean_patches)} patches")
    print(f"  Clean patches shape: {clean_patches.shape}")
    print(f"  Noisy patches shape: {noisy_patches.shape}")
    
    return noisy_patches, clean_patches


# ============================================================================
# Build CNN Denoising Model
# ============================================================================

def build_cnn_denoiser(input_shape=(40, 40, 1)):
    """
    Build a simple CNN autoencoder for denoising.
    
    Architecture:
    - Input: 40x40x1
    - Conv2D (32 filters, 3x3) + ReLU
    - Conv2D (32 filters, 3x3) + ReLU
    - Conv2D (32 filters, 3x3) + ReLU
    - Conv2D (1 filter, 3x3) - Output layer
    
    Args:
        input_shape: Input image shape (height, width, channels)
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional layer
        layers.Conv2D(filters=32, kernel_size=3, padding='same', 
                     activation='relu', name='conv1'),
        
        # Second convolutional layer
        layers.Conv2D(filters=32, kernel_size=3, padding='same', 
                     activation='relu', name='conv2'),
        
        # Third convolutional layer
        layers.Conv2D(filters=32, kernel_size=3, padding='same', 
                     activation='relu', name='conv3'),
        
        # Output layer (no activation, regression task)
        layers.Conv2D(filters=1, kernel_size=3, padding='same', name='output'),
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model
# ============================================================================
# Training Function
# ============================================================================

def train_model(model, noisy_patches, clean_patches, num_epochs=8, batch_size=16):
    """
    Train the CNN denoising model.

    Args:
        model: Keras model to train
        noisy_patches: Array of noisy training images
        clean_patches: Array of clean target images
        num_epochs: Number of training epochs (default: 8)
        batch_size: Batch size (default: 16)
        
    Returns:
        Training history
    """
    print("\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Total Samples: {len(noisy_patches)}")
    print(f"  Steps per Epoch: {len(noisy_patches) // batch_size}")
    
    # Train model
    history = model.fit(
        noisy_patches, clean_patches,
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=True,
        validation_split=0.1  # Use 10% for validation
    )
    
    return history


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    """
    Main training script for CNN denoising model.
    """
    print("="*70)
    print("CNN-based Denoising Model Training (TensorFlow/Keras)")
    print("="*70)
    
    # -------- Configuration --------
    dataset_path = './Dataset/NEWDATASET'
    model_save_dir = Path('./models')
    model_save_path = model_save_dir / 'cnn_denoiser_sigma25.h5'
    
    patch_size = 40
    noise_sigma = 25
    batch_size = 16
    num_epochs = 8
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Patch Size: {patch_size}x{patch_size}")
    print(f"  Noise Sigma: {noise_sigma}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Save Path: {model_save_path}")
    
    # -------- Step 1: Load Dataset --------
    print("\n" + "="*70)
    print("Step 1: Load Dataset and Extract Patches")
    print("="*70)
    
    noisy_patches, clean_patches = load_dataset_patches(
        dataset_path, 
        patch_size=patch_size, 
        noise_sigma=noise_sigma
    )
    
    # -------- Step 2: Build Model --------
    print("\n" + "="*70)
    print("Step 2: Build CNN Denoising Model")
    print("="*70)
    
    model = build_cnn_denoiser(input_shape=(patch_size, patch_size, 1))
    
    print("\nModel Summary:")
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
    
    # -------- Step 3: Train Model --------
    print("\n" + "="*70)
    print("Step 3: Train Model")
    print("="*70)
    
    history = train_model(
        model,
        noisy_patches,
        clean_patches,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    # -------- Step 4: Save Model --------
    print("\n" + "="*70)
    print("Step 4: Save Trained Model")
    print("="*70)
    
    model_save_dir.mkdir(exist_ok=True)
    model.save(str(model_save_path))
    
    print(f"✓ Model saved to: {model_save_path}")
    print(f"  File size: {model_save_path.stat().st_size / (1024*1024):.2f} MB")
    
    # -------- Step 5: Training Summary --------
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\nFinal Results:")
    print(f"  Training Loss: {final_loss:.6f}")
    print(f"  Validation Loss: {final_val_loss:.6f}")
    print(f"  Total Epochs: {num_epochs}")
    print(f"  Trained Model: {model_save_path}")
    
    print(f"\nThe trained model is ready for inference in the Streamlit app.")


if __name__ == '__main__':
    main()