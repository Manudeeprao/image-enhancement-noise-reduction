"""
CNN-based Denoising Model Training using TensorFlow/Keras.

Single parameterized training script. Replaces the four legacy scripts
(train_cnn_keras.py, train_cnn_sigma15.py, train_cnn_sigma35.py,
train_cnn_salt_pepper.py), which duplicated this exact pipeline.

This script:
- Loads clean PNG images from Dataset/NEWDATASET/ (300 images)
- Extracts non-overlapping 40x40 patches
- Injects the requested synthetic noise (Gaussian or Salt & Pepper)
- Builds and trains a small 4-layer CNN denoiser
  (3x Conv2D-32 ReLU + Conv2D-1 output, Adam lr 0.001, MSE loss)
- Saves the trained model as .h5 into ./models/

Usage:
    # Gaussian noise, sigma 25
    python train_cnn.py --noise-type gaussian --noise-param 25

    # Gaussian noise, sigma 15, custom epochs
    python train_cnn.py --noise-type gaussian --noise-param 15 --epochs 12

    # Salt & Pepper noise, 10% corruption probability
    python train_cnn.py --noise-type salt_pepper --noise-param 0.1
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "Dataset" / "NEWDATASET"
DEFAULT_MODELS = BASE_DIR / "models"


# ============================================================================
# Load and Prepare Dataset
# ============================================================================

def add_gaussian_noise_patch(patch, noise_sigma):
    """Inject Gaussian noise (sigma in [0,255] units) into a [0,1] patch."""
    noise = np.random.normal(0, noise_sigma / 255.0, patch.shape)
    return np.clip(patch + noise, 0, 1)


def add_salt_pepper_noise_patch(patch, noise_prob):
    """Inject Salt & Pepper noise (probability per pixel) into a [0,1] patch."""
    noisy_patch = patch.copy()
    mask = np.random.random(patch.shape)
    noisy_patch[mask < (noise_prob / 2)] = 1.0   # salt
    pepper_mask = (mask >= (noise_prob / 2)) & (mask < noise_prob)
    noisy_patch[pepper_mask] = 0.0               # pepper
    return noisy_patch


def load_dataset_patches(dataset_path, patch_size=40, noise_type="gaussian", noise_param=25.0):
    """
    Load dataset and extract (noisy, clean) patch pairs.

    Args:
        dataset_path: Path to directory of clean PNG images
        patch_size: Patch edge length (default 40)
        noise_type: 'gaussian' or 'salt_pepper'
        noise_param: Gaussian sigma in [0,255] units, or S&P probability

    Returns:
        noisy_patches: [N, patch_size, patch_size, 1] float32 in [0,1]
        clean_patches: [N, patch_size, patch_size, 1] float32 in [0,1]
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    image_files = sorted(dataset_path.glob("*.png"))
    if not image_files:
        raise ValueError(f"No PNG files found in {dataset_path}")

    print(f"Found {len(image_files)} images in dataset")

    if noise_type == "gaussian":
        noise_fn = lambda p: add_gaussian_noise_patch(p, noise_param)
        noise_desc = f"Gaussian noise (sigma={noise_param})"
    elif noise_type == "salt_pepper":
        noise_fn = lambda p: add_salt_pepper_noise_patch(p, noise_param)
        noise_desc = f"Salt & Pepper noise (p={noise_param})"
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    clean_patches, noisy_patches = [], []

    print(f"Extracting patches and adding {noise_desc}...")
    for img_path in tqdm(image_files, desc="Loading images"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        img = img.astype(np.float32) / 255.0
        h, w = img.shape
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                clean_patch = img[i:i + patch_size, j:j + patch_size]
                clean_patches.append(clean_patch)
                noisy_patches.append(noise_fn(clean_patch))

    clean_patches = np.array(clean_patches, dtype=np.float32)
    noisy_patches = np.array(noisy_patches, dtype=np.float32)

    if clean_patches.ndim == 3:
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
    Build a small CNN denoiser.

    Architecture:
    - Conv2D (32 filters, 3x3) + ReLU
    - Conv2D (32 filters, 3x3) + ReLU
    - Conv2D (32 filters, 3x3) + ReLU
    - Conv2D (1 filter, 3x3) - Output layer (regression, no activation)

    Args:
        input_shape: Input image shape (height, width, channels)

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", name="conv1"),
        layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", name="conv2"),
        layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", name="conv3"),
        layers.Conv2D(filters=1, kernel_size=3, padding="same", name="output"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
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
        num_epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        Training history
    """
    print("\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Total Samples: {len(noisy_patches)}")
    print(f"  Steps per Epoch: {len(noisy_patches) // batch_size}")

    history = model.fit(
        noisy_patches, clean_patches,
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=True,
        validation_split=0.1,  # 10% for validation
    )
    return history


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the CNN denoising model on noisy 40x40 patches.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Directory of clean PNG training images")
    parser.add_argument("--noise-type", choices=["gaussian", "salt_pepper"],
                        default="gaussian", help="Noise type to inject")
    parser.add_argument("--noise-param", type=float, default=25.0,
                        help="Gaussian sigma (0-255 units) or S&P corruption probability")
    parser.add_argument("--patch-size", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None,
                        help="Model save path (default: models/cnn_denoiser_<noise>.h5)")
    return parser.parse_args()


def default_output_path(noise_type, noise_param):
    if noise_type == "gaussian":
        name = f"cnn_denoiser_sigma{int(noise_param)}.h5"
    else:
        name = "cnn_denoiser_saltpepper.h5"
    return DEFAULT_MODELS / name


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    args = parse_args()
    output_path = args.output or default_output_path(args.noise_type, args.noise_param)

    print("=" * 70)
    print("CNN-based Denoising Model Training (TensorFlow/Keras)")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Patch Size: {args.patch_size}x{args.patch_size}")
    print(f"  Noise Type: {args.noise_type}")
    print(f"  Noise Param: {args.noise_param}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Save Path: {output_path}")

    print("\n" + "=" * 70)
    print("Step 1: Load Dataset and Extract Patches")
    print("=" * 70)
    noisy_patches, clean_patches = load_dataset_patches(
        args.dataset,
        patch_size=args.patch_size,
        noise_type=args.noise_type,
        noise_param=args.noise_param,
    )

    print("\n" + "=" * 70)
    print("Step 2: Build CNN Denoising Model")
    print("=" * 70)
    model = build_cnn_denoiser(input_shape=(args.patch_size, args.patch_size, 1))

    print("\nModel Summary:")
    model.summary()
    print(f"\nTotal Parameters: {model.count_params():,}")

    print("\n" + "=" * 70)
    print("Step 3: Train Model")
    print("=" * 70)
    history = train_model(
        model, noisy_patches, clean_patches,
        num_epochs=args.epochs, batch_size=args.batch_size,
    )

    print("\n" + "=" * 70)
    print("Step 4: Save Trained Model")
    print("=" * 70)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    model.save(str(output_path))

    print(f"Model saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nFinal Results:")
    print(f"  Training Loss: {history.history['loss'][-1]:.6f}")
    print(f"  Validation Loss: {history.history['val_loss'][-1]:.6f}")
    print(f"  Total Epochs: {args.epochs}")
    print(f"  Trained Model: {output_path}")
    print("\nThe trained model is ready for inference in the Streamlit app.")


if __name__ == "__main__":
    main()
