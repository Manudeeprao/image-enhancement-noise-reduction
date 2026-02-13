import numpy as np
import cv2
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

def load_dataset_patches(dataset_path, patch_size=40, noise_sigma=35):
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    image_files = sorted(list(dataset_path.glob('*.png')))
    
    if len(image_files) == 0:
        raise ValueError(f"No PNG files found in {dataset_path}")
    
    print(f"Found {len(image_files)} images in dataset")
    
    clean_patches = []
    noisy_patches = []
    
    print("Extracting patches and adding noise...")
    
    for img_path in tqdm(image_files, desc="Loading images"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        img = img.astype(np.float32) / 255.0
        
        h, w = img.shape
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                clean_patch = img[i:i+patch_size, j:j+patch_size]
                
                noise = np.random.normal(0, noise_sigma / 255.0, clean_patch.shape)
                noisy_patch = np.clip(clean_patch + noise, 0, 1)
                
                clean_patches.append(clean_patch)
                noisy_patches.append(noisy_patch)
    
    clean_patches = np.array(clean_patches, dtype=np.float32)
    noisy_patches = np.array(noisy_patches, dtype=np.float32)
    
    if len(clean_patches.shape) == 3:
        clean_patches = np.expand_dims(clean_patches, axis=-1)
        noisy_patches = np.expand_dims(noisy_patches, axis=-1)
    
    print(f"Extracted {len(clean_patches)} patches")
    print(f"  Clean patches shape: {clean_patches.shape}")
    print(f"  Noisy patches shape: {noisy_patches.shape}")
    
    return noisy_patches, clean_patches


def build_cnn_denoiser(input_shape=(40, 40, 1)):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(filters=32, kernel_size=3, padding='same', 
                     activation='relu', name='conv1'),
        
        layers.Conv2D(filters=32, kernel_size=3, padding='same', 
                     activation='relu', name='conv2'),
        
        layers.Conv2D(filters=32, kernel_size=3, padding='same', 
                     activation='relu', name='conv3'),
        
        layers.Conv2D(filters=1, kernel_size=3, padding='same', name='output'),
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_model(model, noisy_patches, clean_patches, num_epochs=8, batch_size=16):
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
        validation_split=0.1
    )
    
    return history


def main():
    print("="*70)
    print("CNN-based Denoising Model Training (TensorFlow/Keras)")
    print("="*70)
    
    dataset_path = './Dataset/NEWDATASET'
    model_save_dir = Path('./models')
    model_save_path = model_save_dir / 'cnn_denoiser_sigma35.h5'
    
    patch_size = 40
    noise_sigma = 35
    batch_size = 16
    num_epochs = 8
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Patch Size: {patch_size}x{patch_size}")
    print(f"  Noise Sigma: {noise_sigma}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Save Path: {model_save_path}")
    
    print("\n" + "="*70)
    print("Step 1: Load Dataset and Extract Patches")
    print("="*70)
    
    noisy_patches, clean_patches = load_dataset_patches(
        dataset_path, 
        patch_size=patch_size, 
        noise_sigma=noise_sigma
    )
    
    print("\n" + "="*70)
    print("Step 2: Build CNN Denoising Model")
    print("="*70)
    
    model = build_cnn_denoiser(input_shape=(patch_size, patch_size, 1))
    
    print("\nModel Summary:")
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
    
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
    
    print("\n" + "="*70)
    print("Step 4: Save Trained Model")
    print("="*70)
    
    model_save_dir.mkdir(exist_ok=True)
    model.save(str(model_save_path))
    
    print(f"✓ Model saved to: {model_save_path}")
    print(f"  File size: {model_save_path.stat().st_size / (1024*1024):.2f} MB")
    
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


if __name__ == '__main__':
    main()
