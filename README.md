# Image Enhancement and Noise Reduction Using Digital Filters

## 📋 Project Overview

A comprehensive Streamlit application for image enhancement and noise reduction using digital filters, enhancement techniques, and deep learning (DnCNN). This capstone project demonstrates various signal processing and machine learning techniques for image quality improvement.

## 🎯 Features

### PART A - Image Upload & Noise Addition
- Upload JPG/PNG images
- Automatic grayscale conversion
- Synthetic noise generation:
  - **Gaussian Noise**: Configurable standard deviation
  - **Salt & Pepper Noise**: Adjustable salt and pepper probabilities

### PART B - Digital Filters & Enhancements
#### Digital Filters:
- Average Filter (Box Filter)
- Gaussian Blur
- Median Filter
- Bilateral Filter (edge-preserving)
- Sharpening Filter
- Morphological Opening
- Morphological Closing

#### Enhancement Techniques:
- Histogram Equalization
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Adaptive Histogram Equalization (AHE)
- Contrast Stretching
- Gamma Correction

### PART C - Deep Learning Denoising
- **DnCNN (Denoising Convolutional Neural Network)**
- Pretrained model structure
- PyTorch inference
- GPU acceleration support

## 📁 Project Structure

```
image-enhancement-project/
├── app.py                      # Main Streamlit application
├── filters.py                  # Digital filter implementations
├── enhancement.py              # Enhancement techniques
├── cnn_denoise.py              # DnCNN model and inference
├── utils.py                    # Utility functions
├── train_cnn_keras.py          # Training script (Gaussian σ=25)
├── train_cnn_sigma15.py        # Training script (Gaussian σ=15)
├── train_cnn_sigma35.py        # Training script (Gaussian σ=35)
├── train_cnn_salt_pepper.py    # Training script (Salt & Pepper)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── Dataset/
│   └── NEWDATASET/             # Training dataset (300 images)
├── models/                     # Trained CNN models (.h5 files)
└── outputs/                    # Directory for saving results
```

## 🚀 Installation & Usage

### 1. Clone/Download the Project
```bash
cd image-enhancement-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Train CNN Models
The project includes pre-trained models, but you can retrain using your dataset:

```bash
# Train for Gaussian noise (sigma=15)
python train_cnn_sigma15.py

# Train for Gaussian noise (sigma=25)
python train_cnn_keras.py

# Train for Gaussian noise (sigma=35)
python train_cnn_sigma35.py

# Train for Salt & Pepper noise
python train_cnn_salt_pepper.py
```

**Dataset**: Place your training images (PNG format) in `Dataset/NEWDATASET/`. The current dataset contains 300 medical images.

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 How to Use the Application

### Step 1: Upload an Image
- Use the sidebar to upload a JPG or PNG image
- The image will be automatically converted to grayscale

### Step 2: Configure Noise
- Select noise type: Gaussian or Salt & Pepper
- Adjust noise parameters using sliders
- Click "Generate Noisy Image" to add synthetic noise

### Step 3: Apply Filters
- Select desired filters from the checklist
- Select enhancement methods
- Optionally enable DnCNN denoising and load the model
- Click "APPLY ALL FILTERS" to process

### Step 4: View Results
- Results are displayed in a grid layout
- Compare original, noisy, filtered, and denoised images

### Step 5: Save Results
- Click "Save All Images to outputs/" to save all processed images
- Images are saved with descriptive names in the `outputs/` folder

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web UI framework
- **OpenCV (cv2)**: Image processing
- **NumPy**: Numerical computations
- **PyTorch**: Deep learning framework
- **Pillow**: Image manipulation
- **SciPy**: Scientific computing

### Device Support
- **CPU**: Fully supported (default)
- **GPU (CUDA)**: Automatically detected and used for DnCNN inference

### Image Processing Pipeline
1. **Input**: Grayscale image (uint8)
2. **Noise Addition**: Synthetic noise injection
3. **Filtering**: Multiple filter passes for noise reduction
4. **Enhancement**: Contrast and brightness adjustments
5. **Deep Learning**: DnCNN for advanced denoising
6. **Output**: Enhanced image with preserved details

## 📊 Filter Characteristics

| Filter | Best For | Speed | Edge Preservation |
|--------|----------|-------|-------------------|
| Average | Quick smoothing | Very Fast | Poor |
| Gaussian | Smooth blur | Fast | Poor |
| Median | Salt & Pepper noise | Moderate | Good |
| Bilateral | Preserving edges | Slow | Excellent |
| Morphological | Structure analysis | Fast | Good |
| Sharpening | Detail enhancement | Fast | Good |

## 🤖 DnCNN Model Information

- **Architecture**: 17-layer CNN with residual learning
- **Training**: Trained on diverse noise types
- **Input**: Grayscale images (1 channel)
- **Output**: Denoised image (same size as input)
- **Speed**: GPU ~50ms per image, CPU ~500ms

## 📝 Code Examples

### Using Filters Directly
```python
from filters import gaussian_blur, median_filter_cv
import cv2

# Load image
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Apply filters
blurred = gaussian_blur(img, kernel_size=5)
denoised = median_filter_cv(img, kernel_size=5)
```

### Adding Noise
```python
from utils import add_gaussian_noise, add_salt_pepper_noise

# Gaussian noise
noisy1 = add_gaussian_noise(img, mean=0, std=25)

# Salt & pepper noise
noisy2 = add_salt_pepper_noise(img, salt_prob=0.05, pepper_prob=0.05)
```

### CNN Denoising
```python
from cnn_denoise import load_dncnn_model, denoise_image_dncnn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_dncnn_model(device=device)
denoised = denoise_image_dncnn(noisy_img, model=model, device=device)
```

## 📈 Performance Metrics

The application can be evaluated using:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **MAE** (Mean Absolute Error)

To compute metrics, install:
```bash
pip install scikit-image
```

## 🐛 Troubleshooting

### Issue: "No module named 'streamlit'"
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Issue: "CUDA not available for DnCNN"
**Solution**: The app will automatically fallback to CPU. No action needed.

### Issue: "Image upload not working"
**Solution**: Ensure file is JPG/PNG format and less than 200MB

### Issue: Filters are too slow
**Solution**: Use smaller images or smaller kernel sizes

## 📚 References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [DnCNN Paper](https://arxiv.org/abs/1608.03981)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## ✅ Capstone Project Checklist

- [x] Image upload and grayscale conversion
- [x] Noise addition (Gaussian and Salt & Pepper)
- [x] Digital filter implementations (7 filters)
- [x] Enhancement techniques (4 methods)
- [x] DnCNN model integration
- [x] Streamlit UI with sidebar controls
- [x] Image saving to outputs folder
- [x] Modular code structure
- [x] Complete documentation

## 📄 License

This project is provided as-is for educational purposes.

## 👨‍💻 Author

Capstone Project - Image Enhancement and Noise Reduction Using Digital Filters

---

**Last Updated**: December 2025
