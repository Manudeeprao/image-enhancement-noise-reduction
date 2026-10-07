# Image Enhancement and Noise Reduction Using Digital Filters

## 📋 Project Overview

A comprehensive Streamlit application for image enhancement and noise reduction using digital filters, enhancement techniques, and a trained Keras CNN denoiser. This capstone project demonstrates various signal processing and machine learning techniques for image quality improvement.

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
- **Keras CNN denoiser** (small 4-layer convolutional network)
- Four pretrained models (`.h5`), auto-selected by noise type and intensity:
  Gaussian σ15 / σ25 / σ35 and Salt & Pepper
- Direct uploads: noise type/level is auto-detected from the image, so the user
  never has to specify it
- TensorFlow/Keras inference

## 📁 Project Structure

```
image-enhancement-project/
├── app.py                      # Main Streamlit application
├── filters.py                  # Digital filter implementations
├── enhancement.py              # Enhancement techniques
├── cnn_denoise.py              # Keras CNN model loading and inference
├── utils.py                    # Utility functions
├── train_cnn.py                # Training script (parameterized: Gaussian σ15/25/35, Salt & Pepper)
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
python train_cnn.py --noise-type gaussian --noise-param 15

# Train for Gaussian noise (sigma=25)
python train_cnn.py --noise-type gaussian --noise-param 25

# Train for Gaussian noise (sigma=35)
python train_cnn.py --noise-type gaussian --noise-param 35

# Train for Salt & Pepper noise (10% corruption probability)
python train_cnn.py --noise-type salt_pepper --noise-param 0.1
```

Options: `--patch-size`, `--epochs`, `--batch-size`, `--dataset`, `--output`
(see `python train_cnn.py --help`).

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
- Optionally enable CNN denoising and click "Run Full Pipeline" to process
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
- **TensorFlow/Keras**: Deep learning framework for the CNN denoiser
- **scikit-image**: SSIM metric computation
- **Pillow**: Image manipulation
- **SciPy**: Scientific computing

### Device Support
- **CPU**: Fully supported (default)
- **GPU**: Used automatically by TensorFlow when available

### Image Processing Pipeline
1. **Input**: Grayscale image (uint8)
2. **Noise Addition**: Synthetic noise injection
3. **Filtering**: Multiple filter passes for noise reduction
4. **Enhancement**: Contrast and brightness adjustments
5. **Deep Learning**: Trained Keras CNN denoiser, model selected by noise type/intensity
6. **Output**: Enhanced image with preserved details, compared via PSNR and SSIM

## 📊 Filter Characteristics

| Filter | Best For | Speed | Edge Preservation |
|--------|----------|-------|-------------------|
| Average | Quick smoothing | Very Fast | Poor |
| Gaussian | Smooth blur | Fast | Poor |
| Median | Salt & Pepper noise | Moderate | Good |
| Bilateral | Preserving edges | Slow | Excellent |
| Morphological | Structure analysis | Fast | Good |
| Sharpening | Detail enhancement | Fast | Good |

## 🤖 CNN Denoiser Model Information

- **Architecture**: Small 4-layer CNN — 3× Conv2D (32 filters, 3×3, ReLU) + Conv2D output layer (~10k parameters)
- **Training**: 300-image dataset, non-overlapping 40×40 patches, 8 epochs, batch size 16, Adam (lr 0.001), MSE loss
- **Variants**: Four trained `.h5` models — Gaussian σ15, σ25, σ35, and Salt & Pepper — committed under `models/`
- **Input**: Grayscale images (1 channel); **Output**: Denoised image (same size)
- **Model selection**: The Streamlit app automatically picks the model matching the noise type and intensity

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
from cnn_denoise import load_keras_cnn_model, denoise_image_keras_cnn

model = load_keras_cnn_model('models/cnn_denoiser_sigma25.h5')
denoised = denoise_image_keras_cnn(noisy_img, model=model)
```

## 📈 Performance Metrics

The application computes, against the clean reference image:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)

Both appear in the comparison table, and Step 03 shows PSNR/SSIM cards for the
CNN output versus the noisy input.

## 🐛 Troubleshooting

### Issue: "No module named 'streamlit'"
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Issue: "Trained model not found"
**Solution**: Run one of the `python train_cnn.py ...` commands above, or pull the committed models from the repo.

### Issue: "Image upload not working"
**Solution**: Ensure file is JPG/PNG format and less than 200MB

### Issue: Filters are too slow
**Solution**: Use smaller images or smaller kernel sizes

## 📚 References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## ✅ Capstone Project Checklist

- [x] Image upload and grayscale conversion
- [x] Noise addition (Gaussian and Salt & Pepper)
- [x] Digital filter implementations (7 filters)
- [x] Enhancement techniques (5 methods)
- [x] Keras CNN denoiser integration (4 trained models)
- [x] PSNR and SSIM quality comparison
- [x] Streamlit UI with sidebar controls
- [x] Image saving to outputs folder
- [x] Modular code structure
- [x] Complete documentation

## 📄 License

This project is provided as-is for educational purposes.

## 👨‍💻 Author

Capstone Project - Image Enhancement and Noise Reduction Using Digital Filters

---

**Last Updated**: October 2026
