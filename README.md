# Hybrid Image Denoising using DnCNN, FNet, and Wavelet Transforms

This project implements advanced image denoising models based on DnCNN, with enhancements using **Wavelet Transforms** and **FFT-based Attention (FNet)**. It explores hybrid architectures for improving denoising performance on noisy images.


## Model Architectures

### 1. **DnCNN (Baseline)**
- Deep convolutional network with residual learning.
- Learns to predict noise from corrupted images.

### 2. **DnCNN + Wavelet**
- Adds Discrete Wavelet Transform (DWT) preprocessing.
- Enhances noise separation by working in frequency domain.

### 3. **HybridTriBranchModel**
- Combines three branches:
  - Standard DnCNN
  - DWT + DnCNN
  - FNet (FFT-based self-attention)
- Merges outputs for robust denoising under various noise types.


