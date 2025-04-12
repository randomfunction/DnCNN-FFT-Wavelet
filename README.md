# Image Denoising using DnCNN (Baseline & FNet-Attention Enhanced)

This repository contains two implementations of the classic **DnCNN (Denoising Convolutional Neural Network)** model:

1. **Standard DnCNN** - Based on the original 2017 paper for Gaussian noise removal.
2. **DnCNN + FNet Attention** - Our custom enhancement using **FFT-based attention** inspired by FNet, improving denoising performance via frequency domain mixing.

---

## What is DnCNN?

DnCNN is a deep CNN model that performs **residual learning** for image denoising. It predicts the noise from a noisy image and subtracts it to recover the clean image.

> 📄 Paper: [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (2017)](https://arxiv.org/abs/1608.03981)

---

## What's New? (FNet-style Enhancement)

We introduce **FFT-based token mixing** inside convolutional layers to allow the model to capture **global context** more efficiently.

### Why FNet?

- Lower computational cost than self-attention
- Leverages frequency-domain patterns
- Helps the CNN focus on structured noise

---

## Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**

---

## Results

| Model            | PSNR ↑  | SSIM ↑ |
|------------------|---------|--------|
| DnCNN (Baseline) | 27.93 dB | 0.8975 |
| DnCNN + FNet     | 28.26 dB | 0.9047 |

---

## Author -

- Tanishq Parihar
- Contact: tanishqparihar3@gmail.com
- GitHub: randomfunction
