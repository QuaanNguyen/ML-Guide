# Chest X-Ray Deep Learning and Classical Image Analysis (Mini JSRT Dataset)

This repository contains Python scripts for performing **classification**, **localization**, and **segmentation** on the **Mini JSRT Chest X-ray dataset**, using both **deep learning architectures** (e.g., ResNet, U-Net) and **classical image analysis** techniques (e.g., Otsu thresholding, Active Contour/Snake).  

The goal is to demonstrate and compare modern deep learning pipelines with traditional computer vision methods for medical image understanding.

---

## 🧠 Overview

| Task | Description | Approach |
|------|--------------|-----------|
| **Classification** | Predicts abnormal vs. normal chest X-rays | Deep Learning (ResNet) |
| **Localization** | Finds approximate lesion regions | CNN-based Regression |
| **Segmentation** | Extracts lung fields or regions of interest | U-Net / Attention U-Net |
| **Classical Analysis** | Traditional segmentation and preprocessing | Otsu Thresholding, Active Contour (Snake) |

---

## 📁 Repository Structure

```
├── classification/          # Deep learning classification scripts (ResNet)
├── localization/            # Localization models and evaluation scripts
├── segmentation/            # U-Net-based segmentation training and testing
├── classical_methods/       # Classical techniques (Otsu, Active Contour/Snake, etc.)
├── data/                    # Mini JSRT dataset (images and mask
└── utils/                   # Helper functions (metrics, loaders, visualization)
```
---

## ⚙️ Requirements

- Python ≥ 3.8  
- PyTorch ≥ 2.0  
- NumPy, OpenCV, Matplotlib, scikit-learn, tqdm

Install dependencies:
```bash
pip install -r requirements.txt
