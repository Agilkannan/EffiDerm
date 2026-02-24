<div align="center">

# EffiDerm

### An Efficient Deep Learning Model for Skin Cancer Prediction

![Python](https://img.shields.io/badge/Python-3.8-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Accuracy: 93.07% &nbsp;|&nbsp; Parameters: ~1.4M &nbsp;|&nbsp; Inference: ~47 ms &nbsp;|&nbsp; Size: ~5.6 MB**

[Overview](#overview) &bull; [Features](#key-features) &bull; [Dataset](#dataset) &bull; [Architecture](#model-architecture) &bull; [Results](#results) &bull; [Getting Started](#getting-started)

</div>

---

## Overview

Skin cancer is one of the most prevalent cancers globally, and early detection is critical for improving patient outcomes. **EffiDerm** is a lightweight Convolutional Neural Network (CNN) engineered for automated skin lesion classification from dermoscopic images.

The model classifies **7 distinct types of skin lesions** with **93.07% accuracy** while maintaining an exceptionally low computational footprint — only **~1.4M parameters** and **~5.6 MB** in memory. This makes EffiDerm significantly more efficient than conventional architectures such as ResNet-50 (~25.6M params) or VGG16 (~138M params), enabling real-time inference and deployment in resource-constrained healthcare environments.

---

## Key Features

| Feature | Detail |
|:---|:---|
| **Lightweight Architecture** | ~1.4M parameters — 18× smaller than ResNet-50 |
| **High Accuracy** | 93.07% classification accuracy across 7 lesion types |
| **Fast Inference** | ~47 ms per image — suitable for real-time applications |
| **Mobile-Ready** | ~5.6 MB memory footprint for edge/mobile deployment |
| **Class Balancing** | SMOTE oversampling to address rare lesion class imbalance |
| **Data Augmentation** | Rotation, zoom, shift, and flipping for robust generalization |
| **Explainability** | Grad-CAM heatmap visualizations for interpretable predictions |

---

## Dataset

This project uses the [**HAM10000**](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) (*Human Against Machine with 10,000 training images*) benchmark dataset.

| Property | Details |
|:---|:---|
| **Total Samples** | 10,015 dermatoscopic images |
| **Classes** | 7 skin lesion categories |
| **Original Resolution** | 600 × 450 px (RGB) |
| **Input Resolution** | Resized to 64 × 64 × 3 |

### Supported Lesion Classes

| ID | Code | Lesion Type |
|:---:|:---:|:---|
| 0 | `nv` | Melanocytic Nevi |
| 1 | `mel` | Melanoma |
| 2 | `bkl` | Benign Keratosis-like Lesions |
| 3 | `bcc` | Basal Cell Carcinoma |
| 4 | `akiec` | Actinic Keratoses |
| 5 | `vasc` | Vascular Lesions |
| 6 | `df` | Dermatofibroma |

---

## Model Architecture

EffiDerm follows a sequential CNN design optimized for a balance between accuracy and computational efficiency.

```
Input (64×64×3)
  │
  ├─ Conv2D(32) → MaxPool → BatchNorm
  ├─ Conv2D(64) × 2 → MaxPool → BatchNorm
  ├─ Conv2D(128) × 2 → MaxPool → BatchNorm
  ├─ Conv2D(256) × 2 → MaxPool
  │
  ├─ Flatten → Dropout(0.2)
  ├─ Dense(256) → BatchNorm
  ├─ Dense(128) → BatchNorm
  ├─ Dense(64) → BatchNorm
  ├─ Dense(32) → BatchNorm (L1L2 Regularization)
  │
  └─ Dense(7, softmax) → Output
```

---

## Methodology

### 1. Preprocessing

- Resized all images to **64 × 64 × 3**
- Normalized pixel values to the range **[0, 1]**
- Applied **one-hot encoding** to target labels
- Used **SMOTE** (Synthetic Minority Oversampling Technique) to balance underrepresented classes

### 2. Data Augmentation

| Technique | Configuration |
|:---|:---|
| Rotation | ±10° |
| Zoom | ±10% |
| Width/Height Shift | ±10% |
| Horizontal/Vertical Flip | Enabled |

### 3. Training Configuration

| Parameter | Value |
|:---|:---|
| **Optimizer** | Adamax (lr = 0.001) |
| **Loss Function** | Categorical Crossentropy |
| **Epochs** | 50 |
| **Batch Size** | 32 |
| **LR Scheduler** | ReduceLROnPlateau (patience=2, factor=0.5) |
| **Train/Test Split** | 75% / 25% |

### 4. Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Training & Validation Loss/Accuracy Curves
- Grad-CAM Heatmap Visualizations

---

## Results

<div align="center">

| Metric | Value |
|:---:|:---:|
| **Classification Accuracy** | 93.07% |
| **Inference Time** | ~47 ms/image |
| **Total Parameters** | ~1.4M |
| **Model Size** | ~5.6 MB |

</div>

---

## Tech Stack

| Component | Technology |
|:---|:---|
| **Language** | Python 3.8 |
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Data Processing** | NumPy, Pandas, OpenCV |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, Imbalanced-learn |
| **IDE** | Visual Studio Code |

---

## Getting Started

### Prerequisites

```bash
Python >= 3.8
TensorFlow >= 2.x
```

### Installation

```bash
# Clone the repository
git clone https://github.com/agilkannan/EffiDerm.git
cd EffiDerm

# Install dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn imbalanced-learn opencv-python
```

### Usage

1. Place the HAM10000 dataset CSV in `./dataset/hmnist_28_28_RGB.csv`
2. Open and run `main.ipynb` sequentially
3. The trained model will be saved as `skin_cancer_model.h5`

---

## Applications

- **Clinical Decision Support** — Assist dermatologists with automated lesion classification
- **Mobile Screening** — Lightweight enough for on-device skin cancer screening apps
- **Telemedicine** — Enable remote dermatological assessment in underserved areas
- **Medical Education** — Interactive learning tool for dermatology students and residents

---

## Limitations

- The model is trained exclusively on dermoscopic images; performance on smartphone-captured images has not been validated
- Classification accuracy is dependent on the quality and diversity of the training dataset
- Clinical deployment requires further validation through prospective studies and regulatory approval

---

## Future Work

- Enhanced explainability through advanced Grad-CAM and attention-based visualizations
- Model optimization for mobile and edge deployment (TFLite, ONNX)
- Extension to smartphone-captured image datasets for broader applicability
- Integration with cloud-based teledermatology platforms for scalable screening

---

## Citation

If you use this work in your research, please cite:

```
@misc{effiderm2026,
  author       = {Agil Kannan},
  title        = {EffiDerm: An Efficient Deep Learning Model for Skin Cancer Prediction},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/agilkannan/EffiDerm}
}
```

---

<div align="center">

## Author

**Agil Kannan**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/agilkannan)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/agilkannan)

---

If you find this project useful, please consider giving it a ⭐

</div>
