EffiDerm: An Efficient Deep Learning Model for Skin Cancer Prediction

📌 Overview

EffiDerm is a lightweight Convolutional Neural Network (CNN) designed for skin cancer detection and classification.
It classifies 7 types of skin lesions from dermoscopic images with high accuracy while maintaining a low computational footprint.

Unlike heavy models such as ResNet-50 or VGG16, EffiDerm has only ~1.4M parameters, making it suitable for mobile deployment and real-time usage in resource-limited healthcare environments.

✨ Features

🔹 Lightweight CNN with only ~1.4M parameters

🔹 93.07% classification accuracy

🔹 Inference time ~47 ms per image

🔹 Memory footprint ~5.6 MB (mobile-friendly)

🔹 Class balancing with SMOTE for rare lesion types

🔹 Data augmentation to improve generalization

🔹 Grad-CAM visualizations for explainability

🔹 Deployable in VS Code, local GPUs, or mobile frameworks

📊 Dataset

We use the HAM10000 dataset (Human Against Machine with 10,000 training images).

Total Images: 10,015 dermatoscopic images

Classes: 7 skin lesion categories

Format: RGB, 600×450 px (resized to 64×64 px)

Source: Harvard HAM10000 Dataset

🛠️ Tech Stack

Programming Language: Python 3.8

Frameworks: TensorFlow 2.x, Keras

Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn (SMOTE), OpenCV

Development Environment: Visual Studio Code

⚙️ Methodology

Preprocessing

Resize all images to 64×64×3

Normalize pixel values to [0, 1]

One-hot encode labels

Apply SMOTE to handle class imbalance

Data Augmentation

Random rotation, zoom, shift

Horizontal & vertical flipping

Brightness adjustment

Model Training

CNN with Conv2D, MaxPooling, Dropout, BatchNormalization

Optimizer: Adamax

Loss: Categorical Crossentropy with Focal Loss

Epochs: 50, Batch size: 32

Evaluation

Metrics: Accuracy, Precision, Recall, F1-score

Confusion Matrix

Training/Validation Curves

Grad-CAM Heatmaps

📈 Results

Accuracy: 93.07%

Inference Time: ~47 ms

Parameters: ~1.4M

Memory: ~5.6 MB

🌍 Applications

Clinical decision-support tool for dermatologists

Mobile-based early skin cancer screening app

Telemedicine integration for rural healthcare

Educational tool for dermatology students

⚠️ Limitations

Dataset contains only dermoscopic images (not smartphone-quality photos)

Model performance depends on dataset quality and diversity

Requires further validation in real-world clinical settings

🔮 Future Scope

Integration with explainable AI (better Grad-CAM visualizations)

Deployment in mobile and edge devices

Expansion to include real-world smartphone datasets

Cloud-based teledermatology platforms

👨‍💻 Author

Agil Kannan

🎓 B.Tech IT Student (2021–2025)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Agil%20Kannan-blue?logo=linkedin)](https://www.linkedin.com/in/agilkannan)  
[![GitHub](https://img.shields.io/badge/GitHub-Agilkannan-black?logo=github)](https://github.com/Agilkannan)  


✨ If you like this project, don’t forget to ⭐ the repo!
