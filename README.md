# Image-Classification-Cats-vs-Dogs

## Overview
This project implements an **Image Classification** model to distinguish between **cats** and **dogs** using a **Support Vector Machine (SVM)** classifier. The dataset consists of labeled images of cats and dogs, which are processed and trained to achieve optimal classification accuracy.

## Features
- **Dataset Preprocessing:** Image resizing, grayscale conversion, and feature extraction.
- **Feature Extraction:** Using Histogram of Oriented Gradients (HOG), SIFT, or pixel intensities.
- **SVM Model Training:** Support Vector Machine with linear/RBF kernel.
- **Evaluation Metrics:** Accuracy, Precision, Recall, and Confusion Matrix.
- **Visualization:** Data distribution and model performance plots.

## Dataset
- The dataset consists of images of cats and dogs.
- Can be sourced from **Kaggle** or manually curated.
- Images are resized and converted into grayscale before feature extraction.

## Dependencies
Ensure you have the following libraries installed:
```bash
pip install numpy pandas matplotlib scikit-learn opencv-python
```

## Installation & Usage
1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Image-Classification-Cats-vs-Dogs-SVM.git
cd Image-Classification-Cats-vs-Dogs-SVM
```
2. **Run the script:**
```bash
python train.py
```
3. **Predict using the trained model:**
```bash
python predict.py --image test.jpg
```

## Model Training
- The dataset is split into **training** and **testing** sets.
- Feature extraction is performed to transform images into numerical representations.
- An **SVM classifier** is trained using the extracted features.
- Hyperparameter tuning (e.g., kernel selection, regularization) is done for optimization.

## Results & Performance
- The model is evaluated using accuracy, precision, recall, and confusion matrix.
- Example output:
  - Accuracy: **90%** (varies based on dataset and feature extraction method)

## Future Improvements
- Implement deep learning-based CNN models for better accuracy.
- Optimize feature extraction techniques for SVM.
- Extend dataset for better generalization.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
