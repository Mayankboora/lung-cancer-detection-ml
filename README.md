# Lung_Cancer_Detection_Using_AIML
Dataset link:https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset

# Lung Cancer Detection Using AI/ML

## Overview
This project leverages deep learning and computer vision to automatically detect and classify lung cancer from medical images. Using a convolutional neural network (CNN) based on the VGG16 architecture, the model distinguishes between three classes: Normal, Benign, and Malignant lung cases.

## Features
- **Automated Image Classification:** Classifies lung images into Normal, Benign, or Malignant categories.
- **Transfer Learning:** Utilizes a pre-trained VGG16 model for robust feature extraction.
- **Data Augmentation:** Improves model generalization with light augmentation techniques.
- **Performance Visualization:** Includes confusion matrix, ROC curves, and training/validation accuracy/loss plots.
- **Random Image Testing:** Predicts and visualizes the class of a random test image.

## Dataset
- **Source:** The IQ-OTHNCCD lung cancer dataset (colour_data)
- **Classes:**
	- Normal cases
	- Benign cases
	- Malignant cases
- **Preprocessing:**
	- Images are converted to grayscale, resized to 160x160 pixels, and normalized.
	- Labels are one-hot encoded.

## Model Architecture
- **Base Model:** VGG16 (pre-trained on ImageNet, top layers initially frozen)
- **Custom Layers:**
	- Flatten
	- Dense (128 units, ReLU activation)
	- Dropout (0.4)
	- Output Dense (softmax for 3 classes)

## Training Procedure
1. **Phase 1:** Train only the custom layers (VGG16 frozen) for 20 epochs on clean images.
2. **Phase 2:** Unfreeze top layers of VGG16, apply light data augmentation, and fine-tune for 20 more epochs.

## Evaluation
- **Metrics:** Accuracy, confusion matrix, classification report, ROC curves.
- **Visualization:** Training/validation accuracy and loss plots, confusion matrix heatmap, ROC curves for each class.

## Usage
1. **Update Data Path:**
	 - Set `data_dir` in the notebook to the path containing the dataset folders (`Normal cases`, `Bengin cases`, `Malignant cases`).
2. **Run the Notebook:**
	 - Execute all cells in `Lung_Cancer_detection.ipynb`.
3. **Model Output:**
	 - The trained model is saved as `lung_cancer_vgg16_model.h5`.
	 - Evaluation metrics and plots are displayed in the notebook.

## Requirements
- Python 3.7+
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install dependencies with:
```bash
pip install tensorflow keras opencv-python numpy matplotlib seaborn scikit-learn
```

## Results
- The model achieves high accuracy in classifying lung images.
- Visualizations help interpret model performance and areas for improvement.

## Acknowledgements
- [The IQ-OTHNCCD lung cancer dataset](https://www.kaggle.com/datasets/andrewmvd/lung-cancer-dataset)
- VGG16 model authors and TensorFlow/Keras community

## License
This project is for educational and research purposes only. Please refer to the dataset's license for data usage rights.
