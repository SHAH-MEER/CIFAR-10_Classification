# 🖼️ CNN Model for CIFAR-10 Classification

This repository contains a Convolutional Neural Network (CNN) implementation for classifying images in the CIFAR-10 dataset using TensorFlow and Keras.

## 📋 Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The model uses a deep CNN architecture with data augmentation, regularization techniques, and proper training strategies to achieve high accuracy.

## 🗂️ Dataset

CIFAR-10 includes the following classes:
- ✈️ airplane
- 🚗 automobile
- 🐦 bird
- 🐱 cat
- 🦌 deer
- 🐕 dog
- 🐸 frog
- 🐎 horse
- 🚢 ship
- 🚚 truck

## 📁 Repository Structure

```
CIFAR-10_Classification/
├── models/                  # Saved model files
├── visualizations/          # Plots and visual outputs
├── .gitignore               # Git ignore file
├── CIFAR-10_Classification.ipynb  # Main Jupyter notebook with implementation
└── README.md                # This file
```

## 🏗️ Model Architecture

The implemented CNN has the following structure:
- Multiple convolutional layers with batch normalization
- MaxPooling layers for dimensionality reduction
- Dropout layers to prevent overfitting
- Global Average Pooling to reduce parameters
- Dense layers with L2 regularization

## ✨ Features

- **Data Preprocessing**: Normalization and train/validation split
- **Data Augmentation**: Rotation, width/height shifts, and horizontal flips to improve model generalization
- **Regularization Techniques**: Dropout, Batch Normalization, and L2 regularization
- **Advanced Training Strategies**: 
  - Learning rate reduction on plateau
  - Early stopping
  - Model checkpointing

## 📦 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## 🚀 Usage

### Installation

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### Running the Notebook

1. Clone this repository
   ```bash
   git clone https://github.com/SHAH-MEER/CIFAR-10_Classification.git
   cd CIFAR-10_Classification
   ```

2. Open the Jupyter notebook
   ```bash
   jupyter notebook CIFAR-10_Classification.ipynb
   ```

3. Run the cells in the notebook to:
   - Load and preprocess the CIFAR-10 dataset
   - Set up data augmentation
   - Build and compile the model
   - Train with callbacks for optimization
   - Evaluate performance

## 📊 Results

The model achieves competitive accuracy on the CIFAR-10 test set with:
- Effective learning through the implemented CNN architecture
- Reduced overfitting via regularization techniques
- Improved generalization through data augmentation

Visualizations of the training process and model performance can be found in the `visualizations/` directory.

## ⚙️ Customization

You can adjust various hyperparameters in the notebook:
- `BATCH_SIZE`: Number of samples per gradient update (default: 128)
- `EPOCHS`: Maximum number of training epochs (default: 50)
- Learning rate and optimizer settings
- Network architecture (layers, filters, etc.)

## 📭 Contact

- Email: shahmeershahzad67@gmail.com
- GitHub: [SHAH-MEER](https://github.com/SHAH-MEER)

## 📜 License

[MIT License](LICENSE)

## 🙏 Acknowledgements

- The CIFAR-10 dataset was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
