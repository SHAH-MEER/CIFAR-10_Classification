# 🧠 CIFAR-10 Image Classifier with CNN

This repository contains a Convolutional Neural Network (CNN) implementation and deployment for classifying images in the CIFAR-10 dataset using TensorFlow, Keras, and Gradio, hosted on Hugging Face Spaces.

## 📋 Overview

The CIFAR-10 dataset includes 60,000 32x32 color images across 10 categories. This project utilizes a deep CNN architecture with data augmentation and regularization to achieve high accuracy. A user-friendly interface is deployed using Gradio for real-time image classification.

## 🗂️ Dataset

The CIFAR-10 dataset includes the following classes:

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

CIFAR-10_Classification/
├── app/                    # Gradio web app for model inference
├── models/                 # Trained and saved model files
├── notebooks/              # Jupyter notebook for training and experimentation
├── visualizations/         # Plots and visual outputs from training
├── .gitignore              # Specifies intentionally untracked files to ignore
├── README.md               # Project documentation


## 🏗️ Model Architecture

The implemented CNN includes:

- Convolutional layers with Batch Normalization
- MaxPooling layers for downsampling
- Dropout layers to prevent overfitting
- Global Average Pooling to reduce dimensionality
- Dense layers with L2 regularization

## ✨ Features

- **Data Preprocessing**: Normalization and train/validation split
- **Data Augmentation**: Rotation, shift, and flip transformations
- **Regularization**: Dropout, BatchNorm, L2 regularization
- **Callbacks**:
  - ReduceLROnPlateau
  - EarlyStopping
  - ModelCheckpoint
- **Interactive Deployment**: Real-time predictions with Gradio

## 📦 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Gradio

## 🚀 Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model (optional)

```bash
jupyter notebook CIFAR-10_Classification.ipynb
```

### Run the Gradio app

```bash
python app.py
```

### Live Demo

Check out the app on Hugging Face Spaces: [🔗 View Demo](https://huggingface.co/spaces/your-space-url)

## 📊 Results

- Competitive accuracy on the CIFAR-10 test set
- Enhanced generalization via data augmentation
- Reduced overfitting through regularization

Visual outputs and performance plots are available in the `visualizations/` folder.

## ⚙️ Customization

You can modify hyperparameters in the notebook or script:

- `BATCH_SIZE`
- `EPOCHS`
- Learning rate & optimizer settings
- Model architecture

## 📭 Contact

- Email: [shahmeershahzad67@gmail.com](mailto\:shahmeershahzad67@gmail.com)
- GitHub: [SHAH-MEER](https://github.com/SHAH-MEER)

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- CIFAR-10 dataset: Alex Krizhevsky, Vinod Nair, Geoffrey Hinton
- TensorFlow & Keras teams
- Gradio & Hugging Face for deployment tools

