
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model (Hugging Face caches this)
model = tf.keras.models.load_model('../models/final_model.keras')

# Class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def preprocess_image(image):
    """Resize, normalize, and prepare image for model"""
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    """Predict class and return probabilities"""
    processed_img = preprocess_image(image)
    preds = model.predict(processed_img)[0]
    return {CLASS_NAMES[i]: float(preds[i]) for i in range(10)}

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(label="Predicted Class Probabilities"),
    examples=[
        "airplane.jpeg"
    ],
    title="🚀 CIFAR-10 Image Classifier",
    description="Upload an image to classify it into one of 10 CIFAR-10 categories."
)

demo.launch()

