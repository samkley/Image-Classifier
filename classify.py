import os
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Load the MobileNet model (Assuming it's uploaded to Cloud Storage)
model_path = "gs://your-bucket/mobilenet_model.pb"  # Update this to your Cloud Storage path

# Load the model with TensorFlow
def load_model_from_gcs(model_path):
    try:
        print(f"Loading model from {model_path}...")
        model = tf.saved_model.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load the model: {e}")
        raise

# Load the MobileNet model
model = load_model_from_gcs(model_path)

# Image classification function using MobileNet
def classify_image(img_path):
    try:
        # Load the image with PIL
        img = Image.open(img_path)

        # Convert RGBA to RGB if the image has an alpha channel (transparency)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # Resize the image to the model's expected input shape (224x224 for MobileNet)
        img = img.resize((224, 224))  # MobileNet expects 224x224 images

        # Convert the image to a numpy array
        img_array = np.array(img)

        # Preprocess the image for MobileNet
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess the image for MobileNet

        # Make a prediction
        prediction = model(img_array)

        # Decode the predictions to get human-readable class labels
        decoded_predictions = decode_predictions(prediction, top=1)[0]  # Get top prediction
        class_name = decoded_predictions[0][1]  # Class name
        probability = float(decoded_predictions[0][2])  # Convert to a standard float

        return class_name, probability
    except Exception as e:
        print(f"Error during image classification: {e}")
        return None, None
