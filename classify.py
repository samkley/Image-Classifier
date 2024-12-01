import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Function to download the model from a URL
def download_model(url, file_path):
    if not os.path.exists(file_path):
        try:
            print(f"Downloading model from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise HTTPError for bad responses
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Model download complete.")
        except requests.RequestException as e:
            print(f"Failed to download the model: {e}")
            raise

# URL of the model file (the one you provided)
model_url = "https://www.dropbox.com/scl/fi/et7cap9fc4sb8tl8ykl1d/vgg16_model.h5?rlkey=h1qnvnu9rjenkox1c4tcaqrek&st=e3mqkxl7&dl=1"
model_path = "vgg16_model.h5"

# Ensure the model is downloaded and loaded
download_model(model_url, model_path)

# Load the model with error handling
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load the model: {e}")
    raise

def classify_image(img_path):
    try:
        # Load the image with PIL
        img = Image.open(img_path)
        
        # Convert RGBA to RGB if the image has an alpha channel (transparency)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Resize the image to the model's expected input shape (224x224 for VGG16)
        img = img.resize((224, 224))  # VGG16 expects 224x224 images

        # Convert the image to a numpy array
        img_array = np.array(img)

        # Normalize the image for VGG16
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess the image for VGG16

        # Make a prediction
        prediction = model.predict(img_array)

        # Decode the predictions to get human-readable class labels
        decoded_predictions = decode_predictions(prediction, top=1)[0]  # Get top prediction
        class_name = decoded_predictions[0][1]  # Class name
        probability = float(decoded_predictions[0][2])  # Convert to a standard float

        return class_name, probability
    except Exception as e:
        print(f"Error during image classification: {e}")
        return None, None
