# Updated classify.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Load the model (ensure the correct path to your model file)
model = load_model('vgg16_model.h5')

def classify_image(img_path):
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
