import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2  # OpenCV for image resizing
import requests

# Load the corrected class labels from the JSON file
with open('corrected_imagenet_class_index.json', 'r') as f:
    class_labels = json.load(f)

# Load your model
model = tf.keras.models.load_model('mobilenet_model.keras')

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image for the model
def preprocess_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Resize the image to 224x224 (required by the model)
    img = cv2.resize(img, (224, 224))

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to [0, 1]
    img_array = np.array(img_rgb) / 255.0

    # Add batch dimension (shape: [1, 224, 224, 3])
    img_array = np.expand_dims(img_array, axis=0)

    # Ensure data type is float32
    img_array = img_array.astype(np.float32)

    return img_array

# Classify the image and map the prediction
def classify_image(image_path):
    img_array = preprocess_image(image_path)

    # Predict the class
    predictions = model.predict(img_array)

    # Get the class index (the index of the highest prediction)
    class_index = np.argmax(predictions, axis=1)[0]

    # Get the class name from the corrected class labels
    class_name = class_labels[str(class_index)]

    # Get the probability of the prediction and convert it to a native Python float
    probability = float(np.max(predictions))

    return class_name, probability

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filename)

        try:
            class_name, probability = classify_image(filename)
            return jsonify({
                'class_name': class_name,
                'probability': probability,
                'image_url': f"/{UPLOAD_FOLDER}/{file.filename}"
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File type not allowed"}), 400

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
