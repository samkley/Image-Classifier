import os
import json
from PIL import Image
import cv2  # OpenCV for image resizing
import requests
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import io

# Disable GPU and force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

print("TensorFlow devices:", tf.config.list_physical_devices("GPU"))  # Should print an empty list

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Endpoint details
ENDPOINT_URL = (
    "https://us-central1-aiplatform.googleapis.com/v1/projects/630534982182/"
    "locations/us-central1/endpoints/8938599624772419584:predict"
)

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Get an access token using a service account
def get_access_token():
    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not credentials_json:
        raise Exception("Service account credentials not found in environment variables.")
    credentials_dict = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    credentials.refresh(Request())
    return credentials.token

def preprocess_image(image_path):
    # Open the image using OpenCV (grayscale will be avoided)
    img = cv2.imread(image_path)
    
    # Resize the image to reduce resolution (keeping the aspect ratio)
    img = cv2.resize(img, (128, 128))  # Resize to 128x128 pixels (or adjust as needed)
    
    # Convert the image to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize the image (optional depending on model requirements)
    img_array = np.array(img_rgb) / 255.0  # Normalize the pixel values between 0 and 1
    
    # Add batch dimension (making it [1, height, width, channels])
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Convert to float32 (optional depending on model requirements)
    img_array = img_array.astype(np.float32)

    return img_array

# Classify the image using the endpoint
def classify_image_with_endpoint(image_path):
    try:
        # Preprocess the image
        img_array = preprocess_image(image_path)
        print("Image array shape:", img_array.shape)
        
        # Prepare the payload
        instances = [{"input_layer": img_array.tolist()}]
        payload = {"instances": instances}

        # Log the size of the payload data
        json_payload = json.dumps(payload)
        print(f"Raw JSON payload size: {len(json_payload.encode('utf-8')) / 1024:.2f} KB")

        # Get the access token
        access_token = get_access_token()

        # Set headers (no gzip compression)
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        # Send the POST request (without gzip)
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers)

        # Handle response
        if response.status_code == 200:
            predictions = response.json()
            print("Prediction response:", predictions)
            return predictions["predictions"][0].get("output_0"), predictions["predictions"][0].get("probability")
        else:
            print("Error response:", response.text)
            raise Exception(f"Prediction request failed: {response.text}")
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise e

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
            class_name, probability = classify_image_with_endpoint(filename)
            if class_name is None or probability is None:
                return jsonify({"error": "Prediction failed: No response from model"}), 500

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

