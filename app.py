import os
import json
from PIL import Image
import io
import requests
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import base64

# Disable GPU and force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Explicitly disable GPU
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Hide GPU from TensorFlow

# Verify no GPUs are being used
print("TensorFlow devices:", tf.config.list_physical_devices("GPU"))  # Should print an empty list

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Updated AI Platform Endpoint Details
ENDPOINT_URL = (
    "https://us-central1-aiplatform.googleapis.com/v1/projects/630534982182/"
    "locations/us-central1/endpoints/8938599624772419584:predict"
)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to get an access token using a service account
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

# Function to preprocess the image and convert it to a float array
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Ensure it's in RGB format
    img = img.resize((224, 224))  # Resize to 224x224
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = img_array.astype(np.float32)  # Ensure it's float32
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def classify_image_with_endpoint(image_path):
    try:
        # Preprocess image to float type numpy array
        img_array = preprocess_image(image_path)

        # Compress the image to reduce its size
        with open(image_path, "rb") as image_file:
            image = Image.open(image_file)
            image = image.resize((224, 224))  # Resize to the model's expected input size

            # Compress the image to JPEG with a lower quality
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)  # Set quality to 85 to reduce size
            image_content = buffer.getvalue()

        # Check the size of the image content
        print(f"Image content size: {len(image_content)} bytes")
        
        # Ensure the size is within the API limits
        if len(image_content) > 1.5 * 1024 * 1024:  # 1.5 MB
            raise Exception("Image size exceeds the API size limit (1.5 MB)")

        # Encode the image as base64
        image_b64 = base64.b64encode(image_content).decode("utf-8")
        
        # Prepare the instances array with the base64-encoded image
        instances = [{"input_layer": image_b64}]
        
        # Print the instances for debugging purposes
        print("Instances payload:", instances)
        
        # Get the access token
        access_token = get_access_token()
        
        # Set the headers for the API request
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        # Prepare the request payload
        payload = {"instances": instances}

        # Send the POST request to the AI Platform endpoint
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
        
        # Check for successful response
        if response.status_code == 200:
            predictions = response.json()
            print("Prediction response:", predictions)  # Debugging response
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

