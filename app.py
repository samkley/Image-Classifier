import os
import json
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from classify import classify_image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
tf.config.set_visible_devices([], "GPU")  


app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# AI Platform Endpoint Details
ENDPOINT_URL = "https://us-central1-aiplatform.googleapis.com/v1/projects/630534982182/locations/us-central1/endpoints/830994395598684160:predict"

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to get an access token using a service account
def get_access_token():
    # Load the service account JSON from the environment variable
    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not credentials_json:
        raise Exception("Service account credentials not found in environment variables.")
    
    # Parse the JSON and create credentials
    credentials_dict = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    credentials.refresh(Request())
    return credentials.token

# Function to send an image to the AI Platform endpoint for classification
def classify_image_with_endpoint(image_path):
    with open(image_path, "rb") as image_file:
        # Prepare payload
        image_content = image_file.read()
        instances = [{"image_bytes": {"b64": image_content.decode('latin1')}, "key": "value"}]  # Adjust instance format if needed

        # Get the access token
        access_token = get_access_token()

        # Set headers and authorization
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "instances": instances
        }

        # Send request to endpoint
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
        if response.status_code == 200:
            predictions = response.json()
            return predictions["predictions"][0]["class_name"], predictions["predictions"][0]["probability"]
        else:
            raise Exception(f"Prediction request failed: {response.text}")

@app.route('/')
def home():
    return render_template('index.html')  # Ensure this file exists in 'templates'

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Save file to the upload folder
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filename)

        try:
            print("Classifying image...")
            # Classify the image using the AI Platform endpoint
            class_name, probability = classify_image_with_endpoint(filename)
            print(f"Classification result: {class_name}, {probability}")

            if class_name is None:
                return jsonify({"error": "Error during image classification"}), 500

            return jsonify({
                'class_name': class_name,
                'probability': probability,
                'image_url': f"/{UPLOAD_FOLDER}/{file.filename}"
            })

        except Exception as e:
            print(f"Error during classification: {e}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File type not allowed"}), 400

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    feedback = data.get('feedback')
    class_name = data.get('class_name')
    probability = data.get('probability')
    image_path = data.get('image_path')

    # Handle the feedback logic here (store it, analyze it, etc.)
    print(f"Feedback received: {feedback}")
    print(f"Class: {class_name}, Probability: {probability}, Image: {image_path}")

    return jsonify({"success": True})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
