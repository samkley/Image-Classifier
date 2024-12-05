import os
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras import backend as K  # Clear backend memory
import tensorflow as tf

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Explicitly disable GPU
tf.config.set_visible_devices([], "GPU")  # Ensure TensorFlow doesn't use GPU

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

# Function to send an image to the AI Platform endpoint for classification
def classify_image_with_endpoint(image_path):
    with open(image_path, "rb") as image_file:
        image_content = image_file.read()

        # Prepare the request payload
        instances = [{"image_bytes": {"b64": image_content.decode('latin1')}, "key": "value"}]

        # Get the Google Cloud access token
        access_token = os.popen("gcloud auth print-access-token").read().strip()

        # Set headers and payload for the request
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        payload = {"instances": instances}

        # Send request to AI Platform endpoint
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
        if response.status_code == 200:
            predictions = response.json()
            return predictions["predictions"][0]["class_name"], predictions["predictions"][0]["probability"]
        else:
            raise Exception(f"Prediction request failed: {response.text}")

@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' exists in 'templates'

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure folder exists
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            # Classify the image via AI Platform endpoint
            class_name, probability = classify_image_with_endpoint(file_path)

            return jsonify({
                'class_name': class_name,
                'probability': probability,
                'image_url': f"/{UPLOAD_FOLDER}/{file.filename}"
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            K.clear_session()  # Clear the session after classification

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

    # Handle feedback (e.g., store in a database or log it)
    print(f"Feedback received: {feedback}")
    print(f"Class: {class_name}, Probability: {probability}, Image: {image_path}")

    return jsonify({"success": True})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
