import os
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
from google.cloud import storage
from classify import classify_image

# Set environment variable to disable GPU (for non-GPU environments)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')  # Ensure no GPU is used

app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# GCS model path
GCS_BUCKET_NAME = 'imageclassifier'
GCS_MODEL_PATH = 'gs://imageclassifier/vgg16_model.keras'

# Path to save the model locally
MODEL_PATH = 'vgg16_model.keras'

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to download the model from GCS if it doesn't exist locally
def download_model_from_gcs():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from GCS...")

        try:
            # Set up the GCS client
            client = storage.Client()
            bucket = client.get_bucket(GCS_BUCKET_NAME)

            # Download the model file from GCS to local storage
            blob = bucket.blob(GCS_MODEL_PATH)
            blob.download_to_filename(MODEL_PATH)

            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model from GCS: {e}")

# Load the model when the application starts
download_model_from_gcs()
model = load_model(MODEL_PATH)

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
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Create folder if it doesn't exist
        file.save(filename)

        try:
            # Classify the image using the downloaded model
            class_name, probability = classify_image(filename)
            if class_name is None:
                return jsonify({"error": "Error during image classification"}), 500

            # Return the result to be displayed on the same page
            return jsonify({
                'class_name': class_name,
                'probability': probability,
                'image_url': f"/{UPLOAD_FOLDER}/{file.filename}"
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            K.clear_session()  # Clear the session after classification to free memory

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
