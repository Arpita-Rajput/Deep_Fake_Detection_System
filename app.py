from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import logging
from pathlib import Path
import csv

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}  # Images only
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB, matching frontend
MODEL_PATH = 'deepfake_detection_model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
if not Path(MODEL_PATH).is_file():
    logger.error(f"Model file not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise Exception(f"Failed to load model: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(img):
    """Process an image for model input."""
    img = cv2.resize(img, (128, 128))  # Resize to 128x128 to match model input
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize
    logger.debug(f"Processed image shape: {img.shape}")  # Log input shape for debugging
    return img

def interpret_prediction(prediction):
    """Convert model output to label and confidence."""
    try:
        confidence = float(prediction[0])
        label = 'authentic' if confidence > 0.5 else 'deepfake'
        confidence = confidence if label == 'deepfake' else 1 - confidence
    except (IndexError, TypeError):
        prediction = np.array(prediction)
        confidence = float(np.max(prediction))
        label = 'authentic' if np.argmax(prediction) == 0 else 'deepfake'
    logger.debug(f"Prediction output: {prediction}, Label: {label}, Confidence: {confidence}")
    return label, confidence

@app.route('/')
def index():
    logger.info("Rendering index.html")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Predict route called")
    if 'file' not in request.files:
        logger.warning("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({'error': 'No file selected'}), 400

    # Validate file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    if file_size > MAX_FILE_SIZE:
        logger.warning(f"File size exceeds limit: {file_size} bytes")
        return jsonify({'error': 'File size exceeds 10MB'}), 400
    file.seek(0)

    if not allowed_file(file.filename):
        logger.warning(f"Unsupported file type: {file.filename}")
        return jsonify({'error': 'Unsupported file type. Use JPEG or PNG'}), 400

    file_ext = file.filename.rsplit('.', 1)[1].lower()
    try:
        if file_ext in ['jpg', 'jpeg', 'png']:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Invalid image file")
                return jsonify({'error': 'Invalid image file'}), 400
            img = process_image(img)
            prediction = model.predict(img)[0]
            label, confidence = interpret_prediction(prediction)
            logger.info(f"Image prediction: {label}, confidence: {confidence}")
            return jsonify({'prediction': label, 'confidence': confidence})
        else:
            logger.warning(f"Unexpected file extension: {file_ext}")
            return jsonify({'error': 'Unsupported file type'}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f"Processing error: {str(e)}"}), 500

@app.route('/contact', methods=['POST'])
def contact():
    logger.info("Contact route called")
    try:
        first_name = request.form['first-name']
        last_name = request.form['last-name']
        email = request.form['email']
        phone = request.form.get('phone', '')
        company = request.form['company']
        message = request.form.get('message', '')
        logger.info(f"Contact form submitted: {first_name} {last_name}, {email}, {phone}, {company}, {message}")
        
        # Save to contacts.txt with CSV escaping
        try:
            with open('contacts.txt', 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([first_name, last_name, email, phone, company, message])
            logger.info("Contact saved to contacts.txt")
        except IOError as e:
            logger.error(f"Failed to save contact to contacts.txt: {str(e)}")
            return jsonify({'error': f"Failed to save contact: {str(e)}"}), 500
        
        return jsonify({'message': 'Form submitted successfully!'})
    except KeyError as e:
        logger.warning(f"Missing form field: {str(e)}")
        return jsonify({'error': 'Missing required field'}), 400
    except Exception as e:
        logger.error(f"Contact form error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)