import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
from pymongo import MongoClient
from datetime import datetime

# Flask app setup
app = Flask(__name__, static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.JPG', '.jpeg', '.png']
app.config['UPLOAD_PATH'] = './static/images/uploads/'

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB URI
db = client['plant_disease_db']  # Database name
collection = db['detections']  # Collection name

# Define the classes
n_classes = 5
Class_Penyakit = ['Whitefly', 'Yellowish', 'Healthy', 'Leaf Curl', 'Leaf Spot']

# Pesticide recommendations
def suggest_pesticide(disease):
    if disease == "Leaf Curl":
        return """Organic Pesticide: 1.Neem Oil: Neem oil can help control pests like aphids and whiteflies that cause leaf curl.
                2.Chili-Garlic Spray: Blend 5-6 red chillies and 10 garlic cloves in 1 liter of water, strain, and spray on the plants.
                3.Proper Watering: Ensure consistent watering to avoid stress-induced leaf curl."""
    elif disease == "Leaf Spot":
        return """Organic Pesticide: 1.Baking Soda Solution: Mix 1 teaspoon of baking soda, 1 liter of water, and a few drops of liquid soap. Spray on the affected leaves.
                2.Copper Fungicide: Use organic copper-based fungicides to control fungal leaf spots.
                3.Neem Oil: Neem oil also has antifungal properties and can help manage leaf spot."""
    elif disease == "Whitefly":
        return """Organic Pesticide: 1.Neem Oil: Mix 2-5 ml of neem oil with 1 liter of water and a few drops of liquid soap. Spray on the affected plants every 7-10 days.
                2.Garlic Spray: Blend 10-12 garlic cloves with 1 liter of water, strain, and spray on the plants.
                3.Yellow Sticky Traps: Use yellow sticky traps to catch adult whiteflies.
                4.Insecticidal Soap: Use a mild soap solution (1-2%) to spray on the undersides of leaves where whiteflies reside."""
    elif disease == "Yellowish":
        return """Organic Pesticide: 1.Compost Tea: Apply compost tea to improve soil health and nutrient availability.
                2.Epsom Salt Solution: Mix 1 tablespoon of Epsom salt in 1 liter of water and spray on the leaves to address magnesium deficiency.
                3.Seaweed Extract: Use seaweed extract as a foliar spray to boost plant immunity and nutrient uptake."""
    elif disease == "Healthy":
        return "The plant is healthy. No pesticide required."
    else:
        return "No matching disease found."

# Load the pre-trained model
model_path = "./clm.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found!")
    
model = tf.keras.models.load_model(model_path, compile=False)  # Load the entire model
model.summary()

# Function to preprocess image using OpenCV
def preprocess_image(image_path):
    """Preprocess the image for model prediction using OpenCV"""
    if not os.path.exists(image_path):
        print("Error: Image not found at", image_path)
        return None

    image = cv2.imread(image_path)  # Read image
    if image is None:
        print("Error: Unable to read image")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (256, 256))  # Resize to match model input
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Expand dimensions for batch
    return image

# Flask Routes
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    hasil_prediksi = '(none)'
    confidence_score = 0.0
    gambar_prediksi = '(none)'
    pesticide_recommendation = 'No recommendation available.'

    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(uploaded_file.filename)
    file_ext = os.path.splitext(filename)[1]
    gambar_prediksi = os.path.join(app.config['UPLOAD_PATH'], filename)

    # Ensure the upload directory exists
    if not os.path.exists(app.config['UPLOAD_PATH']):
        os.makedirs(app.config['UPLOAD_PATH'])

    if file_ext.lower() in app.config['UPLOAD_EXTENSIONS']:
        uploaded_file.save(gambar_prediksi)

        # Preprocess image
        image_array = preprocess_image(gambar_prediksi)
        if image_array is None:
            return jsonify({"error": "Failed to process image"}), 400

        try:
            # Model prediction
            y_pred = model.predict(image_array)
            y_pred_class = np.argmax(y_pred, axis=1)[0]
            hasil_prediksi = Class_Penyakit[y_pred_class]
            confidence_score = round(float(np.max(y_pred)), 4)

            # Suggest pesticide for diseased plants
            if hasil_prediksi != "Healthy":
                pesticide_recommendation = suggest_pesticide(hasil_prediksi)

            # Store the result in MongoDB
            detection_record = {
                "filename": filename,
                "prediction": hasil_prediksi,
                "confidence": confidence_score,
                "pesticide_suggestion": pesticide_recommendation,
                "timestamp": datetime.now()
            }
            collection.insert_one(detection_record)

        except Exception as e:
            return jsonify({"error": f"Error during model prediction: {str(e)}"}), 500

        return jsonify({
            "prediksi": hasil_prediksi,
            "confidence": confidence_score,
            "gambar_prediksi": gambar_prediksi,
            "pesticide_suggestion": pesticide_recommendation
        })

    return jsonify({"error": "Invalid file format"}), 400

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)