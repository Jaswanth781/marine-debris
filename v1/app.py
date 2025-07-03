from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import rasterio
import os
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__, static_folder="static")
CORS(app)  # Enable CORS

# Load Model, Label Encoder, and Data
with open("tabnet_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

df = pd.read_csv("all_data.csv")

# Define Features
required_features = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
    "NDVI", "FDI", "PI", "NDWI", "WRI", "MNDWI", "SR", "RNDVI", "ARI",
    "MARI", "CHL_RedEdge", "REPI", "EVI", "EVI2", "GNDVI", "MCARI",
    "MSI", "NDMI", "NBR", "NDSI", "SAVI", "OSI", "PNDVI"
]

class_mapping = {
    1: "Water", 2: "Plastic", 3: "Driftwood", 4: "Seaweed",
    5: "Pumice", 6: "Sea Snot", 7: "Sea Foam"
}

natural_classes = {1, 3, 4, 5, 6, 7}  # Natural debris types

# Extract Bands & Validate Data
def extract_bands(image_path):
    extracted_data = {}

    try:
        with rasterio.open(image_path) as src:
            available_bands = src.count
            print(f"✅ Image Loaded: {image_path}, Bands Found: {available_bands}")

            for i in range(available_bands):
                band_name = required_features[i]  # Get corresponding feature name
                extracted_data[band_name] = np.mean(src.read(i + 1))

    except Exception as e:
        print(f"⚠️ Error extracting bands: {e}")
        return None  # Return None if reading fails

    # If no bands were extracted, reject the image
    if not extracted_data:
        return None

    # Fill missing bands with dataset mean
    for feature in required_features:
        if feature not in extracted_data:
            extracted_data[feature] = df[feature].mean()

    return {k: float(v) for k, v in extracted_data.items()}  # Convert NumPy values to Python types

# Route for Frontend
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"})

        image_file = request.files["image"]

        # ✅ Check if file has .tif extension
        if not image_file.filename.lower().endswith(".tif"):
            return jsonify({"error": "Only segmented satellite images (.tif) are allowed."})

        image_path = "temp_image.tif"
        image_file.save(image_path)

        extracted_data = extract_bands(image_path)

        # If no bands were extracted, reject the image
        if extracted_data is None:
            return jsonify({"error": "Only segmented satellite images are supported. Ensure the image contains at least one spectral band."})

        # Convert extracted data to NumPy array
        input_data = np.array([extracted_data[feature] for feature in required_features]).reshape(1, -1)

        prediction = model.predict(input_data)
        predicted_label = prediction[0]
        predicted_class = class_mapping.get(predicted_label, "Unknown")
        classification = "Natural" if predicted_label in natural_classes else "Unnatural"

        return jsonify({
            "predicted_class": predicted_class,
            "classification": classification
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
