from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import numpy as np
import cv2
import os
import requests
import math

app = Flask(__name__)
CORS(app)


# ===============================
# CONFIG
# ===============================
UPLOAD_FOLDER = "uploads"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.55

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# LOAD MODELS
# ===============================
animal_model = tf.keras.models.load_model("animal_model.h5", compile=False)
dog_disease_model = tf.keras.models.load_model("dog_disease_model.h5", compile=False)
cat_disease_model = tf.keras.models.load_model("cat_disease_model.h5", compile=False)

# animal_model = tf.keras.models.load_model("animal_model.h5")
# dog_disease_model = tf.keras.models.load_model("dog_disease_model.h5")
# cat_disease_model = tf.keras.models.load_model("cat_disease_model.h5")

# ===============================
# CLASS NAMES
# ===============================
ANIMAL_CLASSES = ["cat", "dog"]

DOG_DISEASE_CLASSES = [
    "eye_problem",
    "parasite_problem",
    "skin_problem"
]

CAT_DISEASE_CLASSES = [
    "ear_problem",
    "eye_problem",
    "skin_problem"
]

# ===============================
# ADVICE MAPS
# ===============================
DOG_ADVICE = {
    "skin_problem": "Possible skin-related issue. Keep the area clean and consult a veterinarian if it worsens.",
    "parasite_problem": "Possible parasite-related issue. Check for ticks and consult a veterinarian.",
    "eye_problem": "Possible eye-related issue. Clean gently and consult a veterinarian if symptoms persist."
}

CAT_ADVICE = {
    "skin_problem": "Possible skin-related issue. Keep the area clean and dry. Consult a veterinarian.",
    "ear_problem": "Possible ear-related issue. Avoid touching inside the ear and consult a veterinarian.",
    "eye_problem": "Possible eye-related issue. Clean gently and consult a veterinarian."
}

# ===============================
# IMAGE PREPROCESS
# ===============================
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ===============================
# DISTANCE CALCULATION
# ===============================
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in KM

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)

# ===============================
# ROUTES
# ===============================
@app.route("/")
def home():
    return jsonify({"message": "Pet Disease Detection Backend Running"})

# -------- PREDICT ROUTE --------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    img = preprocess_image(image_path)

    # PHASE 1: ANIMAL
    animal_pred = animal_model.predict(img)
    animal_index = np.argmax(animal_pred)
    animal = ANIMAL_CLASSES[animal_index]
    animal_conf = float(animal_pred[0][animal_index])

    # PHASE 2: DISEASE (PROBLEM TYPE)
    if animal == "dog":
        disease_pred = dog_disease_model.predict(img)
        disease_index = np.argmax(disease_pred)
        disease = DOG_DISEASE_CLASSES[disease_index]
        disease_conf = float(disease_pred[0][disease_index])
        advice = DOG_ADVICE.get(disease)

    else:
        disease_pred = cat_disease_model.predict(img)
        disease_index = np.argmax(disease_pred)
        disease = CAT_DISEASE_CLASSES[disease_index]
        disease_conf = float(disease_pred[0][disease_index])
        advice = CAT_ADVICE.get(disease)

    if disease_conf < CONFIDENCE_THRESHOLD:
        disease = "uncertain"
        advice = "Low confidence prediction. Please consult a veterinarian."

    return jsonify({
        "animal": animal,
        "animal_confidence": round(animal_conf * 100, 2),
        "problem": disease,
        "problem_confidence": round(disease_conf * 100, 2),
        "advice": advice
    })

# -------- NEARBY CLINICS ROUTE --------
@app.route("/nearby-clinics", methods=["GET"])
def nearby_clinics():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "Location required"}), 400

    lat = float(lat)
    lon = float(lon)

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": "veterinary clinic",
        "format": "json",
        "limit": 5,
        "viewbox": f"{lon-0.1},{lat+0.1},{lon+0.1},{lat-0.1}",
        "bounded": 1
    }

    headers = {
        "User-Agent": "PetDiseaseDetectionFYP/1.0"
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    clinics = []
    for place in data:
        clinic_lat = float(place.get("lat"))
        clinic_lon = float(place.get("lon"))

        distance = calculate_distance(lat, lon, clinic_lat, clinic_lon)
        map_link = f"https://www.google.com/maps?q={clinic_lat},{clinic_lon}"

        clinics.append({
            "name": place.get("display_name"),
            "distance_km": distance,
            "map_link": map_link
        })

    return jsonify({"clinics": clinics})

# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
