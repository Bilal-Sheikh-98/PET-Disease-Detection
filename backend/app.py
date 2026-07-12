from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import tensorflow as tf
import numpy as np
import cv2
import os
import requests
import math
from requests.exceptions import RequestException
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.efficientnet import decode_predictions

app = Flask(__name__)
CORS(app)


# ===============================
# CONFIG
# ===============================
UPLOAD_FOLDER = "uploads"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.50
ANIMAL_CONFIDENCE_THRESHOLD = 0.7

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# LOAD MODELS
# ===============================

# animal_model = tf.keras.models.load_model("models/animal_model.h5", compile=False)
animal_model = EfficientNetB0(weights="imagenet")
dog_disease_model = tf.keras.models.load_model("models/dog_disease_model.h5", compile=False)
cat_disease_model = tf.keras.models.load_model("models/cat_disease_model.h5", compile=False)

# ===============================
# CLASS NAMES
# ===============================

# ANIMAL_CLASSES = ["cat", "dog"]

DOG_CLASSES = {

    "golden_retriever",
    "labrador_retriever",
    "german_shepherd",
    "beagle",
    "boxer",
    "pug",
    "doberman",
    "rottweiler",
    "husky",
    "malamute",
    "chihuahua",
    "great_dane",
    "saint_bernard",
    "border_collie",
    "bull_mastiff",
    "cocker_spaniel",
    "english_setter",
    "french_bulldog",
    "basset",
    "bloodhound",
    "briard",
    "bull_terrier",
    "collie",
    "dalmatian",
    "dingo",
    "eskimo_dog",
    "newfoundland",
    "papillon",
    "pomeranian",
    "samoyed",
    "schipperke",
    "shih_tzu",
    "toy_poodle",
    "standard_poodle",
    "miniature_poodle",
    "vizsla",
    "weimaraner",
    "whippet",
    "yorkshire_terrier",
    "kuvasz",
    "borzoi",
    "bedlington_terrier",
    "affenpinscher",
"afghan_hound",
"airedale",
"akita",
"basenji",
"basset",
"beagle",
"bedlington_terrier",
"bernese_mountain_dog",
"black-and-tan_coonhound",
"bloodhound",
"bluetick",
"border_collie",
"border_terrier",
"borzoi",
"boston_bull",
"boxer",
"briard",
"bull_mastiff",
"bull_terrier",
"bulldog",
"cairn",
"cardigan",
"chesapeake_bay_retriever",
"chihuahua",
"chow",
"clumber",
"cocker_spaniel",
"collie",
"curly-coated_retriever",
"dalmatian",
"dhole",
"dingo",
"doberman",
"english_foxhound",
"english_setter",
"english_springer",
"eskimo_dog",
"flat-coated_retriever",
"french_bulldog",
"german_shepherd",
"golden_retriever",
"great_dane",
"great_pyrenees",
"groenendael",
"ibizan_hound",
"irish_setter",
"irish_terrier",
"irish_water_spaniel",
"irish_wolfhound",
"keeshond",
"kelpie",
"komondor",
"kuvasz",
"labrador_retriever",
"leonberg",
"malamute",
"miniature_pinscher",
"newfoundland",
"norfolk_terrier",
"norwegian_elkhound",
"papillon",
"pekinese",
"pembroke",
"pomeranian",
"pug",
"redbone",
"rottweiler",
"saint_bernard",
"samoyed",
"schipperke",
"scotch_terrier",
"scottish_deerhound",
"sealyham_terrier",
"shetland_sheepdog",
"shih_tzu",
"siberian_husky",
"staffordshire_bullterrier",
"standard_poodle",
"toy_poodle",
"miniature_poodle",
"vizsla",
"weimaraner",
"whippet",
"wire-haired_fox_terrier",

}

CAT_CLASSES = {

    "tabby",
    "tiger_cat",
    "persian_cat",
    "siamese_cat",
    "egyptian_cat",

}

DOG_DISEASE_CLASSES = [
    "ear_problem",
    "eye_problem",
    "paws_problem",
    "skin_problem"
]

CAT_DISEASE_CLASSES = [
    "ear_problem",
    "eye_problem",
    "paws_problem",
    "skin_problem"
]

# ===============================
# ADVICE MAPS
# ===============================
DOG_ADVICE = {
    "skin_problem": "Possible skin-related issue. Keep the area clean and consult a veterinarian if it worsens.",
    "ear_problem": "Possible ear-related issue. Avoid touching inside the ear and consult a veterinarian.",
    "eye_problem": "Possible eye-related issue. Clean gently and consult a veterinarian if symptoms persist.",
    "paws_problem": "Possible fungal infection in the paw area. Keep the paws clean and dry, avoid excessive moisture, and consult a veterinarian for proper diagnosis and treatment."

}

CAT_ADVICE = {
    "skin_problem": "Possible skin-related issue. Keep the area clean and dry. Consult a veterinarian.",
    "ear_problem": "Possible ear-related issue. Avoid touching inside the ear and consult a veterinarian.",
    "eye_problem": "Possible eye-related issue. Clean gently and consult a veterinarian.",
    "paws_problem": "Possible fungal infection in the paw area. Keep the paws clean and dry, avoid excessive moisture, and consult a veterinarian for proper diagnosis and treatment."

}

# ===============================
# IMAGE PREPROCESS
# ===============================
# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img


def preprocess_animal_image(image_path):

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224,224))

    img = np.expand_dims(img.astype(np.float32), axis=0)

    img = preprocess_input(img)

    return img

def preprocess_disease_image(image_path):

    img = cv2.imread(image_path)

    img = cv2.resize(img, (224,224))

    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)

    return img

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

    start_time = time.time()

    try:

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files["image"]

        image_path = os.path.join(UPLOAD_FOLDER, image.filename)

        image.save(image_path)


      # ===============================
        # PHASE 1 : ANIMAL DETECTION
        # ===============================

        animal_img = preprocess_animal_image(image_path)

        preds = animal_model.predict(animal_img, verbose=0)

        decoded = decode_predictions(preds, top=5)[0]

        print("\n========== IMAGENET ==========")
        print(decoded)
        print("==============================")

        animal = "unknown"
        animal_conf = 0

        for _, class_name, confidence in decoded:

            class_name = class_name.lower()


            if class_name in DOG_CLASSES:
                animal = "dog"
                animal_conf = float(confidence)
                break

            elif class_name in CAT_CLASSES:
                animal = "cat"
                animal_conf = float(confidence)
                break

        if animal == "unknown" or animal_conf <  0.10:

            response = {

                "status": "failed",

                "animal": "Unknown",

                "animal_confidence": round(animal_conf * 100, 2),

                "problem": "Unknown",

                "problem_confidence": 0,

                "prediction_time": round(time.time() - start_time, 2),

                "advice":
                "Please upload a clear image containing only a Cat or Dog."

            }

            print("\n==============================")

            print("UNKNOWN ANIMAL DETECTED")

            print(response)

            print("==============================\n")

            return jsonify(response)

        # ===============================
        # PHASE 2 : DISEASE DETECTION
        # ===============================

        img = preprocess_disease_image(image_path)


        if animal == "dog":

            disease_pred = dog_disease_model.predict(img, verbose=0)

            disease_index = np.argmax(disease_pred)

            disease = DOG_DISEASE_CLASSES[disease_index]

            disease_conf = float(disease_pred[0][disease_index])

            advice = DOG_ADVICE[disease]

        else:

            disease_pred = cat_disease_model.predict(img, verbose=0)

            disease_index = np.argmax(disease_pred)

            disease = CAT_DISEASE_CLASSES[disease_index]

            disease_conf = float(disease_pred[0][disease_index])

            advice = CAT_ADVICE[disease]

        if disease_conf < CONFIDENCE_THRESHOLD:

            disease = "uncertain"

            advice = "Low confidence prediction. Please consult a veterinarian."

        prediction_time = round(time.time() - start_time, 2)

        response = {

            "status": "success",

            "animal": animal,

            "animal_confidence": round(animal_conf * 100, 2),

            "problem": disease,

            "problem_confidence": round(disease_conf * 100, 2),

            "prediction_time": prediction_time,

            "advice": advice

        }

        print("\n========================================")

        print("PETGUARD AI PREDICTION")

        print("========================================")

        print(f"Animal              : {animal}")

        print(f"Animal Confidence   : {round(animal_conf*100,2)} %")

        print(f"Disease             : {disease}")

        print(f"Disease Confidence  : {round(disease_conf*100,2)} %")

        print(f"Prediction Time     : {prediction_time} sec")

        print("========================================\n")

        return jsonify(response)

    except Exception as e:

        print("\nPrediction Error:", e)

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500
    
    
# -------- NEARBY CLINICS ROUTE --------

# -------- NEARBY CLINICS ROUTE --------
@app.route("/nearby-clinics", methods=["GET"])
def nearby_clinics():

    try:

        lat = request.args.get("lat")
        lon = request.args.get("lon")

        if not lat or not lon:
            return jsonify({"error": "Location required"}), 400

        lat = float(lat)
        lon = float(lon)

        overpass_query = f"""
        [out:json];
        (
          node
            ["amenity"="veterinary"]
            (around:5000,{lat},{lon});
          way
            ["amenity"="veterinary"]
            (around:5000,{lat},{lon});
          relation
            ["amenity"="veterinary"]
            (around:5000,{lat},{lon});
        );
        out center tags;
        """

        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=overpass_query,
            headers={
                "User-Agent": "PetGuardAI/1.0"
            },
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        clinics = []
        added = set()

        for place in data["elements"]:

            if place["type"] == "node":
                clinic_lat = place["lat"]
                clinic_lon = place["lon"]

            else:

                center = place.get("center")

                if not center:
                    continue

                clinic_lat = center["lat"]
                clinic_lon = center["lon"]

            tags = place.get("tags", {})
            print("test==>",tags)

            name = tags.get("name", "Veterinary Clinic")
            phone = tags.get("phone") or tags.get("contact:phone") or "Not Available"
            if name in added:
                continue

            added.add(name)

            distance = calculate_distance(
                lat,
                lon,
                clinic_lat,
                clinic_lon
            )

            # address = ""

            # if "addr:street" in tags:
            #     address += tags["addr:street"]

            # if "addr:city" in tags:
            #     address += ", " + tags["addr:city"]
            address_parts = []

            if tags.get("addr:housenumber"):
                address_parts.append(tags["addr:housenumber"])

            if tags.get("addr:street"):
                address_parts.append(tags["addr:street"])

            if tags.get("addr:suburb"):
                address_parts.append(tags["addr:suburb"])

            if tags.get("addr:city"):
                address_parts.append(tags["addr:city"])

            if tags.get("addr:postcode"):
                address_parts.append(tags["addr:postcode"])

            address = ", ".join(address_parts)

            if not address:
                address = "Address Not Available"
            clinics.append({

                "name": name,

                "address": address,

                "latitude": clinic_lat,

                "longitude": clinic_lon,

                "distance_km": distance,

                "phone": phone,

                "address": address,

                "map_link":
                f"https://www.google.com/maps?q={clinic_lat},{clinic_lon}"

            })

        clinics.sort(key=lambda x: x["distance_km"])

        print("\n===============================")
        print("NEARBY VETERINARY CLINICS")
        print("===============================")

        for i, clinic in enumerate(clinics, start=1):

            print(
                f"{i}. {clinic['name']} "
                f"({clinic['distance_km']} km)"
            )

        print("===============================\n")

        return jsonify({

            "status": "success",

            "total": len(clinics),

            "clinics": clinics

        })

    except Exception as e:

        print("Clinic Search Error:", e)

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
