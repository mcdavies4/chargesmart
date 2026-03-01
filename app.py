from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import json
import pgeocode
from geopy.distance import geodesic

app = Flask(__name__)

import os

# Train model if it doesn't exist
if not os.path.exists("charger_model.pkl"):
    print("No model found - training now...")
    import subprocess
    subprocess.run(["python", "model.py"])

# Load the trained AI model
with open("charger_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load all chargers from latest data snapshot
def load_chargers():
    chargers = []
    try:
        with open("enriched_data.csv", "r") as f:
            lines = f.readlines()
            seen = set()
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 8:
                    lat = parts[6]
                    lon = parts[7]
                    key = f"{lat},{lon}"
                    if key not in seen:
                        seen.add(key)
                        chargers.append({
                            "lat": float(lat),
                            "lon": float(lon),
                            "operator": parts[8] if len(parts) > 8 else "unknown",
                            "capacity": 1.0,
                            "location_type": parts[4]
                        })
    except Exception as e:
        print(f"Error loading chargers: {e}")
    return chargers

ALL_CHARGERS = load_chargers()
print(f"Loaded {len(ALL_CHARGERS)} unique chargers")

def get_nearby_chargers(lat, lon, radius_miles):
    nearby = []
    for c in ALL_CHARGERS:
        dist = geodesic((lat, lon), (c["lat"], c["lon"])).miles
        if dist <= radius_miles:
            c_copy = c.copy()
            c_copy["distance_miles"] = round(dist, 2)
            nearby.append(c_copy)
    nearby.sort(key=lambda x: x["distance_miles"])
    return nearby[:50]

def predict_charger(charger, hour, day_of_week):
    is_weekend = 1 if day_of_week >= 5 else 0
    try:
        capacity = float(charger["capacity"])
    except:
        capacity = 1.0
    features = [[hour, day_of_week, is_weekend, capacity,
                 charger["lat"], charger["lon"]]]
    probability = model.predict_proba(features)[0]
    prediction = model.predict(features)[0]
    return {
        "prediction": "busy" if prediction == 1 else "free",
        "probability_free": round(probability[0] * 100, 1),
        "probability_busy": round(probability[1] * 100, 1)
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    hour = int(request.args.get("hour", 12))
    day_of_week = int(request.args.get("day_of_week", 0))
    capacity = int(request.args.get("capacity", 2))
    lat = float(request.args.get("lat", 51.5))
    lon = float(request.args.get("lon", -0.1))
    is_weekend = 1 if day_of_week >= 5 else 0
    features = [[hour, day_of_week, is_weekend, capacity, lat, lon]]
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    return jsonify({
        "prediction": "busy" if prediction == 1 else "free",
        "probability_free": round(probability[0] * 100, 1),
        "probability_busy": round(probability[1] * 100, 1),
        "hour": hour,
        "day_of_week": day_of_week
    })

@app.route("/nearby")
def nearby():
    hour = int(request.args.get("hour", 12))
    day_of_week = int(request.args.get("day_of_week", 0))
    radius = float(request.args.get("radius", 1.0))
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    postcode = request.args.get("postcode")

    if postcode:
        nomi = pgeocode.Nominatim("GB")
        result = nomi.query_postal_code(postcode.strip().upper())
        if result is None or result.latitude != result.latitude:
            return jsonify({"error": "Postcode not found"}), 400
        lat = float(result.latitude)
        lon = float(result.longitude)
    else:
        lat = float(lat)
        lon = float(lon)

    chargers = get_nearby_chargers(lat, lon, radius)

    results = []
    for c in chargers:
        pred = predict_charger(c, hour, day_of_week)
        results.append({
            "lat": c["lat"],
            "lon": c["lon"],
            "operator": c["operator"],
            "capacity": c["capacity"],
            "distance_miles": c["distance_miles"],
            "prediction": pred["prediction"],
            "probability_free": pred["probability_free"],
            "probability_busy": pred["probability_busy"]
        })

    return jsonify({
        "total": len(results),
        "lat": lat,
        "lon": lon,
        "chargers": results
    })

if __name__ == "__main__":
    app.run(debug=True)