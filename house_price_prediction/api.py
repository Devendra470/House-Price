"""
api.py
------
Flask REST API backend for House Price Prediction.
Exposes a /predict endpoint that the Streamlit app (or any client) can call.

Run with:  python api.py
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model once at startup
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
model      = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "House Price Prediction API",
        "version": "1.0",
        "endpoints": {
            "POST /predict": "Predict house price given area and rooms",
            "GET  /health":  "Health check",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Request JSON:
        { "area": <int>, "rooms": <int> }

    Response JSON:
        { "predicted_price": <float>, "area": <int>, "rooms": <int> }
    """
    data = request.get_json(force=True)

    # Validate input
    if "area" not in data or "rooms" not in data:
        return jsonify({"error": "Both 'area' and 'rooms' fields are required."}), 400

    try:
        area  = float(data["area"])
        rooms = float(data["rooms"])
    except (ValueError, TypeError):
        return jsonify({"error": "'area' and 'rooms' must be numeric values."}), 400

    if area <= 0 or rooms <= 0:
        return jsonify({"error": "'area' and 'rooms' must be positive numbers."}), 400

    # Predict
    features = pd.DataFrame({"area": [area], "rooms": [rooms]})
    price    = float(model.predict(features)[0])
    price    = max(price, 0)  # floor at 0

    return jsonify({
        "area":            int(area),
        "rooms":           int(rooms),
        "predicted_price": round(price, 2),
    })


if __name__ == "__main__":
    print("Starting House Price Prediction API on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
