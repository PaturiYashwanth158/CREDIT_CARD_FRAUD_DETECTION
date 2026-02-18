import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load full pipeline model (includes scaler)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "ensemble_model.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return "Fraud Detection ML API is Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    features = np.array(data).reshape(1, -1)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return jsonify({
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
