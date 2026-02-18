
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/ensemble_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "FraudX Enterprise API Running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    features = np.array([data["features"]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return jsonify({
        "fraud": bool(prediction),
        "riskScore": round(probability * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
