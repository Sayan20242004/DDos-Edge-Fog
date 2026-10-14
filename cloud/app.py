
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import time

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    "ddos_random_forest_local_v4.joblib"
)

model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "cloud-ddos-detector"})

@app.post("/predict")
def predict():
    start = time.perf_counter()

    data = request.get_json(silent=True) or {}
    features = data.get("features")

    if not isinstance(features, list) or not features:
        return jsonify({"error": "Provide a non-empty features list"}), 400

    try:
        X = np.asarray(features, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != model.n_features_in_:
            return jsonify({
                "error": "Incorrect feature count",
                "expected": int(model.n_features_in_),
                "received": int(X.shape[1])
            }), 400

        inference_start = time.perf_counter()
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        inference_ms = (time.perf_counter() - inference_start) * 1000

        return jsonify({
            "prediction": predictions.astype(int).tolist(),
            "ddos_probability": probabilities[:, 1].tolist(),
            "inference_time_ms": round(inference_ms, 3),
            "server_processing_time_ms": round(
                (time.perf_counter() - start) * 1000, 3
            )
        })

    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
