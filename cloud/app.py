
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

PACKETS_PER_WINDOW = 10

MODEL_FEATURES = [
    "Header_Length",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "Min",
    "Max",
    "AVG",
    "Std",
    "IAT",
    "Variance",
]


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "cloud-ddos-detector"
    })


def extract_window_features(packets):
    """Aggregate packet metadata using the Edge detector's logic."""

    sizes = np.array(
        [float(p["packet_size"]) for p in packets],
        dtype=float
    )
    timestamps = np.array(
        [float(p["timestamp"]) for p in packets],
        dtype=float
    )

    if not np.all(np.isfinite(sizes)) or not np.all(
        np.isfinite(timestamps)
    ):
        raise ValueError("Packet sizes and timestamps must be finite")

    if np.any(sizes < 0):
        raise ValueError("Packet sizes cannot be negative")

    # Match Edge's aggregate_packets() duration calculation.
    duration = float(timestamps[-1] - timestamps[0])

    if duration <= 0:
        duration = 1e-9

    flags = [
        int(p.get("tcp_flags", 0))
        for p in packets
    ]

    fin = np.array([1 if f & 0x01 else 0 for f in flags])
    syn = np.array([1 if f & 0x02 else 0 for f in flags])
    rst = np.array([1 if f & 0x04 else 0 for f in flags])
    psh = np.array([1 if f & 0x08 else 0 for f in flags])
    ack = np.array([1 if f & 0x10 else 0 for f in flags])
    ece = np.array([1 if f & 0x40 else 0 for f in flags])
    cwr = np.array([1 if f & 0x80 else 0 for f in flags])

    ttls = [
        float(p.get("ttl", 0))
        for p in packets
        if float(p.get("ttl", 0)) > 0
    ]

    protocols = [
        int(p.get("protocol", 0))
        for p in packets
    ]
    values, counts = np.unique(protocols, return_counts=True)
    most_common_protocol = int(values[np.argmax(counts)])

    # These fields must be in exactly the trained model's order.
    features = {
        "Header_Length": float(np.mean([
            float(p.get("header_length", 0)) for p in packets
        ])),
        "Time_To_Live": float(np.mean(ttls)) if ttls else 0.0,
        "Rate": len(packets) / duration,
        "fin_flag_number": float(np.mean(fin)),
        "syn_flag_number": float(np.mean(syn)),
        "rst_flag_number": float(np.mean(rst)),
        "psh_flag_number": float(np.mean(psh)),
        "ack_flag_number": float(np.mean(ack)),
        "ece_flag_number": float(np.mean(ece)),
        "cwr_flag_number": float(np.mean(cwr)),
        "ack_count": int(np.sum(ack)),
        "syn_count": int(np.sum(syn)),
        "fin_count": int(np.sum(fin)),
        "rst_count": int(np.sum(rst)),
        "Min": float(np.min(sizes)),
        "Max": float(np.max(sizes)),
        "AVG": float(np.mean(sizes)),
        "Std": float(np.std(sizes, ddof=1)) if len(sizes) > 1 else 0.0,
        "IAT": duration / len(packets),
        "Variance": float(np.var(sizes, ddof=1)) if len(sizes) > 1 else 0.0,
    }

    return features, most_common_protocol


def make_prediction(X):
    predictions = model.predict(X).astype(int)

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    if 1 in classes:
        ddos_index = classes.index(1)
        ddos_probabilities = probabilities[:, ddos_index]
    else:
        ddos_probabilities = np.zeros(len(predictions))

    return predictions, ddos_probabilities


@app.post("/predict")
def predict():
    """Existing endpoint: accepts already-extracted model features."""
    start = time.perf_counter()

    data = request.get_json(silent=True) or {}
    features = data.get("features")

    if not isinstance(features, list) or not features:
        return jsonify({
            "error": "Provide a non-empty features list"
        }), 400

    try:
        X = np.asarray(features, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2 or X.shape[1] != model.n_features_in_:
            return jsonify({
                "error": "Incorrect feature count",
                "expected": int(model.n_features_in_),
                "received": int(X.shape[1]) if X.ndim == 2 else None
            }), 400

        inference_start = time.perf_counter()
        predictions, probabilities = make_prediction(X)
        inference_ms = (
            time.perf_counter() - inference_start
        ) * 1000

        return jsonify({
            "prediction": predictions.tolist(),
            "ddos_probability": probabilities.tolist(),
            "inference_time_ms": round(inference_ms, 3),
            "server_processing_time_ms": round(
                (time.perf_counter() - start) * 1000, 3
            )
        })

    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/predict_packets")
def predict_packets():
    """New endpoint: accepts one window of packet metadata."""
    start = time.perf_counter()
    data = request.get_json(silent=True) or {}
    packets = data.get("packets")

    if not isinstance(packets, list) or len(packets) != PACKETS_PER_WINDOW:
        return jsonify({
            "error": "Provide exactly 10 packet records",
            "required_packets": PACKETS_PER_WINDOW
        }), 400

    required_fields = {"timestamp", "packet_size"}

    try:
        for i, packet in enumerate(packets):
            if not isinstance(packet, dict):
                raise ValueError(f"Packet {i} must be an object")

            missing = required_fields - packet.keys()
            if missing:
                raise ValueError(
                    f"Packet {i} missing fields: {sorted(missing)}"
                )

        feature_row, protocol = extract_window_features(packets)

        X = np.array(
            [[feature_row[name] for name in MODEL_FEATURES]],
            dtype=float
        )

        inference_start = time.perf_counter()
        predictions, probabilities = make_prediction(X)
        inference_ms = (
            time.perf_counter() - inference_start
        ) * 1000

        prediction = int(predictions[0])

        return jsonify({
            "prediction": prediction,
            "label": "DDoS" if prediction == 1 else "BENIGN",
            "ddos_probability": float(probabilities[0]),
            "features": feature_row,
            "protocol_type": protocol,
            "packets_processed": len(packets),
            "inference_time_ms": round(inference_ms, 3),
            "server_processing_time_ms": round(
                (time.perf_counter() - start) * 1000, 3
            )
        })

    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
