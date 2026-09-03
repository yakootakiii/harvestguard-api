"""Model loading, input validation and inference.

Artifacts are produced by `model_train/train.py`, which also writes
`training_metadata.json` describing the domain the model was fitted on. That
metadata drives the range check below.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

# Paths are resolved relative to this file, not the process CWD, so the API
# starts correctly regardless of where uvicorn is launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

MODEL_PATH = MODELS_DIR / "harvestguard_baseline.keras"
SCALER_PATH = MODELS_DIR / "harvestguard_scaler.pkl"
METADATA_PATH = MODELS_DIR / "training_metadata.json"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(METADATA_PATH) as handle:
    METADATA = json.load(handle)

FEATURE_NAMES = METADATA["feature_names"]
DOMAIN_SIGMA = METADATA["domain_sigma"]
WINDOW_SIZE = METADATA["window_size_seconds"]

LABELS = {0: "Safe", 1: "Act Soon", 2: "Critical"}

_SCALER_MEAN = np.asarray(METADATA["scaler_mean"], dtype=float)
_SCALER_SCALE = np.asarray(METADATA["scaler_scale"], dtype=float)


def check_domain(features: np.ndarray) -> dict:
    """Flag inputs the model was never fitted on.

    StandardScaler does not clip and softmax always normalises, so an
    out-of-domain reading otherwise comes back as a confident class rather than
    an error. A refrigerated 4 C reading, for instance, sits about 16 sigma
    below the fitted temperature mean and previously returned "Act Soon" at
    1.000 confidence.
    """
    z_scores = (features - _SCALER_MEAN) / _SCALER_SCALE

    out_of_range = [
        {
            "feature": name,
            "value": float(features[i]),
            "z_score": round(float(z_scores[i]), 2),
            "trained_range": [
                round(float(_SCALER_MEAN[i] - DOMAIN_SIGMA * _SCALER_SCALE[i]), 3),
                round(float(_SCALER_MEAN[i] + DOMAIN_SIGMA * _SCALER_SCALE[i]), 3),
            ],
        }
        for i, name in enumerate(FEATURE_NAMES)
        if abs(z_scores[i]) > DOMAIN_SIGMA
    ]

    return {
        "in_domain": not out_of_range,
        "max_abs_z": round(float(np.max(np.abs(z_scores))), 2),
        "out_of_range": out_of_range,
    }


def predict(
    temperature,
    humidity,
    gas_raw,
    temperature_rate,
    humidity_rate,
    gas_rate,
):
    """Classify one sensor window.

    The three `_rate` arguments must be least-squares slopes per second over a
    `WINDOW_SIZE`-second window at 1 Hz -- the same definition used to build the
    training set. A consecutive-sample delta is not interchangeable.
    """
    features = np.array(
        [temperature, humidity, gas_raw, temperature_rate, humidity_rate, gas_rate],
        dtype=float,
    )

    domain = check_domain(features)

    features_scaled = scaler.transform(features.reshape(1, -1))
    probabilities = model.predict(features_scaled, verbose=0)[0]

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return {
        "class_id": predicted_class,
        "classification": LABELS[predicted_class],
        "confidence": confidence,
        "probabilities": {
            "safe": float(probabilities[0]),
            "act_soon": float(probabilities[1]),
            "critical": float(probabilities[2]),
        },
        "domain": domain,
    }
