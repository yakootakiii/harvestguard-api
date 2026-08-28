import joblib
import numpy as np
import tensorflow as tf


MODEL_PATH = "models/harvestguard_baseline.keras"
SCALER_PATH = "models/harvestguard_scaler.pkl"


model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


LABELS = {
    0: "Safe",
    1: "Act Soon",
    2: "Critical",
}


def predict(
    temperature,
    humidity,
    gas_raw,
    temperature_rate,
    humidity_rate,
    gas_rate,
):
    features = np.array([[
        temperature,
        humidity,
        gas_raw,
        temperature_rate,
        humidity_rate,
        gas_rate,
    ]])

    # Apply the exact scaler used during training
    features_scaled = scaler.transform(features)

    probabilities = model.predict(
        features_scaled,
        verbose=0
    )[0]

    predicted_class = int(np.argmax(probabilities))

    return {
        "class_id": predicted_class,
        "classification": LABELS[predicted_class],
        "confidence": float(probabilities[predicted_class]),
        "probabilities": {
            "safe": float(probabilities[0]),
            "act_soon": float(probabilities[1]),
            "critical": float(probabilities[2]),
        },
    }