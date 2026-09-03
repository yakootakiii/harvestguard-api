from fastapi import FastAPI

from app.schemas import SensorInput
from app.model import METADATA, WINDOW_SIZE, predict
from app.recommendations import get_recommendation


app = FastAPI(
    title="HarvestGuard ML API",
    description="HarvestGuard sensor classification API",
    version="2.0.0"
)


@app.get("/")
def root():
    return {
        "message": "HarvestGuard ML API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/model")
def model_info():
    """Training provenance and the limits of what this model can be trusted for."""
    return {
        "window_size_seconds": WINDOW_SIZE,
        "rate_definition": METADATA["rate_definition"],
        "feature_names": METADATA["feature_names"],
        "held_out_accuracy": METADATA["held_out_accuracy"],
        "theoretical_ceiling": METADATA["theoretical_ceiling"],
        "permutation_importance": METADATA["permutation_importance"],
        "real_signal": METADATA["real_signal"],
        "caveats": METADATA["caveats"],
    }


@app.post("/predict")
def prediction(data: SensorInput):

    result = predict(
        temperature=data.temperature,
        humidity=data.humidity,
        gas_raw=data.gas_raw,
        temperature_rate=data.temperature_rate,
        humidity_rate=data.humidity_rate,
        gas_rate=data.gas_rate,
    )

    domain = result["domain"]

    recommendation = get_recommendation(
        result["class_id"],
        in_domain=domain["in_domain"],
    )

    return {
        "classification": {
            "class_id": result["class_id"],
            "label": result["classification"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
        },

        "domain": domain,

        "recommendation": recommendation
    }
