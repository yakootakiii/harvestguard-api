from fastapi import FastAPI

from app.schemas import SensorInput
from app.model import predict
from app.recommendations import get_recommendation


app = FastAPI(
    title="HarvestGuard ML API",
    description="HarvestGuard sensor classification API",
    version="1.0.0"
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

    recommendation = get_recommendation(
        result["class_id"]
    )

    return {
        "classification": {
            "class_id": result["class_id"],
            "label": result["classification"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
        },

        "recommendation": recommendation
    }