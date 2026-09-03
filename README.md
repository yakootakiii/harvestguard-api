# harvestguard-api

FastAPI service that classifies food-storage sensor windows as **Safe**, **Act Soon** or
**Critical**, with a recommendation and an out-of-domain guard.

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
```

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "temperature": 28.0, "humidity": 70.0, "gas_raw": 118.0,
  "temperature_rate": 0.0, "humidity_rate": 0.0, "gas_rate": 0.35
}'
```

`_rate` fields are least-squares slopes per second over a 60-second window at 1 Hz.
`GET /model` reports training provenance, measured accuracy and known caveats.

## Status

The classifier is trained on synthetic windows seeded from a single 4-minute gas
recording of a stable banana. Class boundaries are calibrated against that sensor's
measured noise floor, but **no real spoilage event has been recorded yet**, so the
absolute rates that indicate real spoilage are unvalidated. See
`model_train/collect_recording.md`.

Retrain with `.venv/bin/python model_train/train.py`.
