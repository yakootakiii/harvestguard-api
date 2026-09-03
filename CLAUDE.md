# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Committed-in-place virtualenv at `.venv` (Python 3.11).

```bash
# Serve the API (path-independent, but the repo root is the conventional CWD)
.venv/bin/uvicorn app.main:app --reload

# Install
.venv/bin/pip install -r requirements.txt          # inference only
.venv/bin/pip install -r requirements-train.txt    # adds pandas/matplotlib/jupyter

# Retrain — rewrites everything in models/
.venv/bin/python model_train/train.py
.venv/bin/python model_train/train.py --csv "model_train/Other.csv" --channel MQ135

# Run the notebook end to end (must run from model_train/)
cd model_train && ../.venv/bin/python -m jupyter nbconvert \
    --to notebook --execute --inplace HarvestGuard_ML_v1.ipynb

# Tests (~4 s; needs requirements-train.txt)
.venv/bin/pytest tests/ -q
.venv/bin/pytest tests/test_model_accuracy.py::test_gas_rate_carries_the_signal -q

# Regenerate the test fixture — only when the data pipeline changes on purpose
.venv/bin/python tests/generate_fixture.py
```

No linter is configured. For ad-hoc API poking use `/docs` or
`fastapi.testclient.TestClient` (httpx is in `requirements-train.txt`).

## Tests

`tests/test_model_accuracy.py` scores the shipped artifacts against
`tests/fixtures/holdout_windows.csv` — a frozen copy of the same held-out set
`train.py` reports on. It imports only `app.model`, never the training pipeline, so a
generator change that made the task easier surfaces as a *rising* score against
unchanged data rather than being absorbed silently.

Thresholds come from measured spread across five training seeds (accuracy
0.7671 +/- 0.0076; recall 0.960/0.649/0.692), set about 4 sigma below the means. Two
guards are unusual and deliberate:

- **An upper bound on accuracy** (0.85). The ceiling for these class bands is 0.7789.
  Scoring well above it means the label leaked into a feature — the previous pipeline
  reported 0.9717 that way.
- **`gas_rate` must rank first** by permutation importance, dropping accuracy at least
  0.15 when shuffled. This is the direct regression guard for the bug where the model
  ignored the gas channel entirely.

Verified by mutation: swapping two feature columns fails 8 of 17; disabling the domain
guard fails 1; desyncing metadata from the model file fails 1.

### Simulation

`tests/simulate_deployment.py` streams a simulated cargo through a **running server**
over hours of virtual time, which surfaces behaviour per-window tests cannot: detection
latency, alert flapping, false-alarm rate, and response to firmware or sensor faults.

```bash
.venv/bin/uvicorn app.main:app --port 8961 &
.venv/bin/python tests/simulate_deployment.py --url http://localhost:8961
```

`tests/test_simulation.py` is the CI-sized version — shorter scenarios, in-process,
asserting only what must not regress. Its trajectories are invented and calibrated to
the same noise floor the model trained on, so it measures **system behaviour, not
spoilage-detection skill**.

Two findings from it are recorded as known gaps rather than fixed:

- **A flatlined sensor reads as healthy cargo.** A stuck MQ3 produces a zero-slope
  window, which is a valid Safe window. Held as a `strict=True` xfail so it flips to a
  failure the moment a liveness check (window variance) is added.
- **Physically realistic spoilage is invisible to `gas_rate`.** A +30-unit rise over 6 h
  is 0.02 sigma_null, deep inside the Safe band. What escalation does occur comes from
  `gas_raw` crossing an absolute level, not from the trend — with heavy flapping. The
  60-second window can only resolve implausibly fast events; a window matched to real
  spoilage kinetics is part of what the new recording enables.

## Architecture

FastAPI wrapper around a Keras classifier that turns six sensor features into a 3-class
risk assessment plus a recommendation.

```
SensorInput (app/schemas.py)
  -> predict()            (app/model.py)          scale -> forward pass -> class_id
     +  check_domain()    (app/model.py)          reject out-of-distribution inputs
  -> get_recommendation() (app/recommendations.py)
  -> {"classification": {...}, "domain": {...}, "recommendation": {...}}
```

`model_train/` is the other half: it produces the three files in `models/` that the API
loads at import. `harvestguard_data.py` owns data generation, `train.py` owns fitting,
and the notebook imports both so it cannot drift from the shipped artifacts.

### `class_id` is the contract between modules

The integer class id (0 = Safe, 1 = Act Soon, 2 = Critical) couples four places that must
stay aligned: `LABELS` in `app/model.py`, `RECOMMENDATIONS` in `app/recommendations.py`,
and `LABEL_NAMES` / `GAS_BANDS_SIGMA` in `model_train/harvestguard_data.py`. The
`probabilities` dict is built **positionally** off the output vector. Changing the class
set means retraining and updating all four.

### Feature ordering is load-bearing

`FEATURE_NAMES` in `harvestguard_data.py` is the canonical order:

```
temperature, humidity, gas_raw, temperature_rate, humidity_rate, gas_rate
```

`app/model.py` builds its array in that order and `training_metadata.json` records it.
`SensorInput` field order mirrors it for readability, but Pydantic does not enforce it.
Reordering without re-fitting the scaler produces plausible-looking, wrong predictions.

### The `_rate` contract

The API is stateless and trusts the caller's `_rate` values. They must be **least-squares
slopes per second over a 60-sample window at 1 Hz** — `np.polyfit(range(60), values, 1)[0]`
— matching `harvestguard_data.extract_features`. A consecutive-sample delta is roughly 8x
noisier and lands outside the fitted domain.

(v1 used `(last - first) / n` over 30 samples, via a latent bug: `len((temperature)-1)`
is elementwise subtraction then `len()`, i.e. 30, not the intended 29. It only worked
because the inputs were NumPy arrays.)

### Domain guard

`check_domain()` flags any feature more than `domain_sigma` (4.0) from the training mean.
`StandardScaler` does not clip and softmax always normalises, so without this an
out-of-domain reading returns a confident class rather than an error — a 4 °C
refrigerated reading sits ~10σ out and previously came back as `Act Soon` at 1.000.
When `domain.in_domain` is false the API still reports the class but swaps in the
`OUT_OF_DOMAIN` recommendation. Do not treat `confidence` as calibrated.

## What the model is trained on — read before trusting a prediction

Training data is synthetic, seeded from one real recording. The details matter because
several non-obvious choices exist specifically to stop the model cheating:

- **Only the gas channel is real.** `Banana D1.csv` is a 4-minute, 10-channel MQ log of a
  *stable* banana: no spoilage event, no labels. Gas windows are real segments,
  mean-centred so only sensor texture survives, with a synthetic trend added.
- **Temperature and humidity are entirely fabricated**, and their drift is drawn
  **independently of the label**. This is deliberate. When v1 keyed them to the label, the
  network solved the task from two clean synthetic channels and ignored gas completely
  (`gas_rate` = 5.0, fourteen times its own Critical maximum, still returned Safe at
  1.000). They are retained only to keep the API contract stable; the model ignores them,
  and `permutation_importance` in the metadata confirms it each run.
- **Class bands are defined in units of the measured noise floor** (`sigma_null`, the sd of
  window slopes in the stable recording), not as absolute rates. v1's invented bands all
  sat within ±1.3σ of the sensor doing nothing, and overlapped each other on
  `[0.03, 0.05]` and `[0.12, 0.18]`.
- **Sensor warm-up is detected and trimmed.** The first ~15 s is MQ heater stabilisation
  ramping at +1.10 units/s. Leaving it in inflates the noise floor by 47%
  (0.068 → 0.101) and injects spurious upward drift.
- **No `np.clip()`.** v1 clipped gas to the observed range, flattening the trend it was
  supposed to learn.
- **The train/test split is by window position, not random.** Windows are 60 samples wide
  over 226, so neighbouring starts are near-duplicates and a random split leaks.

### Interpreting the accuracy

Held-out accuracy is **0.7700 against a theoretical ceiling of 0.7789** — 98.9% of what is
achievable. Bands are contiguous and a one-window slope carries ~1 σ_null of noise by
definition, so boundary windows are irreducibly ambiguous. `train.py` computes and records
the ceiling for exactly this reason: v1's 0.9717 was higher only because its classes were
separated in clean fabricated channels, and the number meant nothing.

Critical recall (~0.69) is depressed by regional drift — 226 usable seconds is too short
for train and test segments to share drift statistics. More recording time is the only fix.

### Model size is a hard constraint: this runs on an STM32

The target is on-device inference via TFLite Micro, not a server. `build_model` is
275 parameters (`Dense(16) -> Dense(8) -> Dense(3)`, no dropout) and that number was
measured, not guessed — across three seeds, 16-8 scored 0.7721 while an 803-parameter
32-16 network with dropout scored 0.7551 and a 2627-parameter 64-32 one scored 0.7591.
The task is close to a 1-D decision on `gas_rate`, so extra capacity only adds variance.

Converted footprint (measured, `tests/fixtures/holdout_windows.csv` as the
representative dataset):

| build | flash | arena | accuracy |
|---|---|---|---|
| float32 | 3,288 B | 1,244 B | 0.7700 |
| int8 | 3,512 B | 392 B | 0.7587 |

int8 costs 1.1 points and cuts working RAM 3x. Ops are `FULLY_CONNECTED` / `RELU` /
`SOFTMAX` only, all standard TFLite Micro. Do not add layer types or grow this network
without re-checking both numbers. The arena figures are computed from tensor sizes, not
measured with `tflite::MicroInterpreter` — a real arena runs somewhat larger.

On-device, the scaler is 12 float constants (`scaler_mean` / `scaler_scale` in the
metadata) and the rate is a single-pass slope: `sum((i - 29.5) * w[i]) / 17995` over the
60-sample window, which matches `np.polyfit` to 3e-15 and is float32-safe.

### The limitation that modelling cannot fix

No real spoilage event has ever been recorded, so the absolute gas rate corresponding to
real spoilage is unvalidated. `model_train/collect_recording.md` specifies the recording
that would close this, and names the parts of `harvestguard_data.py` that get deleted once
it exists (the synthetic generator is scaffolding, not the destination).

## Notes

- `models/training_metadata.json` is generated, not hand-edited. It carries the feature
  order, scaler statistics, domain width, measured noise floor, permutation importances
  and caveats; `app/model.py` and `GET /model` both read from it.
- `scikit-learn` must stay pinned to the version that fitted the scaler or unpickling
  warns and the transform is not guaranteed to match.
- `postman/` and `.postman/` exist locally but are gitignored.
