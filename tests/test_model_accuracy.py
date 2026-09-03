"""Accuracy of the shipped model against a frozen sample of held-out windows.

Deliberately imports only `app.model` -- never the training pipeline. The
fixture is a static CSV, so a change to the generator that made the task easier
would show up here as a *rising* score against unchanged data rather than being
silently absorbed.

Thresholds come from measured run-to-run spread across five training seeds, set
roughly 4 sigma below the observed means:

    accuracy          0.7469 +/- 0.0114   (range 0.7330 - 0.7610)
    recall Safe       0.945  +/- 0.015
    recall Act Soon   0.676  +/- 0.040
    recall Critical   0.620  +/- 0.007

Regenerate the fixture with `python tests/generate_fixture.py` only when the
data pipeline changes on purpose.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from app.model import (
    FEATURE_NAMES,
    LABELS,
    METADATA,
    check_domain,
    model,
    predict,
    scaler,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "holdout_windows.csv"

# --- floors (a regression drops below) -------------------------------------
MIN_ACCURACY = 0.70
MIN_RECALL = {0: 0.85, 1: 0.50, 2: 0.55}

# --- ceiling (leakage pushes above) ----------------------------------------
# The theoretical ceiling for these class bands is 0.7789: bands are contiguous
# and a one-window slope carries ~1 sigma_null of noise, so boundary windows are
# irreducibly ambiguous. Scoring materially above that means labels have leaked
# into the features -- which is exactly what the previous pipeline did, reporting
# 0.9717 while the model ignored the gas channel entirely.
MAX_PLAUSIBLE_ACCURACY = 0.85

# `gas_rate` must remain the feature carrying the signal. The regression this
# guards against is real: the earlier model scored 0.97 while returning "Safe" at
# 1.000 confidence for a gas_rate of 5.0, fourteen times its own Critical maximum.
MIN_GAS_RATE_IMPORTANCE = 0.15


@pytest.fixture(scope="module")
def holdout():
    """Frozen held-out windows: (features, labels)."""
    with open(FIXTURE_PATH, newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows, f"fixture is empty: {FIXTURE_PATH}"

    features = np.array(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=float
    )
    labels = np.array([int(row["label"]) for row in rows], dtype=int)
    return features, labels


@pytest.fixture(scope="module")
def predictions(holdout):
    """Batch-scored class ids. Batched because `predict` is one row per call."""
    features, _ = holdout
    return np.argmax(model.predict(scaler.transform(features), verbose=0), axis=1)


def test_fixture_is_balanced_and_well_formed(holdout):
    features, labels = holdout

    assert features.shape[1] == len(FEATURE_NAMES)
    assert np.isfinite(features).all()

    counts = np.bincount(labels, minlength=len(LABELS))
    assert set(np.unique(labels)) == set(LABELS)
    assert counts.min() == counts.max(), f"unbalanced fixture: {counts}"


def test_accuracy_above_floor(holdout, predictions):
    _, labels = holdout
    accuracy = float((predictions == labels).mean())

    assert accuracy >= MIN_ACCURACY, (
        f"accuracy {accuracy:.4f} below floor {MIN_ACCURACY}. Either the model "
        f"regressed or the artifacts in models/ are stale -- rerun "
        f"`python model_train/train.py`."
    )


def test_accuracy_not_implausibly_high(holdout, predictions):
    """A high score here means label leakage, not a better model."""
    _, labels = holdout
    accuracy = float((predictions == labels).mean())

    assert accuracy <= MAX_PLAUSIBLE_ACCURACY, (
        f"accuracy {accuracy:.4f} exceeds {MAX_PLAUSIBLE_ACCURACY}, above the "
        f"{METADATA['theoretical_ceiling']:.4f} ceiling these class bands allow. "
        f"Check that the label has not leaked into a feature."
    )


def test_accuracy_matches_recorded_metadata(holdout, predictions):
    """Guards against a model file swapped in without regenerating metadata."""
    _, labels = holdout
    accuracy = float((predictions == labels).mean())
    recorded = METADATA["held_out_accuracy"]

    assert accuracy == pytest.approx(recorded, abs=0.005), (
        f"fixture accuracy {accuracy:.4f} disagrees with the {recorded:.4f} "
        f"recorded in training_metadata.json; artifacts and metadata are out of sync."
    )


def test_every_class_is_predicted(predictions):
    """A collapsed model can still clear an accuracy floor on balanced data."""
    predicted = set(np.unique(predictions).tolist())

    assert predicted == set(LABELS), (
        f"model never predicts {sorted(set(LABELS) - predicted)}"
    )


@pytest.mark.parametrize("class_id", sorted(LABELS))
def test_per_class_recall(holdout, predictions, class_id):
    _, labels = holdout
    mask = labels == class_id
    recall = float((predictions[mask] == class_id).mean())

    assert recall >= MIN_RECALL[class_id], (
        f"{LABELS[class_id]} recall {recall:.3f} below floor "
        f"{MIN_RECALL[class_id]}"
    )


def test_critical_is_never_called_safe(holdout, predictions):
    """The costly confusion: spoiled cargo reported as fine."""
    _, labels = holdout
    critical_called_safe = int(((labels == 2) & (predictions == 0)).sum())

    assert critical_called_safe == 0, (
        f"{critical_called_safe} Critical windows classified Safe"
    )


def test_gas_rate_carries_the_signal(holdout, predictions):
    """Shuffle each feature; the accuracy drop is its contribution."""
    features, labels = holdout
    baseline = float((predictions == labels).mean())
    rng = np.random.default_rng(0)

    drops = {}
    for index, name in enumerate(FEATURE_NAMES):
        shuffled = features.copy()
        rng.shuffle(shuffled[:, index])
        scored = np.argmax(model.predict(scaler.transform(shuffled), verbose=0), axis=1)
        drops[name] = baseline - float((scored == labels).mean())

    ranked = sorted(drops, key=drops.get, reverse=True)

    assert ranked[0] == "gas_rate", (
        f"gas_rate is not the dominant feature; ranking was {ranked}. The model "
        f"is solving the task from something other than the gas signal."
    )
    assert drops["gas_rate"] >= MIN_GAS_RATE_IMPORTANCE, (
        f"shuffling gas_rate costs only {drops['gas_rate']:.3f} accuracy"
    )


def test_public_predict_matches_batch_scoring(holdout, predictions):
    """The API path and the batch path must agree."""
    features, _ = holdout
    rng = np.random.default_rng(1)
    sample = rng.choice(len(features), size=25, replace=False)

    for index in sample:
        result = predict(**dict(zip(FEATURE_NAMES, features[index])))

        assert result["class_id"] == int(predictions[index])
        assert result["classification"] == LABELS[result["class_id"]]
        assert result["confidence"] == pytest.approx(
            max(result["probabilities"].values())
        )
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)


def test_holdout_windows_are_in_domain(holdout):
    """Sanity check on the guard: real training-distribution data must pass."""
    features, _ = holdout
    flagged = sum(1 for row in features if not check_domain(row)["in_domain"])

    assert flagged / len(features) < 0.02, (
        f"{flagged}/{len(features)} in-distribution windows flagged out-of-domain; "
        f"the guard is too tight."
    )


def test_out_of_domain_reading_is_flagged():
    """A 4 C refrigerated reading previously returned Act Soon at 1.000."""
    result = predict(
        temperature=4.0,
        humidity=70.0,
        gas_raw=103.0,
        temperature_rate=0.0,
        humidity_rate=0.0,
        gas_rate=0.02,
    )

    assert result["domain"]["in_domain"] is False
    assert "temperature" in {
        entry["feature"] for entry in result["domain"]["out_of_range"]
    }
