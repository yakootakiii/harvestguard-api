"""Train the HarvestGuard classifier and write the artifacts the API loads.

Run from the repository root:

    .venv/bin/python model_train/train.py

Writes models/harvestguard_baseline.keras, models/harvestguard_scaler.pkl and
models/training_metadata.json. The metadata file records the feature order, the
measured noise floor and the accepted input domain; app/model.py reads it at
startup to range-check incoming requests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harvestguard_data import (  # noqa: E402
    CSV_PATH,
    FEATURE_NAMES,
    GAS_BANDS_SIGMA,
    LABEL_NAMES,
    WINDOW_SIZE,
    build_dataset,
    load_real_gas,
    split_start_indices,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

SAMPLES_PER_CLASS_TRAIN = 4000
SAMPLES_PER_CLASS_TEST = 1000
SEED = 42

# Domain guard width. Requests further than this many standard deviations from
# the training mean on any feature are flagged out-of-domain by the API rather
# than answered with a confident label.
DOMAIN_SIGMA = 4.0


def build_model(n_features: int, n_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def theoretical_ceiling(rng, n=200_000, noise_sigma=1.0):
    """Best achievable accuracy given band overlap and slope-estimation noise.

    Class bands are contiguous in sigma_null, and a slope measured over one
    window carries ~1 sigma_null of noise by definition. So even a perfect
    classifier misreads windows near a boundary. Reporting this alongside the
    held-out score keeps the accuracy number interpretable: the original
    pipeline's 0.97 looked better only because its classes were separated in
    fully-synthetic, low-noise channels.
    """
    labels = rng.integers(0, len(GAS_BANDS_SIGMA), n)
    true = np.array([rng.uniform(*GAS_BANDS_SIGMA[int(l)]) for l in labels])
    edges = [GAS_BANDS_SIGMA[k][1] for k in sorted(GAS_BANDS_SIGMA)[:-1]]
    predicted = np.digitize(true + rng.normal(0.0, noise_sigma, n), edges)
    return float((predicted == labels).mean())


def permutation_importance(model, scaler, X, y, rng, repeats=5):
    """Drop in accuracy when each feature is shuffled.

    This is the check the original pipeline lacked: it shows directly whether
    the gas channel is doing any work, rather than assuming it is.
    """
    baseline = float((np.argmax(model.predict(scaler.transform(X), verbose=0), axis=1) == y).mean())
    out = {}
    for j, name in enumerate(FEATURE_NAMES):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            acc = float((np.argmax(model.predict(scaler.transform(Xp), verbose=0), axis=1) == y).mean())
            drops.append(baseline - acc)
        out[name] = float(np.mean(drops))
    return baseline, out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="Sensor log to draw real gas segments from. Point this at a new "
             "recording (see collect_recording.md) to retrain on better data.",
    )
    parser.add_argument(
        "--channel",
        default="MQ3",
        help="Gas column to use. MQ135 (ammonia) and MQ4 (methane) are likely "
             "better spoilage proxies than MQ3 (alcohol) and are already logged.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    signal = load_real_gas(args.csv, channel=args.channel)
    print(signal.summary())

    train_starts, test_starts = split_start_indices(signal)
    print(
        f"\nLeakage-safe split of real gas segments:\n"
        f"  train starts {train_starts.min()}..{train_starts.max()} (n={len(train_starts)})\n"
        f"  test  starts {test_starts.min()}..{test_starts.max()} (n={len(test_starts)})"
    )

    train_df = build_dataset(signal, SAMPLES_PER_CLASS_TRAIN, train_starts, seed=SEED)
    test_df = build_dataset(signal, SAMPLES_PER_CLASS_TEST, test_starts, seed=SEED + 1)

    # Shuffle before fitting. build_dataset emits all of class 0, then 1, then 2,
    # and Keras's validation_split takes the LAST fraction of the array without
    # shuffling -- on ordered data that makes the validation set 100% Critical.
    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    X_train = train_df[FEATURE_NAMES].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    X_test = test_df[FEATURE_NAMES].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy()
    print(f"\ntrain {X_train.shape}  test {X_test.shape}")

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tf.keras.utils.set_random_seed(SEED)
    model = build_model(len(FEATURE_NAMES), len(LABEL_NAMES))
    model.summary()

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=64,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            )
        ],
    )

    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)

    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    print(f"\nHeld-out loss {test_loss:.4f}  accuracy {test_accuracy:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion matrix (rows = true):")
    print(confusion_matrix(y_test, y_pred))

    rng = np.random.default_rng(SEED)
    ceiling = theoretical_ceiling(rng)
    print(
        f"\nTheoretical ceiling for these bands at 1.0 sigma_null slope noise: "
        f"{ceiling:.4f}\n  held-out {test_accuracy:.4f} is "
        f"{100 * test_accuracy / ceiling:.1f}% of achievable"
    )

    baseline_acc, importance = permutation_importance(model, scaler, X_test, y_test, rng)
    print(f"\nPermutation importance (accuracy drop when shuffled), baseline {baseline_acc:.4f}:")
    for name, drop in sorted(importance.items(), key=lambda kv: -kv[1]):
        print(f"  {name:18} {drop:+.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "harvestguard_baseline.keras")
    joblib.dump(scaler, MODELS_DIR / "harvestguard_scaler.pkl")

    metadata = {
        "feature_names": FEATURE_NAMES,
        "labels": {str(k): v for k, v in LABEL_NAMES.items()},
        "window_size_seconds": WINDOW_SIZE,
        "rate_definition": "least-squares slope per second over the window",
        "gas_bands_sigma_null": {str(k): list(v) for k, v in GAS_BANDS_SIGMA.items()},
        "real_signal": {
            "source": f"{args.csv.name} ({args.channel})",
            "duration_seconds": signal.duration_seconds,
            "samples_1hz": len(signal.series),
            "sigma_null": signal.sigma_null,
            "mean": signal.mean,
            "sd": signal.sd,
        },
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "domain_sigma": DOMAIN_SIGMA,
        "training_feature_min": X_train.min(axis=0).tolist(),
        "training_feature_max": X_train.max(axis=0).tolist(),
        "held_out_accuracy": float(test_accuracy),
        "held_out_loss": float(test_loss),
        "theoretical_ceiling": ceiling,
        "permutation_importance": importance,
        "epochs_trained": len(history.history["loss"]),
        "sklearn_version": __import__("sklearn").__version__,
        "tensorflow_version": tf.__version__,
        "caveats": [
            "Temperature and humidity channels are fully synthetic; no real "
            "environmental sensor recording exists yet.",
            "No real spoilage event has ever been recorded. Class separability "
            "is calibrated against the measured sensor noise floor, but the "
            "absolute gas rates that correspond to real spoilage are unvalidated.",
            "Held-out accuracy is depressed by regional drift: the 226 usable "
            "seconds of real gas are too short for the train and test segments "
            "to share the same wander statistics. More recording time is the fix.",
        ],
    }
    (MODELS_DIR / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nWrote artifacts to {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
