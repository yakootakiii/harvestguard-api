"""
HarvestGuard training data pipeline.

Source of truth for how training windows are built and how the six API features
are computed. `train.py` and the notebook both import from here so the notebook
can never drift from the artifacts actually shipped in `models/`.

Design notes
------------
The only real measurement available is `Banana D1.csv`: a 4-minute, 10-channel
MQ gas log of a stable banana. It contains no spoilage event and no labels, so
classes must still be synthesised. What the real recording *can* provide -- and
what the original notebook did not use it for -- is the sensor's **noise floor**:
the distribution of window slopes observed when nothing is happening.

Class boundaries are therefore expressed in units of that measured noise floor
(`sigma_null`) rather than as invented absolute rates. This guarantees the
classes are statistically distinguishable from a stable sensor, which the
original ranges were not (all three sat within +/-1.25 sigma_null).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "Banana D1.csv"

# Feature order is the API contract. Must match app/model.py and app/schemas.py.
FEATURE_NAMES = [
    "temperature",
    "humidity",
    "gas_raw",
    "temperature_rate",
    "humidity_rate",
    "gas_rate",
]

LABEL_NAMES = {0: "Safe", 1: "Act Soon", 2: "Critical"}

# Samples per window at 1 Hz. 60 s rather than the original 30 s: the measured
# gas noise floor falls from 0.279 to 0.101 units/s between the two, which is
# what makes the gas channel usable at all.
WINDOW_SIZE = 60

# Class bands in units of sigma_null, contiguous and non-overlapping.
# The original ranges overlapped on [0.03, 0.05] and [0.12, 0.18] units/s.
GAS_BANDS_SIGMA = {
    0: (-1.0, 1.5),   # Safe      - within sensor noise
    1: (1.5, 3.5),    # Act Soon  - a rise the sensor can actually resolve
    2: (3.5, 7.0),    # Critical  - unambiguous
}

# MQ3 baseline level varies between sensors, sessions and ambient conditions.
# Without this jitter the generator pins every window to the same baseline, so
# the absolute `gas_raw` at the end of a window becomes a direct read-out of the
# trend and the model can shortcut past `gas_rate` entirely.
GAS_BASELINE_JITTER = 18.0

# SHT31 datasheet accuracy: +/-0.2 C, +/-2 %RH. No real environmental recording
# exists, so these are assumptions, not measurements -- flagged as such because
# the temperature and humidity channels remain fully synthetic.
TEMP_NOISE_SD = 0.2
HUMIDITY_NOISE_SD = 2.0

# Environmental drift is sampled INDEPENDENTLY of the label. Spoilage evidence
# is the gas trend; temperature and humidity are ambient context. Coupling them
# to the label (as the original generator did) let the network solve the task
# from two fully-synthetic channels and ignore the real-derived gas signal.
TEMP_DRIFT_RANGE = (-0.004, 0.010)      # deg C per second
HUMIDITY_DRIFT_RANGE = (-0.020, 0.040)  # %RH per second

# MQ-series sensors need their heater to stabilise before readings mean
# anything. In Banana D1 the first ~25 s ramp from 80 to ~103 at up to
# +1.10 units/s -- an order of magnitude above anything the fruit does. Left in,
# that transient inflates the measured noise floor by ~47% (0.1011 -> 0.0688)
# and injects a spurious upward drift into every window drawn from it.
WARMUP_SLOPE_WINDOW = 30
WARMUP_SLOPE_FACTOR = 3.0

TEMP_BASE_RANGE = (24.0, 32.0)
HUMIDITY_BASE_RANGE = (55.0, 85.0)


def ols_slope(values: np.ndarray) -> float:
    """Least-squares slope per sample.

    Uses every sample in the window. The original endpoint difference
    ((last - first) / n) used only two, which is why its estimate was swamped
    by sensor noise.
    """
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def extract_features(temperature, humidity, gas) -> list[float]:
    """Collapse three aligned 1 Hz windows into the six API features."""
    return [
        float(temperature[-1]),
        float(humidity[-1]),
        float(gas[-1]),
        ols_slope(temperature),
        ols_slope(humidity),
        ols_slope(gas),
    ]


def detect_warmup_end(values: np.ndarray, window: int = WARMUP_SLOPE_WINDOW,
                      factor: float = WARMUP_SLOPE_FACTOR) -> int:
    """Index of the first sample after the sensor heater has stabilised.

    Compares the leading rolling slope against the typical absolute slope in the
    back half of the recording; the warm-up ramp is whatever exceeds it by
    `factor`. Returns 0 when no transient is detected.
    """
    if len(values) <= 2 * window:
        return 0

    slopes = np.array(
        [ols_slope(values[i : i + window]) for i in range(len(values) - window)]
    )

    reference = np.median(np.abs(slopes[len(slopes) // 2 :]))
    threshold = factor * max(reference, 1e-9)

    settled = np.flatnonzero(np.abs(slopes) <= threshold)
    return int(settled[0]) if len(settled) else 0


@dataclass
class RealGasSignal:
    """The 1 Hz MQ3 trace plus the noise statistics derived from it."""

    series: pd.Series
    mean: float
    sd: float
    minimum: float
    maximum: float
    sigma_null: float          # sd of WINDOW_SIZE-sample OLS slopes, units/s
    duration_seconds: float
    raw_rows: int
    warmup_seconds: int

    @property
    def values(self) -> np.ndarray:
        return self.series.to_numpy(dtype=float)

    def summary(self) -> str:
        return (
            f"Banana D1: {self.raw_rows} raw rows over {self.duration_seconds:.1f} s\n"
            f"  dropped {self.warmup_seconds} s of sensor warm-up "
            f"-> {len(self.series)} usable samples at 1 Hz\n"
            f"  MQ3 mean={self.mean:.2f} sd={self.sd:.2f} "
            f"range=[{self.minimum:.0f}, {self.maximum:.0f}]\n"
            f"  noise floor at W={WINDOW_SIZE}s: sigma_null={self.sigma_null:.4f} units/s"
        )


def load_real_gas(csv_path: Path | str = CSV_PATH, channel: str = "MQ3") -> RealGasSignal:
    """Load the banana log, resample to 1 Hz medians, and measure the noise floor."""
    frame = pd.read_csv(csv_path)
    subset = frame[["Ticks", channel]].copy()
    subset["datetime"] = pd.to_datetime(subset["Ticks"], unit="ms")

    series = (
        subset.set_index("datetime")[channel]
        .resample("1s")
        .median()
        .dropna()
        .rename("gas_raw")
    )

    values = series.to_numpy(dtype=float)

    warmup = detect_warmup_end(values)
    if warmup:
        series = series.iloc[warmup:]
        values = series.to_numpy(dtype=float)

    if len(values) <= WINDOW_SIZE:
        raise ValueError(
            f"Recording has {len(values)} samples at 1 Hz, which is not enough "
            f"for a {WINDOW_SIZE}-sample window."
        )

    # Null distribution: slopes measured while nothing is happening.
    null_slopes = np.array(
        [ols_slope(values[i : i + WINDOW_SIZE]) for i in range(len(values) - WINDOW_SIZE)]
    )

    ticks = subset["Ticks"].to_numpy()
    return RealGasSignal(
        series=series,
        mean=float(values.mean()),
        sd=float(values.std(ddof=1)),
        minimum=float(values.min()),
        maximum=float(values.max()),
        sigma_null=float(null_slopes.std(ddof=1)),
        duration_seconds=float((ticks[-1] - ticks[0]) / 1000.0),
        raw_rows=int(len(frame)),
        warmup_seconds=int(warmup),
    )


def generate_window(
    label: int,
    signal: RealGasSignal,
    rng: np.random.Generator,
    start_index: int,
    window_size: int = WINDOW_SIZE,
):
    """Build one synthetic sensor window seeded from a real MQ3 segment.

    The gas channel is a real segment (mean-centred, so only its genuine noise
    texture is kept) with a class-dependent trend added. Temperature and
    humidity are fully synthetic and their drift is drawn independently of
    `label`.
    """
    time = np.arange(window_size, dtype=float)

    real_segment = signal.values[start_index : start_index + window_size].copy()
    baseline = signal.mean + rng.uniform(-GAS_BASELINE_JITTER, GAS_BASELINE_JITTER)
    real_segment = real_segment - real_segment.mean() + baseline

    low_sigma, high_sigma = GAS_BANDS_SIGMA[label]
    gas_trend = rng.uniform(low_sigma, high_sigma) * signal.sigma_null

    gas = real_segment + gas_trend * time
    # No clipping. The original np.clip() to the observed range silently
    # flattened exactly the trend the model was meant to learn.

    temperature = (
        rng.uniform(*TEMP_BASE_RANGE)
        + rng.uniform(*TEMP_DRIFT_RANGE) * time
        + rng.normal(0.0, TEMP_NOISE_SD, window_size)
    )
    humidity = (
        rng.uniform(*HUMIDITY_BASE_RANGE)
        + rng.uniform(*HUMIDITY_DRIFT_RANGE) * time
        + rng.normal(0.0, HUMIDITY_NOISE_SD, window_size)
    )

    return temperature, humidity, gas


def build_dataset(
    signal: RealGasSignal,
    samples_per_class: int,
    start_indices: np.ndarray,
    seed: int,
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """Generate a labelled feature table drawing only from `start_indices`."""
    rng = np.random.default_rng(seed)
    rows = []

    for label in sorted(GAS_BANDS_SIGMA):
        for _ in range(samples_per_class):
            start = int(rng.choice(start_indices))
            temperature, humidity, gas = generate_window(
                label, signal, rng, start, window_size
            )
            rows.append(extract_features(temperature, humidity, gas) + [label])

    return pd.DataFrame(rows, columns=FEATURE_NAMES + ["label"])


def split_start_indices(signal: RealGasSignal, test_fraction: float = 0.25,
                        window_size: int = WINDOW_SIZE):
    """Partition real-window start positions into disjoint train/test regions.

    Windows are 60 samples wide over a 241-sample recording, so neighbouring
    start positions produce near-duplicate gas segments. A random split would
    put near-copies on both sides and inflate the score; splitting by position
    (with a `window_size` guard band) keeps the test segments genuinely unseen.
    """
    n_starts = len(signal.values) - window_size
    cut = int(n_starts * (1.0 - test_fraction))

    train_starts = np.arange(0, cut - window_size)
    test_starts = np.arange(cut, n_starts)

    if len(train_starts) == 0 or len(test_starts) == 0:
        raise ValueError("Recording too short to split into disjoint regions.")

    return train_starts, test_starts
