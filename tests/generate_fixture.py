"""Freeze the held-out evaluation set to a CSV fixture.

Run only when the data pipeline intentionally changes:

    .venv/bin/python tests/generate_fixture.py

The fixture is the *same* held-out set train.py scores itself against: the same
disjoint real-gas segments, the same seed, the same size. Freezing it to disk
means tests/test_model_accuracy.py can evaluate the shipped model without
importing the training pipeline at all -- so the test would still catch a change
in the generator that silently made the task easier.

Regenerating this file invalidates the thresholds in the test. If accuracy moves
after a regeneration, that is a real signal about the data, not a stale fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "model_train"))

from harvestguard_data import (  # noqa: E402
    FEATURE_NAMES,
    build_dataset,
    load_real_gas,
    split_start_indices,
)
import train as hg_train  # noqa: E402

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "holdout_windows.csv"


def main() -> int:
    signal = load_real_gas()
    _, test_starts = split_start_indices(signal)

    frame = build_dataset(
        signal,
        hg_train.SAMPLES_PER_CLASS_TEST,
        test_starts,
        seed=hg_train.SEED + 1,
    )

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FIXTURE_PATH, index=False, float_format="%.6f")

    print(f"wrote {FIXTURE_PATH} ({len(frame)} rows)")
    print(f"  columns: {FEATURE_NAMES + ['label']}")
    print(f"  class balance: {frame['label'].value_counts().sort_index().to_dict()}")
    print(f"  real gas segments: starts {test_starts.min()}..{test_starts.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
