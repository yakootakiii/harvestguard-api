"""Stream simulated sensor data through the running API, as a device would.

The unit tests score the model on individual windows. This asks a different
question: what does the *system* do when a cargo is monitored continuously for
hours? That surfaces behaviour no per-window test can -- detection latency,
alert flapping, false-alarm rate, and how the service responds to a firmware bug
or a sensor fault.

    .venv/bin/uvicorn app.main:app --port 8961 &
    .venv/bin/python tests/simulate_deployment.py --url http://localhost:8961

IMPORTANT: the spoilage trajectories below are invented, and calibrated against
the same noise floor the model was trained on. This measures how the system
*behaves*, not whether it detects real spoilage. Nothing here can establish the
latter -- that needs the recording described in model_train/collect_recording.md.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np

WINDOW = 60          # samples per window, 1 Hz -- must match training
EMIT_EVERY = 120     # seconds between API calls

SIGMA_NULL = 0.06823367407026865   # measured from Banana D1, see training_metadata
GAS_BASELINE = 103.17
GAS_NOISE = 2.72                   # sd of the real stable recording


@dataclass
class Scenario:
    name: str
    hours: float
    gas_rate_per_s: float = 0.0      # spoilage ramp
    onset_hours: float = 0.0         # when the ramp starts
    temperature: float = 28.0
    humidity: float = 70.0
    note: str = ""
    broken_rate: bool = False        # firmware sends consecutive delta, not slope
    stuck_gas: bool = False          # sensor flatlines
    expect: str = ""


SCENARIOS = [
    Scenario(
        "stable cargo", hours=8.0,
        note="nothing happening; measures the false-alarm rate",
        expect="Safe throughout",
    ),
    Scenario(
        "fast spoilage", hours=8.0, gas_rate_per_s=0.30, onset_hours=3.0,
        note="ramp at 4.4 sigma_null -- inside the Critical band",
        expect="escalates to Critical shortly after onset",
    ),
    Scenario(
        "realistic slow spoilage", hours=8.0, gas_rate_per_s=0.0014, onset_hours=2.0,
        note="+30 MQ3 units over 6 h, a physically plausible fruit ramp",
        expect="unknown -- this is the question the simulation answers",
    ),
    Scenario(
        "refrigerated transport", hours=4.0, temperature=4.0, humidity=85.0,
        note="a normal shipping condition, far outside the fitted domain",
        expect="flagged out-of-domain, not silently answered",
    ),
    Scenario(
        "firmware bug: consecutive delta", hours=6.0,
        gas_rate_per_s=0.30, onset_hours=2.0, broken_rate=True,
        note="device sends (last - previous) instead of the window slope",
        expect="spoilage missed -- the contract failure I cannot test on hardware",
    ),
    Scenario(
        "sensor flatline", hours=4.0, stuck_gas=True,
        note="MQ3 stops responding and reports a constant value",
        expect="unknown -- a stuck sensor looks identical to a calm one",
    ),
]


@dataclass
class Result:
    scenario: Scenario
    labels: list[str] = field(default_factory=list)
    times_h: list[float] = field(default_factory=list)
    out_of_domain: list[bool] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)


def simulate_stream(scenario: Scenario, rng: np.random.Generator) -> np.ndarray:
    """The full 1 Hz gas trace for a scenario."""
    n = int(scenario.hours * 3600)
    time = np.arange(n, dtype=float)

    if scenario.stuck_gas:
        return np.full(n, GAS_BASELINE)

    # Correlated sensor wander, matching the real recording's character, plus
    # white noise -- a stable sensor is not i.i.d. around its mean.
    wander = np.cumsum(rng.normal(0.0, 0.05, n))
    wander = wander - wander.mean()
    wander *= GAS_NOISE / max(wander.std(), 1e-9) * 0.7
    white = rng.normal(0.0, GAS_NOISE * 0.5, n)

    onset = int(scenario.onset_hours * 3600)
    ramp = np.clip(time - onset, 0.0, None) * scenario.gas_rate_per_s

    return GAS_BASELINE + wander + white + ramp


def device_rate(window: np.ndarray, broken: bool) -> float:
    """What the firmware reports as `_rate`."""
    if broken:
        # The failure mode: a difference between the last two samples.
        return float(window[-1] - window[-2])
    return float(np.polyfit(np.arange(len(window)), window, 1)[0])


def call_api(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def run_scenario(scenario: Scenario, url: str, rng: np.random.Generator) -> Result:
    gas = simulate_stream(scenario, rng)
    result = Result(scenario=scenario)

    for end in range(WINDOW, len(gas), EMIT_EVERY):
        window = gas[end - WINDOW:end]

        payload = {
            "temperature": scenario.temperature + float(rng.normal(0, 0.2)),
            "humidity": scenario.humidity + float(rng.normal(0, 2.0)),
            "gas_raw": float(window[-1]),
            "temperature_rate": 0.0,
            "humidity_rate": 0.0,
            "gas_rate": device_rate(window, scenario.broken_rate),
        }

        response = call_api(url, payload)
        result.labels.append(response["classification"]["label"])
        result.confidences.append(response["classification"]["confidence"])
        result.out_of_domain.append(not response["domain"]["in_domain"])
        result.times_h.append(end / 3600.0)

    return result


def summarise(result: Result) -> None:
    scenario = result.scenario
    labels = np.array(result.labels)
    times = np.array(result.times_h)

    print(f"\n{'=' * 78}")
    print(f"{scenario.name.upper()}  ({scenario.hours:g} h, {len(labels)} API calls)")
    print(f"  {scenario.note}")
    print(f"  expected: {scenario.expect}")
    print(f"{'-' * 78}")

    counts = {name: int((labels == name).sum()) for name in ["Safe", "Act Soon", "Critical"]}
    total = len(labels)
    print("  label distribution: " + "  ".join(
        f"{k} {v}/{total} ({100 * v / total:.0f}%)" for k, v in counts.items()
    ))

    ood = int(np.sum(result.out_of_domain))
    print(f"  out-of-domain:      {ood}/{total} ({100 * ood / total:.0f}%)")

    flips = int((labels[1:] != labels[:-1]).sum())
    print(f"  label changes:      {flips}  (alert flapping)")

    if scenario.onset_hours and scenario.gas_rate_per_s:
        for target in ["Act Soon", "Critical"]:
            hit = np.flatnonzero((labels == target) & (times >= scenario.onset_hours))
            if len(hit):
                lag = times[hit[0]] - scenario.onset_hours
                print(f"  first {target:9}:  {lag * 60:6.1f} min after onset")
            else:
                print(f"  first {target:9}:  NEVER")

    print("  timeline: " + "".join(
        {"Safe": ".", "Act Soon": "-", "Critical": "#"}[x] for x in labels
    ))
    print("            (. Safe   - Act Soon   # Critical)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8961")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    try:
        urllib.request.urlopen(f"{args.url}/health", timeout=10)
    except urllib.error.URLError as exc:
        print(f"No API at {args.url} ({exc}). Start uvicorn first.")
        return 1

    print(f"Streaming simulated cargo through {args.url}")
    print("Trajectories are invented; this measures system behaviour, not "
          "spoilage-detection skill.")

    for scenario in SCENARIOS:
        summarise(run_scenario(scenario, args.url, rng))

    print(f"\n{'=' * 78}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
