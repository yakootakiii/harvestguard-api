"""Behavioural invariants from streaming a simulated cargo through the model.

The full harness (tests/simulate_deployment.py) runs against a live server and
prints a report. This is the CI-sized version: shorter scenarios, in-process, no
HTTP, asserting only the behaviours that must not regress.

As with the harness, the trajectories are invented. These tests check how the
system *behaves* over time, not whether it detects real spoilage.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.model import predict
from tests.simulate_deployment import (
    EMIT_EVERY,
    SIGMA_NULL,
    WINDOW,
    Scenario,
    device_rate,
    simulate_stream,
)


def run(scenario: Scenario, seed: int = 7):
    """Score a scenario in-process; returns (labels, out_of_domain flags)."""
    rng = np.random.default_rng(seed)
    gas = simulate_stream(scenario, rng)

    labels, ood = [], []
    for end in range(WINDOW, len(gas), EMIT_EVERY):
        window = gas[end - WINDOW:end]
        result = predict(
            temperature=scenario.temperature,
            humidity=scenario.humidity,
            gas_raw=float(window[-1]),
            temperature_rate=0.0,
            humidity_rate=0.0,
            gas_rate=device_rate(window, scenario.broken_rate),
        )
        labels.append(result["classification"])
        ood.append(not result["domain"]["in_domain"])

    return np.array(labels), np.array(ood)


def test_stable_cargo_raises_no_alarms():
    """Six hours of nothing happening must not produce a Critical."""
    labels, _ = run(Scenario("stable", hours=6.0))

    assert len(labels) > 50
    assert (labels == "Critical").sum() == 0
    assert (labels == "Safe").mean() >= 0.95, (
        f"false-alarm rate too high: {(labels != 'Safe').mean():.1%}"
    )


def test_fast_spoilage_escalates():
    """A ramp inside the Critical band must be caught."""
    labels, _ = run(
        Scenario("fast", hours=6.0, gas_rate_per_s=4.4 * SIGMA_NULL, onset_hours=2.0)
    )

    assert (labels == "Critical").sum() > 0, "Critical-band ramp never escalated"
    assert labels[-1] == "Critical", "escalation did not persist to end of run"


def test_refrigerated_transport_is_flagged_not_answered():
    """A 4 C shipment is a normal condition the model was never fitted on."""
    _, ood = run(Scenario("fridge", hours=3.0, temperature=4.0, humidity=85.0))

    assert ood.all(), (
        f"only {ood.mean():.0%} of refrigerated readings flagged out-of-domain"
    )


def test_broken_firmware_rate_is_caught_by_the_domain_guard():
    """A device sending consecutive deltas instead of window slopes.

    This is the integration failure that cannot be tested without hardware. The
    guard makes it loud rather than silent: consecutive deltas on a noisy signal
    are ~9 sigma from the fitted gas_rate distribution.
    """
    _, ood = run(
        Scenario(
            "broken", hours=4.0, gas_rate_per_s=4.4 * SIGMA_NULL,
            onset_hours=1.0, broken_rate=True,
        )
    )

    assert ood.mean() >= 0.5, (
        f"only {ood.mean():.0%} of malformed-rate requests were flagged; a "
        f"firmware contract break would pass as valid data"
    )


@pytest.mark.xfail(
    strict=True,
    reason="Known gap: a flatlined sensor is indistinguishable from calm cargo. "
           "Needs a liveness check (window variance) the API cannot currently do.",
)
def test_flatlined_sensor_is_detected():
    """A stuck MQ3 reporting a constant value should not read as healthy."""
    labels, ood = run(Scenario("flatline", hours=3.0, stuck_gas=True))

    assert not ((labels == "Safe").all() and not ood.any())
