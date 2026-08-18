"""Generate a deterministic monitoring report from synthetic data."""

from __future__ import annotations

import json

import numpy as np

from .report import build_report


def main() -> None:
    rng = np.random.default_rng(42)
    reference = {
        "requests_last_hour": rng.normal(20, 4, 400),
        "active_drivers": rng.normal(26, 5, 400),
        "avg_pickup_eta_minutes": rng.normal(6, 1, 400),
    }
    current = {
        "requests_last_hour": rng.normal(23, 5, 400),
        "active_drivers": rng.normal(23, 5, 400),
        "avg_pickup_eta_minutes": rng.normal(7, 1.2, 400),
    }
    actual = rng.normal(24, 5, 200)
    predicted = actual + rng.normal(0, 2.5, 200)
    report = build_report(reference, current, actual, predicted)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
