"""Data quality, drift, and forecast performance metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityReport:
    rows: int
    missing_values: int
    non_finite_values: int
    negative_values: int

    @property
    def healthy(self) -> bool:
        return self.missing_values == 0 and self.non_finite_values == 0 and self.negative_values == 0


def quality_report(values: object) -> QualityReport:
    array = np.asarray(values, dtype=float)
    return QualityReport(
        rows=int(array.size),
        missing_values=int(np.isnan(array).sum()),
        non_finite_values=int((~np.isfinite(array)).sum()),
        negative_values=int(np.sum(np.isfinite(array) & (array < 0))),
    )


def _histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    distribution = counts.astype(float) / max(counts.sum(), 1)
    return np.clip(distribution, 1e-6, None)


def population_stability_index(reference: object, current: object, bins: int = 10) -> float:
    """Compute PSI using shared reference/current quantile boundaries."""
    reference_array = np.asarray(reference, dtype=float)
    current_array = np.asarray(current, dtype=float)
    if reference_array.size < 2 or current_array.size < 2:
        raise ValueError("reference and current samples need at least two values")
    boundaries = np.unique(np.quantile(reference_array, np.linspace(0, 1, bins + 1)))
    if boundaries.size < 3:
        return 0.0
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    reference_distribution = _histogram(reference_array, boundaries)
    current_distribution = _histogram(current_array, boundaries)
    return float(np.sum((current_distribution - reference_distribution) * np.log(current_distribution / reference_distribution)))


def forecast_metrics(actual: object, predicted: object) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted arrays must have the same shape")
    errors = actual_array - predicted_array
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }
