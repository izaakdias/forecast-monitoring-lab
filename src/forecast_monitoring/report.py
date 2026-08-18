"""Monitoring report assembly and status thresholds."""

from __future__ import annotations

from typing import Any

from .metrics import forecast_metrics, population_stability_index, quality_report


def build_report(reference_features: dict[str, object], current_features: dict[str, object], actual: object, predicted: object) -> dict[str, Any]:
    drift = {
        feature: round(population_stability_index(reference_features[feature], current_features[feature]), 4)
        for feature in reference_features
    }
    quality = {feature: quality_report(values).__dict__ for feature, values in current_features.items()}
    performance = forecast_metrics(actual, predicted)
    alerts: list[str] = []
    if any(value >= 0.25 for value in drift.values()):
        alerts.append("Investigate severe feature drift before trusting new predictions.")
    elif any(value >= 0.1 for value in drift.values()):
        alerts.append("Review feature drift and compare it with expected seasonality.")
    if performance["mae"] >= 5:
        alerts.append("Forecast MAE is above the configured operating threshold.")
    if not all(item["missing_values"] == 0 and item["non_finite_values"] == 0 for item in quality.values()):
        alerts.append("Input quality checks found missing or non-finite values.")
    status = "critical" if any("severe" in alert or "quality" in alert for alert in alerts) else "warning" if alerts else "healthy"
    return {"status": status, "drift_psi": drift, "quality": quality, "performance": performance, "alerts": alerts}
