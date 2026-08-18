import numpy as np
import pytest

from forecast_monitoring.metrics import forecast_metrics, population_stability_index, quality_report
from forecast_monitoring.report import build_report


def test_quality_report_detects_invalid_values():
    report = quality_report([1, 2, np.nan, -1])
    assert report.rows == 4
    assert report.missing_values == 1
    assert report.negative_values == 1
    assert not report.healthy


def test_psi_is_zero_for_identical_distributions():
    values = np.arange(20, dtype=float)
    assert population_stability_index(values, values) == pytest.approx(0.0)


def test_forecast_metrics_are_reproducible():
    metrics = forecast_metrics([10, 12, 14], [9, 13, 15])
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)


def test_report_returns_health_status_and_alerts():
    reference = {"requests": np.arange(20, dtype=float)}
    current = {"requests": np.arange(20, dtype=float) + 1}
    report = build_report(reference, current, [10, 12], [10, 12])
    assert report["status"] == "healthy"
    assert report["performance"]["mae"] == 0
    assert report["alerts"] == []
