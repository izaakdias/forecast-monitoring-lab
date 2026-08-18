# Forecast Monitoring Lab

A lightweight monitoring toolkit for detecting data drift, forecast degradation, and input-quality issues in demand prediction systems.

This repository uses synthetic baseline and current-period data. It is designed as a safe, reproducible portfolio project that demonstrates the operational side of machine learning without exposing proprietary data.

## What this demonstrates

- Data-quality checks for missing, non-finite, and out-of-range values
- Population Stability Index (PSI) for numeric feature drift
- Forecast MAE and RMSE tracking
- Threshold-based monitoring status and actionable alerts
- A small command-line report with JSON output
- Tests and GitHub Actions CI

These controls map directly to the maintenance and monitoring concerns that arise after a model is deployed.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m forecast_monitoring.cli
```

Run the tests:

```bash
pytest
```

## Example report

The default command creates deterministic synthetic observations and reports:

- PSI for `requests_last_hour`, `active_drivers`, and `avg_pickup_eta_minutes`;
- current-period MAE and RMSE;
- data-quality violations;
- an overall `healthy`, `warning`, or `critical` status;
- recommended follow-up actions.

The thresholds are intentionally explicit and configurable in code so that a production team can tune them by segment, model, and business impact.

## Next iterations

1. Add time-windowed metrics by H3 cell and demand level.
2. Persist monitoring events to a metrics store.
3. Add alert routing and runbook links.
4. Compare drift against seasonality-aware reference windows.
5. Add a dashboard for model owners and operations teams.
