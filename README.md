# Power Generation Forecasting & Variance Explanation Agent

A runnable Greenko AI/ML case-study prototype using the bundled dummy dataset.

## What it does

1. Loads and validates the energy-generation CSV.
2. Splits records chronologically by unique date (80% earlier dates for training, 20% later dates for testing).
3. Evaluates the supplied forecast against actual generation.
4. Trains a Random Forest regression model using irradiance, wind speed, temperature and outage flag as features.
5. Compares the ML model with the supplied forecast baseline using MAE, RMSE and R².
6. Calculates variance as `actual_mwh - forecast_mwh`.
7. Runs a transparent rule-based diagnostic workflow that checks outage, irradiance and temperature indicators.
8. Displays the results in a browser dashboard.

## Important interpretation

The supplied `forecast_mwh` column is the operational forecast benchmark in this dataset. The Random Forest is an experimental weather-feature model trained to estimate `actual_mwh`. It is compared against the supplied forecast; it is not assumed to be better.

The bundled data contains 60 rows over 30 dates, with two records per date. It has no plant/asset identifier, so the project does not assume that the two rows represent particular plants. The dataset has no separate forecast-weather and observed-weather fields. Therefore, the diagnostic checks identify possible signals, not proven causes.

The current diagnostic component is a deterministic, rule-based workflow. It does **not** call an external LLM and does **not** use RAG. This is stated explicitly rather than claiming GenAI functionality that is not implemented. An LLM can be added later behind an environment-variable-based provider integration.

## Architecture

`CSV -> validation/preprocessing -> chronological train/test split -> Random Forest benchmark + supplied forecast baseline -> variance calculation -> rule-based diagnostic tools -> FastAPI -> HTML dashboard`

## Requirements

- Python 3.10+
- pip

## Run locally (Windows PowerShell)

From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.app:app --reload
```

Open:

- Dashboard: http://127.0.0.1:8000
- API demo/metrics: http://127.0.0.1:8000/api/demo
- Holdout records: http://127.0.0.1:8000/api/records
- Interactive API docs: http://127.0.0.1:8000/docs

## API example

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/run -Method Post -ContentType "application/json" -Body '{"row_index":48}'
```

The API only accepts row indexes in the chronological holdout set, to keep the interactive example out-of-sample.

## Evaluation method

- Unique dates are sorted chronologically.
- The first 80% of unique dates are training data.
- The final 20% are holdout test data.
- Duplicate-date rows stay in the same split.
- MAE and RMSE are reported in MWh; R² is also reported.
- The supplied forecast is evaluated as a baseline.
- The Random Forest uses `n_estimators=100`, `max_depth=3`, `min_samples_leaf=2`, and `random_state=42`.

On the bundled dataset and current split, the supplied forecast baseline performs better than the initial Random Forest weather-only comparison. Treat these results as preliminary because the dataset is small and synthetic/dummy.

## Variance convention

`variance_mwh = actual_mwh - forecast_mwh`

- Positive: actual generation was higher than forecast.
- Negative: actual generation was lower than forecast.
- Percentage: `(variance_mwh / forecast_mwh) * 100`.

## Diagnostic workflow tools

1. `calculate_variance`: computes absolute and percentage variance.
2. `inspect_operational_signals`: checks outage flag, lower irradiance and high temperature.
3. `compose_grounded_explanation`: creates a cautious explanation from observed values.

These rules are transparent heuristics, not a trained causal model. Thresholds should be validated with domain experts and more operational data before production use.

## Limitations and next improvements

- Small dummy dataset; obtain more historical data from multiple assets and seasons.
- Add plant/asset identifiers and clear units/metadata.
- Add forecast weather and actual weather to investigate forecast misses more reliably.
- Compare additional models and tune using time-series cross-validation.
- Add uncertainty intervals and drift monitoring.
- Add an optional LLM explanation layer with strict grounding, output validation, key management and failure fallback.
- Add authentication, logging, automated tests and production monitoring.

## Railway deployment

1. Push this project to a GitHub repository.
2. Create a Railway project and deploy from the GitHub repository.
3. Railway uses the included `Procfile` to start the app.
4. Confirm the generated public URL serves the dashboard and `/api/demo`.
5. Keep credentials out of source control; use Railway variables if adding an external LLM later.
