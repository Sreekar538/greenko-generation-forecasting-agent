
# Greenko Generation Forecasting & Variance Explanation Agent

A renewable energy forecasting and variance analysis prototype developed for the Greenko AI/ML case study. The application combines machine learning, rule-based operational diagnostics, TF-IDF-based Retrieval-Augmented Generation (RAG), and Google Gemini to explain generation variances.

## 1. Project Overview

The system analyzes actual versus forecasted power generation, identifies operational indicators associated with generation differences, retrieves relevant renewable-energy reference knowledge, and generates natural-language explanations.

### Key Features

1. CSV data loading, validation, and preprocessing.
2. Chronological training and testing split.
3. Random Forest regression model.
4. Supplied forecast baseline evaluation.
5. MAE, RMSE, and R² model evaluation.
6. Actual-versus-forecast variance calculation.
7. Rule-based operational signal detection.
8. TF-IDF-based knowledge retrieval.
9. Google Gemini-generated explanations using retrieved context.
10. Fallback explanations when Gemini is unavailable.
11. FastAPI backend and browser-based dashboard.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| RAG retrieval | TF-IDF, Cosine Similarity |
| Generative AI | Google Gemini |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| API documentation | Swagger UI |
| Version control | Git, GitHub |

## 3. Project Architecture

```text
Generation Forecast CSV
          |
          v
Data Loading and Validation
          |
          v
Chronological Train-Test Split
          |
          v
Random Forest Model
          |
          v
Forecast and Actual Comparison
          |
          v
Variance Calculation
          |
          v
Rule-Based Diagnostics
          |
          v
Operational Context Preparation
          |
          v
TF-IDF Knowledge Retrieval
          |
          v
Retrieved Reference Documents
          |
          v
Google Gemini Explanation
          |
          v
FastAPI JSON Response
          |
          v
Browser Dashboard
```

## 4. Dataset

The project uses the bundled sample generation dataset:

`data/generation_forecast.csv`

### Input Features

- `irradiance_wm2`: Solar irradiance.
- `wind_speed_ms`: Wind speed.
- `temperature_c`: Temperature.
- `outage_flag`: Recorded outage indicator.

### Target Variable

- `actual_mwh`: Actual power generation.

### Forecast Benchmark

- `forecast_mwh`: Supplied operational forecast.

The dataset contains 60 records across 30 dates, with two records per date.

## 5. Machine Learning Model

The project uses a Random Forest Regressor to estimate actual generation from weather and outage features.

### Model Configuration

- `n_estimators = 100`
- `max_depth = 3`
- `min_samples_leaf = 2`
- `random_state = 42`

### Data Splitting

- First 80% of unique chronological dates: Training.
- Final 20% of unique chronological dates: Testing.
- Records sharing the same date remain in the same split.

### Evaluation Metrics

- MAE: Mean Absolute Error.
- RMSE: Root Mean Squared Error.
- R²: Coefficient of Determination.

### Current Evaluation Results

| Metric | Supplied Forecast | Random Forest |
|---|---:|---:|
| MAE (MWh) | 4.604 | 7.281 |
| RMSE (MWh) | 5.169 | 8.762 |
| R² | 0.859 | 0.594 |

The supplied forecast performs better than the experimental Random Forest model on the current holdout dataset. The project reports this comparison transparently rather than claiming that the ML model is superior.

## 6. Variance Calculation

The system calculates generation variance using:

`variance_mwh = actual_mwh - forecast_mwh`

Interpretation:

- Positive variance: Actual generation exceeded forecast.
- Negative variance: Actual generation was below forecast.
- Zero variance: Actual generation matched forecast.

Variance percentage is calculated relative to forecast generation.

## 7. Operational Diagnostic Workflow

The rule-based diagnostic component checks:

1. Recorded equipment outage.
2. Lower solar irradiance (below 450 W/m²).
3. High recorded temperature (above 38°C).

These indicators provide operational context. They do not establish that a particular factor caused the generation variance.

## 8. Retrieval-Augmented Generation (RAG)

The project implements a small local knowledge base:

`data/knowledge_base.json`

It contains six general renewable-energy reference documents covering:

- Solar irradiance and generation.
- High temperature and solar equipment.
- Equipment outages.
- Wind speed and wind generation.
- Forecast variance interpretation.
- Operational data limitations.

### RAG Workflow

1. Prepare a query from the selected operational record and its diagnostic context.
2. Load the local knowledge-base documents.
3. Convert document text into TF-IDF vectors.
4. Convert the operational query into a TF-IDF vector.
5. Calculate cosine similarity.
6. Rank documents by relevance.
7. Retrieve the top two relevant documents.
8. Pass the retrieved reference knowledge and operational data to Gemini.

The API response includes the retrieved document IDs, titles, content, and similarity scores.

The knowledge base contains general reference guidance, not Greenko internal documentation.

## 9. Google Gemini Integration

Google Gemini generates natural-language explanations using:

- Actual generation.
- Forecast generation.
- Calculated variance.
- Recorded operational indicators.
- Retrieved RAG reference documents.

### Safety and Reliability

- API credentials are loaded from environment variables.
- The application avoids unsupported causal claims.
- Retrieved knowledge is treated as general reference guidance.
- A rule-based fallback is provided if Gemini is unavailable.
- External API failures are handled without stopping variance calculations.

## 10. FastAPI Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard |
| `/api/demo` | GET | Dataset and model information |
| `/api/records` | GET | Chronological holdout records |
| `/api/run` | POST | Forecast, variance, diagnostics, RAG and Gemini |
| `/docs` | GET | Interactive Swagger documentation |

### API Test Example

Request:

```json
{
  "row_index": 48
}
```

Example verified output:

- Forecast: 51.42 MWh.
- Actual: 56.80 MWh.
- Variance: +5.38 MWh.
- Variance percentage: +10.46%.
- Retrieved documents: 2.
- Gemini status: success (during the verified test).

The API accepts only row indexes belonging to the chronological holdout dataset.

## 11. Local Installation and Execution

### Requirements

- Python 3.10 or later.
- pip.
- Google Gemini API key for external LLM explanations.

### Windows PowerShell Setup

Clone the repository:

```powershell
git clone https://github.com/Sreekar538/greenko-generation-forecasting-agent.git
cd greenko-generation-forecasting-agent
```

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Never commit the `.env` file to GitHub.

### Start the Application

```powershell
uvicorn backend.app:app --reload --port 8000
```

Open the application:

- Dashboard: http://127.0.0.1:8000
- API documentation: http://127.0.0.1:8000/docs
- Demo information: http://127.0.0.1:8000/api/demo
- Holdout records: http://127.0.0.1:8000/api/records

If port 8000 is occupied, use:

```powershell
uvicorn backend.app:app --reload --port 8001
```

Then use port 8001 in the browser.

## 12. Project Limitations

- The bundled dataset is small and marked as dummy data.
- There are no plant or asset identifiers.
- There are no separate forecast-weather and observed-weather fields.
- The Random Forest model does not outperform the supplied forecast baseline.
- Operational thresholds are simple heuristics.
- The local RAG knowledge base contains only six general reference documents.
- TF-IDF retrieval depends on the available knowledge-base content.
- Gemini availability depends on the external API.
- The project is a case-study prototype, not a production-ready operational platform.

## 13. Future Improvements

- Add larger historical generation datasets.
- Include plant and equipment identifiers.
- Improve feature engineering and time-series model evaluation.
- Expand and validate the operational knowledge base.
- Evaluate retrieval quality using a labelled test set.
- Add RAG source citations and stronger response validation.
- Add authentication, logging, and monitoring.
- Deploy the application using a suitable hosting platform.

## 14. Repository Structure

```text
greenko-generation-forecasting-agent/
|
|-- backend/
|   |-- app.py
|
|-- data/
|   |-- generation_forecast.csv
|   |-- knowledge_base.json
|
|-- frontend/
|   |-- index.html
|   |-- style.css
|   |-- script.js
|
|-- .env
|-- .gitignore
|-- Procfile
|-- README.md
|-- requirements.txt
|-- runtime.txt
```

The `.env` file is local configuration and must not be committed to the repository.

## 15. Conclusion

This project demonstrates an integrated renewable-energy generation analysis workflow combining machine learning, rule-based diagnostics, local knowledge retrieval, generative AI, and API-based delivery.

It provides transparent forecast variance calculations and contextual explanations while clearly communicating model performance and data limitations.
