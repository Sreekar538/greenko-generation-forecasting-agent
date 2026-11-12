
from pathlib import Path
import json
from typing import Any
import os

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from google import genai

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# --------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# --------------------------------------------------
# 2. PROJECT PATHS AND DATA CONFIGURATION
# --------------------------------------------------

BASE = Path(__file__).resolve().parents[1]

DATA_PATH = BASE / "data" / "generation_forecast.csv"
FRONTEND_PATH = BASE / "frontend"

REQUIRED_COLUMNS = [
    "date",
    "irradiance_wm2",
    "wind_speed_ms",
    "temperature_c",
    "outage_flag",
    "forecast_mwh",
    "actual_mwh"
]

FEATURES = [
    "irradiance_wm2",
    "wind_speed_ms",
    "temperature_c",
    "outage_flag"
]

TARGET = "actual_mwh"


# --------------------------------------------------
# 3. RAG KNOWLEDGE RETRIEVAL
# --------------------------------------------------

KNOWLEDGE_BASE_PATH = BASE / "data" / "knowledge_base.json"


def retrieve_knowledge(query: str, top_k: int = 2) -> list[dict]:

    if not KNOWLEDGE_BASE_PATH.exists():
        print("RAG knowledge base not found.")
        return []

    try:
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8-sig") as file:
            documents = json.load(file)

        if not documents:
            return []

        corpus = [
            f"{doc['title']} {doc['content']}"
            for doc in documents
        ]

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        document_vectors = vectorizer.fit_transform(corpus)
        query_vector = vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            document_vectors
        ).flatten()

        ranked_indices = similarities.argsort()[::-1]

        results = []

        for index in ranked_indices:
            score = float(similarities[index])

            if score <= 0:
                continue

            doc = documents[index]

            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "similarity": round(score, 4)
            })

            if len(results) >= top_k:
                break

        return results

    except Exception as error:
        print(f"RAG retrieval failed: {type(error).__name__}: {error}")
        return []


# --------------------------------------------------
# 3. GEMINI LLM EXPLANATION
# --------------------------------------------------


def generate_llm_explanation(
    context: str,
    retrieved_knowledge: list[dict] | None = None
) -> tuple[str, str]:

    if gemini_client is None:
        return (
            "Gemini API key is not configured. "
            "Using the rule-based explanation.",
            "not_configured"
        )

    knowledge_context = "\n\n".join(
        f"[{doc['id']}] {doc['title']}: {doc['content']}"
        for doc in (retrieved_knowledge or [])
    )

    if not knowledge_context:
        knowledge_context = (
            "No relevant knowledge-base documents were retrieved. "
            "Use only the supplied operational data."
        )

    prompt = f"""
You are an assistant for renewable energy operations.

Analyze generation variance using the supplied operational data
and retrieved reference knowledge.

Requirements:
1. Explain whether actual generation was above or below forecast.
2. Mention variance in MWh and percentage.
3. Discuss relevant recorded operational indicators.
4. Use retrieved knowledge only as general reference guidance.
5. Never invent missing information.
6. Do not claim that an indicator proves causation.
7. Clearly mention uncertainty.
8. Keep the explanation between 2 and 4 sentences.

Retrieved reference knowledge:

{knowledge_context}

Operational data:

{context}
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "http_options": {"timeout": 12000},
                "max_output_tokens": 250,
                "temperature": 0.2
            }
        )

        if response.text:
            return response.text.strip(), "success"

        return (
            "Gemini returned an empty response. "
            "Using the rule-based explanation.",
            "fallback"
        )

    except Exception as error:
        print(
            f"Gemini API request failed: "
            f"{type(error).__name__}: {error}"
        )

        return (
            "Gemini is temporarily unavailable. "
            "Please use the rule-based explanation.",
            "fallback"
        )


# --------------------------------------------------
# 4. LOAD AND VALIDATE DATASET
# --------------------------------------------------

def load_dataset() -> pd.DataFrame:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    frame = pd.read_csv(DATA_PATH)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce"
    )

    numeric_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column != "date"
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce"
        )

    frame = frame.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    frame = frame.sort_values(
        ["date"],
        kind="stable"
    ).reset_index(drop=True)

    frame["outage_flag"] = frame["outage_flag"].astype(int)

    if len(frame) < 3:
        raise ValueError(
            "At least three valid records are required."
        )

    return frame


DATA = load_dataset()


# --------------------------------------------------
# 5. CHRONOLOGICAL TRAIN-TEST SPLIT
# --------------------------------------------------

def chronological_split(frame: pd.DataFrame):

    unique_dates = sorted(
        frame["date"].unique()
    )

    if len(unique_dates) < 2:
        raise ValueError(
            "At least two unique dates are required."
        )

    split_at = max(
        1,
        int(len(unique_dates) * 0.80)
    )

    split_at = min(
        split_at,
        len(unique_dates) - 1
    )

    train_dates = set(
        unique_dates[:split_at]
    )

    train_mask = frame["date"].isin(
        train_dates
    )

    train = frame.loc[train_mask].copy()
    test = frame.loc[~train_mask].copy()

    return train, test


TRAIN, TEST = chronological_split(DATA)


# --------------------------------------------------
# 6. RANDOM FOREST MODEL
# --------------------------------------------------

MODEL = RandomForestRegressor(
    n_estimators=100,
    max_depth=3,
    min_samples_leaf=2,
    random_state=42
)

MODEL.fit(
    TRAIN[FEATURES],
    TRAIN[TARGET]
)

TEST = TEST.copy()

TEST["ml_prediction_mwh"] = MODEL.predict(
    TEST[FEATURES]
)


# --------------------------------------------------
# 7. MODEL EVALUATION
# --------------------------------------------------

def regression_metrics(
    actual,
    predicted
) -> dict[str, float]:

    return {
        "mae_mwh": round(
            float(mean_absolute_error(actual, predicted)),
            3
        ),
        "rmse_mwh": round(
            float(
                np.sqrt(
                    mean_squared_error(actual, predicted)
                )
            ),
            3
        ),
        "r2": round(
            float(r2_score(actual, predicted)),
            3
        )
    }


BASELINE_METRICS = regression_metrics(
    TEST[TARGET],
    TEST["forecast_mwh"]
)

ML_METRICS = regression_metrics(
    TEST[TARGET],
    TEST["ml_prediction_mwh"]
)


# --------------------------------------------------
# 8. VARIANCE CALCULATION
# --------------------------------------------------

def calculate_variance(
    forecast_mwh: float,
    actual_mwh: float
) -> dict[str, float]:

    variance = float(
        actual_mwh - forecast_mwh
    )

    percentage = (
        variance / forecast_mwh * 100.0
        if forecast_mwh
        else 0.0
    )

    return {
        "variance_mwh": round(variance, 2),
        "variance_percent": round(percentage, 2)
    }


# --------------------------------------------------
# 9. OPERATIONAL SIGNAL INSPECTION
# --------------------------------------------------

def inspect_operational_signals(
    row: pd.Series
) -> list[dict[str, str]]:

    signals = []

    if int(row["outage_flag"]) == 1:

        signals.append({
            "signal": "Recorded outage",
            "observation": "The outage flag is 1 for this record.",
            "interpretation": (
                "An outage may have reduced available generation."
            )
        })

    if float(row["irradiance_wm2"]) < 450:

        signals.append({
            "signal": "Lower solar irradiance",
            "observation": (
                f"Irradiance is "
                f"{float(row['irradiance_wm2']):.1f} W/m²."
            ),
            "interpretation": (
                "Lower irradiance can be associated "
                "with lower solar generation."
            )
        })

    if float(row["temperature_c"]) > 38:

        signals.append({
            "signal": "High recorded temperature",
            "observation": (
                f"Temperature is "
                f"{float(row['temperature_c']):.1f} °C."
            ),
            "interpretation": (
                "High temperature can affect equipment efficiency; "
                "this dataset cannot prove that it caused the variance."
            )
        })

    if not signals:

        signals.append({
            "signal": "No threshold-based warning found",
            "observation": (
                "The outage, irradiance and temperature "
                "checks did not trigger."
            ),
            "interpretation": (
                "Other operational factors may explain the difference; "
                "more detailed plant data is needed."
            )
        })

    return signals


# --------------------------------------------------
# 10. RULE-BASED DIAGNOSTIC WORKFLOW
# --------------------------------------------------

def run_diagnostic_agent(
    row: pd.Series,
    variance: dict[str, float]
) -> dict[str, Any]:

    signals = inspect_operational_signals(row)

    amount = variance["variance_mwh"]

    if amount > 0:
        direction = (
            "Actual generation was higher than the supplied forecast."
        )

    elif amount < 0:
        direction = (
            "Actual generation was lower than the supplied forecast."
        )

    else:
        direction = (
            "Actual generation matched the supplied forecast."
        )

    signal_text = " ".join(
        item["interpretation"]
        for item in signals
    )

    explanation = (
        f"{direction} "
        f"The variance was {variance['variance_mwh']:.2f} MWh "
        f"({variance['variance_percent']:.2f}%). "
        f"{signal_text} "
        "These are observable indicators, not confirmed causal findings."
    )

    return {
        "agent_type": "transparent rule-based diagnostic workflow",

        "tools_used": [
            "calculate_variance",
            "inspect_operational_signals",
            "compose_grounded_explanation"
        ],

        "decision_steps": [
            "Calculate actual minus supplied forecast.",
            "Check the outage flag, irradiance and temperature against simple thresholds.",
            "Generate an explanation using observed values and state uncertainty."
        ],

        "signals": signals,

        "explanation": explanation
    }


# --------------------------------------------------
# 11. RECORD SERIALIZATION
# --------------------------------------------------

def serialize_record(
    index: int,
    row: pd.Series
) -> dict[str, Any]:

    return {
        "row_index": int(index),

        "date": row["date"].strftime(
            "%Y-%m-%d"
        ),

        "irradiance_wm2": round(
            float(row["irradiance_wm2"]), 2
        ),

        "wind_speed_ms": round(
            float(row["wind_speed_ms"]), 2
        ),

        "temperature_c": round(
            float(row["temperature_c"]), 2
        ),

        "outage_flag": int(
            row["outage_flag"]
        ),

        "forecast_mwh": round(
            float(row["forecast_mwh"]), 2
        ),

        "actual_mwh": round(
            float(row["actual_mwh"]), 2
        )
    }


# --------------------------------------------------
# 12. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Power Generation Forecasting & Variance Explanation Agent",

    description=(
        "A power generation forecasting evaluation "
        "and variance explanation application."
    ),

    version="1.1.0"
)


app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_PATH)),
    name="static"
)


class RunRequest(BaseModel):

    row_index: int = Field(
        default=48,
        ge=0,
        description="Row index from the chronological holdout set."
    )


# --------------------------------------------------
# 13. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return FileResponse(
        FRONTEND_PATH / "index.html"
    )


# --------------------------------------------------
# 14. DEMO INFORMATION ENDPOINT
# --------------------------------------------------

@app.get("/api/demo")
def demo():

    return {
        "case_study": (
            "Power Generation Forecasting "
            "& Variance Explanation Agent"
        ),

        "status": "working",

        "dataset": {
            "records": int(len(DATA)),

            "unique_dates": int(
                DATA["date"].nunique()
            ),

            "missing_values_after_validation": int(
                DATA[REQUIRED_COLUMNS].isna().sum().sum()
            ),

            "train_records": int(len(TRAIN)),

            "test_records": int(len(TEST)),

            "train_last_date": (
                TRAIN["date"].max().strftime("%Y-%m-%d")
            ),

            "test_first_date": (
                TEST["date"].min().strftime("%Y-%m-%d")
            )
        },

        "metrics": {
            "supplied_forecast_baseline": BASELINE_METRICS,

            "random_forest_weather_model": ML_METRICS
        },

        "note": (
            "The supplied forecast is the benchmark. "
            "The Random Forest model is a comparison "
            "trained on weather and outage features."
        )
    }


# --------------------------------------------------
# 15. HOLDOUT RECORDS ENDPOINT
# --------------------------------------------------

@app.get("/api/records")
def records():

    return {
        "records": [
            serialize_record(int(index), row)

            for index, row in TEST.iterrows()
        ]
    }


# --------------------------------------------------
# 16. MAIN FORECAST AND VARIANCE ENDPOINT
# --------------------------------------------------


@app.post("/api/run")
def run_case(request: RunRequest):

    selected = TEST[
        TEST.index == request.row_index
    ]

    if selected.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "Choose a row from the chronological "
                "holdout records shown in the dashboard."
            )
        )

    row = selected.iloc[0]

    # 1. Calculate forecast variance
    variance = calculate_variance(
        float(row["forecast_mwh"]),
        float(row["actual_mwh"])
    )

    # 2. Run rule-based diagnostic workflow
    diagnostics = run_diagnostic_agent(
        row,
        variance
    )

    # 3. Prepare operational context
    context = f"""
Date: {row['date'].strftime('%Y-%m-%d')}

Forecast generation: {row['forecast_mwh']} MWh

Actual generation: {row['actual_mwh']} MWh

Variance: {variance['variance_mwh']} MWh

Variance percentage: {variance['variance_percent']}%

Solar irradiance: {row['irradiance_wm2']} W/m2

Wind speed: {row['wind_speed_ms']} m/s

Temperature: {row['temperature_c']} C

Outage flag: {row['outage_flag']}

Rule-based signals: {diagnostics['signals']}

Rule-based explanation: {diagnostics['explanation']}
"""

    # 4. Retrieve relevant knowledge using RAG
    retrieved_knowledge = retrieve_knowledge(
        context,
        top_k=2
    )

    # 5. Generate Gemini explanation using retrieved knowledge
    llm_explanation, llm_status = generate_llm_explanation(
        context,
        retrieved_knowledge
    )

    # 6. Return complete API response
    return {

        "record": serialize_record(
            request.row_index,
            row
        ),

        "forecast_mwh": round(
            float(row["forecast_mwh"]), 2
        ),

        "actual_mwh": round(
            float(row["actual_mwh"]), 2
        ),

        "random_forest_estimate_mwh": round(
            float(row["ml_prediction_mwh"]), 2
        ),

        "variance": variance,

        "diagnostic": diagnostics,

        "rag": {
            "retrieval_enabled": True,
            "documents_retrieved": len(retrieved_knowledge),
            "retrieved_documents": retrieved_knowledge
        },

        "llm_explanation": llm_explanation,

        "llm_status": llm_status,

        "evaluation": {
            "supplied_forecast_baseline": BASELINE_METRICS,
            "random_forest_weather_model": ML_METRICS
        },

        "limitations": [
            "The bundled dataset is small and marked as dummy data in the assignment README.",

            "There are two records per date but no plant or asset identifier.",

            "The data has no separate forecast-weather versus observed-weather fields.",

            "Threshold-based signals are indicators, not proof that a factor caused the variance.",

            "The knowledge base contains general renewable-energy reference guidance, not Greenko internal documentation.",

            "RAG uses TF-IDF retrieval over a small local knowledge base. Retrieval quality depends on the available documents.",

            "Gemini availability is subject to external API availability. A rule-based fallback is provided."
        ]
    }
