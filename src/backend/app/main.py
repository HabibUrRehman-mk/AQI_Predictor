from collections import defaultdict
from time import monotonic

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.schemas.prediction import PredictionResponse
from app.services.feature_engineering import build_features
from app.services.open_meteo import fetch_data
from app.services.predictor import Predictor


app = FastAPI(
    title="AQI Predictor API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.allowed_origins.split(",")
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)

predictor = Predictor()
request_log: dict[str, list[float]] = defaultdict(list)


def to_float(value) -> float:
    """Convert NumPy/Pandas numeric values to native Python float."""
    return float(value.item() if hasattr(value, "item") else value)


def enforce_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    now = monotonic()
    window_start = now - 60

    request_log[client_ip] = [
        timestamp
        for timestamp in request_log[client_ip]
        if timestamp > window_start
    ]

    if len(request_log[client_ip]) >= settings.requests_per_minute:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
        )

    request_log[client_ip].append(now)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": predictor.ready,
    }


@app.get(
    "/api/predict",
    response_model=PredictionResponse,
)
async def predict(
    request: Request,
    city: str = Query("Faisalabad", min_length=2, max_length=60),
    latitude: float = Query(31.4187, ge=-90, le=90),
    longitude: float = Query(73.0791, ge=-180, le=180),
):
    enforce_rate_limit(request)

    try:
        raw_df = await fetch_data(
            latitude=latitude,
            longitude=longitude,
            timeout=settings.request_timeout,
        )

        feature_df = build_features(raw_df)

        latest_features = feature_df.iloc[[-1]]
        latest_api_row = raw_df.iloc[-1]

        predictions = predictor.predict(latest_features)

        weather = {
            "temperature_2m": to_float(
                latest_api_row["temperature_2m"]
            ),
            "relative_humidity_2m": to_float(
                latest_api_row["relative_humidity_2m"]
            ),
            "surface_pressure": to_float(
                latest_api_row["surface_pressure"]
            ),
            "wind_speed_10m": to_float(
                latest_api_row["wind_speed_10m"]
            ),
            "wind_direction_10m": to_float(
                latest_api_row["wind_direction_10m"]
            ),
        }

        return PredictionResponse(
            city=str(city),
            latitude=float(latitude),
            longitude=float(longitude),
            timestamp=latest_api_row["time"].to_pydatetime(),
            current_aqi=to_float(latest_api_row["us_aqi"]),
            weather=weather,
            predicted_aqi_24h=float(predictions[0]),
            predicted_aqi_48h=float(predictions[1]),
            predicted_aqi_72h=float(predictions[2]),
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Prediction failed: {exc}",
        ) from exc