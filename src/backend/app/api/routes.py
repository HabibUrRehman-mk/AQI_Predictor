import asyncio
from collections import defaultdict
from time import monotonic

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.config import settings
from app.schemas.prediction import PredictionResponse
from app.services.feature_engineering import engineer_features_for_inference
from app.services.open_meteo import fetch_recent_data

router = APIRouter()
request_log: dict[str, list[float]] = defaultdict(list)


def to_float(value) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def enforce_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    now = monotonic()
    window_start = now - 60

    request_log[client_ip] = [
        timestamp for timestamp in request_log.get(client_ip, []) if timestamp > window_start
    ]

    if len(request_log[client_ip]) >= settings.requests_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    request_log[client_ip].append(now)


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    predictor = getattr(request.app.state, "predictor", None)
    return {
        "status": "ok",
        "model_loaded": bool(predictor and predictor.ready),
        "models": list(getattr(predictor.registry, "loaded_models", {}).keys()) if predictor else [],
        "startup_error": getattr(request.app.state, "startup_error", None),
    }


@router.post("/admin/models/download")
async def download_models(request: Request) -> dict[str, object]:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor service is not configured.")

    try:
        loaded_models = await asyncio.to_thread(predictor.reload_models)
        return {
            "status": "success",
            "message": "Models downloaded from Hopsworks and loaded into RAM.",
            "models": list(loaded_models.keys()),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to download and load models: {exc}") from exc


@router.get(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    request: Request,
    city: str = Query("Faisalabad", min_length=2, max_length=60),
    latitude: float = Query(31.4187, ge=-90, le=90),
    longitude: float = Query(73.0791, ge=-180, le=180),
):
    enforce_rate_limit(request)
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor service is not configured.")

    try:
        raw_df = await fetch_recent_data(
            latitude=latitude,
            longitude=longitude,
            timeout=settings.request_timeout,
            past_days=5,
        )
        feature_df = engineer_features_for_inference(raw_df)
        latest_features = feature_df.iloc[[-1]].copy()
        latest_api_row = raw_df.iloc[-1]

        predictions = predictor.predict(latest_features)

        weather = {
            "temperature_2m": to_float(latest_api_row["temperature_2m"]),
            "relative_humidity_2m": to_float(latest_api_row["relative_humidity_2m"]),
            "surface_pressure": to_float(latest_api_row["surface_pressure"]),
            "wind_speed_10m": to_float(latest_api_row["wind_speed_10m"]),
            "wind_direction_10m": to_float(latest_api_row["wind_direction_10m"]),
            "pm2_5": to_float(latest_api_row["pm2_5"]),
            "pm10": to_float(latest_api_row["pm10"]),
            "nitrogen_dioxide": to_float(latest_api_row["nitrogen_dioxide"]),
            "ozone": to_float(latest_api_row["ozone"]),
        }

        history = [
            {
                "time": row["time"].isoformat(),
                "aqi": to_float(row["us_aqi"]),
            }
            for _, row in raw_df.tail(73).iterrows()
        ]

        return PredictionResponse(
            city=str(city),
            latitude=float(latitude),
            longitude=float(longitude),
            timestamp=latest_api_row["time"].to_pydatetime(),
            current_aqi=to_float(latest_api_row["us_aqi"]),
            weather=weather,
            history=history,
            predicted_aqi_24h=float(predictions[0]),
            predicted_aqi_48h=float(predictions[1]),
            predicted_aqi_72h=float(predictions[2]),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Prediction failed: {exc}") from exc
