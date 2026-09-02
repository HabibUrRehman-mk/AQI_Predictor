import asyncio

import httpx
import pandas as pd
from app.core.config import settings

WEATHER_URL = settings.WEATHER_URL
AQ_URL = settings.AQ_URL

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]

AQ_VARIABLES = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "us_aqi",
]


async def fetch_recent_data(
    latitude: float,
    longitude: float,
    timeout: float,
    past_days: int = 5,
) -> pd.DataFrame:
    """Fetch the last n days of hourly weather and AQ data required for lag features."""
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": past_days,
        "forecast_days": 1,
        "hourly": WEATHER_VARIABLES,
        "timezone": "UTC",
    }
    aq_params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": past_days,
        "forecast_days": 1,
        "hourly": AQ_VARIABLES,
        "timezone": "UTC",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        weather_response, aq_response = await asyncio.gather(
            client.get(WEATHER_URL, params=weather_params),
            client.get(AQ_URL, params=aq_params),
        )

    weather_response.raise_for_status()
    aq_response.raise_for_status()

    weather_df = pd.DataFrame(weather_response.json()["hourly"])
    aq_df = pd.DataFrame(aq_response.json()["hourly"])

    weather_df["time"] = pd.to_datetime(weather_df["time"], utc=True)
    aq_df["time"] = pd.to_datetime(aq_df["time"], utc=True)

    merged_df = pd.merge(weather_df, aq_df, on="time", how="inner")
    return merged_df.sort_values("time").reset_index(drop=True)


async def fetch_data(
    latitude: float,
    longitude: float,
    timeout: float,
) -> pd.DataFrame:
    """Backward-compatible alias for the fetch_recent_data function."""
    return await fetch_recent_data(
        latitude=latitude,
        longitude=longitude,
        timeout=timeout,
        past_days=5,
    )