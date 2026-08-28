import asyncio

import httpx
import pandas as pd


WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]

AIR_QUALITY_VARIABLES = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "us_aqi",
]


async def fetch_data(
    latitude: float,
    longitude: float,
    timeout: float,
) -> pd.DataFrame:
    common_params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": 4,
        "forecast_days": 3,
        "timezone": "UTC",
    }

    weather_params = {
        **common_params,
        "hourly": WEATHER_VARIABLES,
    }

    air_quality_params = {
        **common_params,
        "hourly": AIR_QUALITY_VARIABLES,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        weather_response, air_quality_response = await asyncio.gather(
            client.get(WEATHER_URL, params=weather_params),
            client.get(AIR_QUALITY_URL, params=air_quality_params),
        )

    weather_response.raise_for_status()
    air_quality_response.raise_for_status()

    weather_payload = weather_response.json()
    air_quality_payload = air_quality_response.json()

    weather_df = pd.DataFrame(weather_payload["hourly"])
    air_quality_df = pd.DataFrame(air_quality_payload["hourly"])

    weather_df["time"] = pd.to_datetime(
        weather_df["time"], utc=True
    )
    air_quality_df["time"] = pd.to_datetime(
        air_quality_df["time"], utc=True
    )

    result = weather_df.merge(
        air_quality_df,
        on="time",
        how="inner",
    )

    return result.sort_values("time").reset_index(drop=True)