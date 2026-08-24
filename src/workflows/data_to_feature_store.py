from datetime import datetime, timedelta
import logging
import os
import hopsworks
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Config
CITY_NAME = "Faisalabad"
LATITUDE = 31.4187
LONGITUDE = 73.0791

PROJECT_NAME = "AQI_Predictor_fsd"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

FEATURE_GROUP_NAME = "weather_aqi_hourly"
FEATURE_GROUP_VERSION = 6
PRIMARY_KEY = ["city", "time"]
EVENT_TIME = "time"


def fetch_raw_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches weather and air quality for the rolling window directly into memory."""
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        "timezone": "UTC",
    }

    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "us_aqi"],
        "timezone": "UTC",
    }

    log.info("Fetching API data from %s to %s...", start_date, end_date)
    res_weather = requests.get(weather_url, params=weather_params, timeout=30).json()
    res_aq = requests.get(aq_url, params=aq_params, timeout=30).json()

    df_weather = pd.DataFrame(res_weather["hourly"])
    df_weather["time"] = pd.to_datetime(df_weather["time"])

    df_aq = pd.DataFrame(res_aq["hourly"])
    df_aq["time"] = pd.to_datetime(df_aq["time"])

    df = pd.merge(df_weather, df_aq, on="time", how="inner")
    df["city"] = CITY_NAME
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes features across the dataset."""
    df = df.sort_values("time").reset_index(drop=True)

    # 1. Wind Vector Components
    wind_rad = np.radians(df["wind_direction_10m"])
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(wind_rad)
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(wind_rad)

    # 2. Temporal & Cyclical Features
    df["hour"] = df["time"].dt.hour.astype("int64")
    df["dayofweek"] = df["time"].dt.dayofweek.astype("int64")
    df["month"] = df["time"].dt.month.astype("int64")
    df["dayofyear"] = df["time"].dt.dayofyear.astype("int64")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int64")

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # 3. Lag Features
    df["pm2_5_lag_1h"] = df["pm2_5"].shift(1)
    df["pm2_5_lag_24h"] = df["pm2_5"].shift(24)
    df["us_aqi_lag_1h"] = df["us_aqi"].shift(1)
    df["us_aqi_lag_24h"] = df["us_aqi"].shift(24)
    df["us_aqi_lag_48h"] = df["us_aqi"].shift(48)
    df["us_aqi_lag_72h"] = df["us_aqi"].shift(72)

    # 4. Rolling Features
    df["pm2_5_roll_mean_6h"] = df["pm2_5"].shift(1).rolling(window=6).mean()
    df["pm2_5_roll_mean_24h"] = df["pm2_5"].shift(1).rolling(window=24).mean()
    df["us_aqi_roll_mean_24h"] = df["us_aqi"].shift(1).rolling(window=24).mean()

    # 5. Rates of Change
    df["aqi_change_rate_1h"] = (df["pm2_5_lag_1h"] - df["pm2_5"].shift(2)) / (df["pm2_5"].shift(2) + 1e-5)
    df["aqi_change_rate_24h"] = (df["pm2_5_lag_1h"] - df["pm2_5_lag_24h"]) / (df["pm2_5_lag_24h"] + 1e-5)

    # 6. Targets
    df["target_aqi_24h"] = df["us_aqi"].shift(-24)
    df["target_aqi_48h"] = df["us_aqi"].shift(-48)
    df["target_aqi_72h"] = df["us_aqi"].shift(-72)

    # Drop boundaries where lag calculations created NaNs
    df = df.dropna(subset=["us_aqi_lag_72h"]).reset_index(drop=True)

    # Hopsworks formatting
    df["city"] = df["city"].astype(str)
    df["us_aqi"] = df["us_aqi"].astype("int64")
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC").dt.tz_localize(None).astype("datetime64[us]")

    return df


def main():
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY environment variable is not set!")

    today = datetime.now()
    seven_days_ago = today - timedelta(days=7)

    # 1. Fetch raw data into memory
    raw_df = fetch_raw_data(
        start_date=seven_days_ago.strftime("%Y-%m-%d"),
        end_date=today.strftime("%Y-%m-%d")
    )

    # 2. Engineer features in memory
    log.info("Engineering features across memory DataFrame...")
    features_df = engineer_features(raw_df)

    # 3. Connect to Hopsworks and insert directly
    log.info("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        port=443,
        project=PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=PRIMARY_KEY,
        event_time=EVENT_TIME,
        time_travel_format="HUDI",
        online_enabled=True,
    )

    log.info("Upserting %d rows into Feature Group v%d...", len(features_df), fg.version)
    fg.insert(features_df, write_options={"wait_for_job": True})
    log.info("Hourly execution completed successfully.")


if __name__ == "__main__":
    main()