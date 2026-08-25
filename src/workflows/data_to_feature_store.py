from datetime import datetime, timedelta
import logging
import os
import sys
import traceback

from dotenv import load_dotenv
import hopsworks
import numpy as np
import pandas as pd
import requests

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Configuration
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


def fetch_raw_data():
    """Fetch raw weather and air quality data from Open-Meteo."""
    end_date = datetime.now()
    # Fetch 14 days of context to safely calculate lags up to 72h and target leads up to 72h
    start_date = end_date - timedelta(days=14)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    log.info(f"Requesting raw data from Open-Meteo ({start_str} to {end_str})...")

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m"
        f"&start_date={start_str}&end_date={end_str}&timezone=auto"
    )

    aqi_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&hourly=pm10,pm2_5,nitrogen_dioxide,ozone,us_aqi"
        f"&start_date={start_str}&end_date={end_str}&timezone=auto"
    )

    w_res = requests.get(weather_url).json()
    a_res = requests.get(aqi_url).json()

    df_w = pd.DataFrame(w_res["hourly"])
    df_a = pd.DataFrame(a_res["hourly"])

    df = pd.merge(df_w, df_a, on="time")
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = CITY_NAME

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all required features matching Hopsworks Feature Group v6 schema."""
    df = df.sort_values("time").copy()

    # 1. Cast integer API features
    df["wind_direction_10m"] = df["wind_direction_10m"].astype("int64")

    # 2. Wind U/V components
    wind_rad = np.radians(df["wind_direction_10m"])
    df["wind_u"] = (-df["wind_speed_10m"] * np.sin(wind_rad)).astype(float)
    df["wind_v"] = (-df["wind_speed_10m"] * np.cos(wind_rad)).astype(float)

    # 3. Calendar & Cyclical features
    df["hour"] = df["time"].dt.hour.astype("int64")
    df["dayofweek"] = df["time"].dt.dayofweek.astype("int64")
    df["month"] = df["time"].dt.month.astype("int64")
    df["dayofyear"] = df["time"].dt.dayofyear.astype("int64")
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype("int64")

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype(float)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype(float)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(float)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(float)

    # 4. Ratios & Seasonal features
    df["pm25_pm10_ratio"] = (df["pm2_5"] / (df["pm10"] + 1e-6)).astype(float)
    df["is_burning_season"] = df["month"].isin([10, 11, 12]).astype("int64")

    # 5. Lag features
    df["pm2_5_lag_1h"] = df["pm2_5"].shift(1).astype(float)
    df["pm2_5_lag_24h"] = df["pm2_5"].shift(24).astype(float)
    df["us_aqi_lag_1h"] = df["us_aqi"].shift(1).astype(float)
    df["us_aqi_lag_24h"] = df["us_aqi"].shift(24).astype(float)
    df["us_aqi_lag_48h"] = df["us_aqi"].shift(48).astype(float)
    df["us_aqi_lag_72h"] = df["us_aqi"].shift(72).astype(float)
    df["wind_speed_lag_24h"] = df["wind_speed_10m"].shift(24).astype(float)
    df["humidity_lag_24h"] = df["relative_humidity_2m"].shift(24).astype(float)

    # 6. Rolling Statistics
    df["pm2_5_roll_mean_6h"] = df["pm2_5"].rolling(6).mean().astype(float)
    df["pm2_5_roll_mean_24h"] = df["pm2_5"].rolling(24).mean().astype(float)
    df["pm2_5_roll_std_24h"] = df["pm2_5"].rolling(24).std().astype(float)
    df["us_aqi_roll_mean_24h"] = df["us_aqi"].rolling(24).mean().astype(float)
    df["us_aqi_roll_std_24h"] = df["us_aqi"].rolling(24).std().astype(float)
    df["us_aqi_roll_min_24h"] = df["us_aqi"].rolling(24).min().astype(float)
    df["us_aqi_roll_max_24h"] = df["us_aqi"].rolling(24).max().astype(float)

    # 7. AQI Change Rates
    df["aqi_change_rate_1h"] = (df["us_aqi"] - df["us_aqi_lag_1h"]).astype(float)
    df["aqi_change_rate_24h"] = (df["us_aqi"] - df["us_aqi_lag_24h"]).astype(float)

    # 8. Target variables (Lead predictions for future AQI)
    df["target_aqi_24h"] = df["us_aqi"].shift(-24).astype(float)
    df["target_aqi_48h"] = df["us_aqi"].shift(-48).astype(float)
    df["target_aqi_72h"] = df["us_aqi"].shift(-72).astype(float)

    # CRITICAL FIX: Only drop rows missing HISTORICAL lags (first 72h of fetch window)
    # Do NOT drop rows with NaN targets (recent rows up to current hour)
    lag_cols = [c for c in df.columns if "lag" in c or "roll" in c]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    return df


def main():
    try:
        if not HOPSWORKS_API_KEY:
            raise ValueError("HOPSWORKS_API_KEY environment variable is missing or empty!")

        raw_df = fetch_raw_data()
        features_df = engineer_features(raw_df)

        log.info("Connecting to Hopsworks Feature Store...")
        project = hopsworks.login(
            host=HOPSWORKS_HOST,
            api_key_value=HOPSWORKS_API_KEY,
            project=PROJECT_NAME,
        )
        fs = project.get_feature_store()

        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
        )

        log.info(f"Upserting {len(features_df)} rows (current hour + updated target history) to Feature Group v{FEATURE_GROUP_VERSION}...")

        try:
            fg.insert(features_df, write_options={"wait_for_job": False})
        except Exception as e:
            if "No materialization job was found" in str(e):
                log.warning("Data uploaded successfully (materialization job check skipped for local Python engine).")
            else:
                raise e
        log.info("Successfully updated Feature Group!")

    except Exception as e:
        log.error("Pipeline failure encountered!")
        log.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()