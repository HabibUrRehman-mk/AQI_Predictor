import numpy as np
import pandas as pd
from app.core.config import settings

BURNING_SEASON_MONTHS = {10, 11}

FINAL_FEATURES = settings.FINAL_FEATURES


def engineer_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the exact feature set used during training and inference."""
    df = df.sort_values("time").reset_index(drop=True).copy()

    wind_rad = np.radians(df["wind_direction_10m"])
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(wind_rad)
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(wind_rad)

    df["hour"] = df["time"].dt.hour.astype("int64")
    df["dayofweek"] = df["time"].dt.dayofweek.astype("int64")
    df["month"] = df["time"].dt.month.astype("int64")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int64")
    df["is_burning_season"] = df["month"].isin(BURNING_SEASON_MONTHS).astype("int64")

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

    df["pm2_5_lag_1h"] = df["pm2_5"].shift(1)
    df["us_aqi_lag_1h"] = df["us_aqi"].shift(1)
    df["us_aqi_lag_24h"] = df["us_aqi"].shift(24)
    df["us_aqi_lag_48h"] = df["us_aqi"].shift(48)
    df["us_aqi_lag_72h"] = df["us_aqi"].shift(72)

    df["pm2_5_roll_mean_6h"] = df["pm2_5"].shift(1).rolling(6).mean()
    df["pm2_5_roll_mean_24h"] = df["pm2_5"].shift(1).rolling(24).mean()
    df["pm2_5_roll_std_24h"] = df["pm2_5"].shift(1).rolling(24).std()

    df["us_aqi_roll_mean_24h"] = df["us_aqi"].shift(1).rolling(24).mean()
    df["us_aqi_roll_std_24h"] = df["us_aqi"].shift(1).rolling(24).std()
    df["us_aqi_roll_min_24h"] = df["us_aqi"].shift(1).rolling(24).min()
    df["us_aqi_roll_max_24h"] = df["us_aqi"].shift(1).rolling(24).max()

    df["pm25_pm10_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-5)

    required_columns = FINAL_FEATURES + ["time"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for inference: {missing}")

    df = df.dropna(subset=FINAL_FEATURES).reset_index(drop=True)
    if df.empty:
        raise ValueError("Insufficient data to create model features.")

    return df