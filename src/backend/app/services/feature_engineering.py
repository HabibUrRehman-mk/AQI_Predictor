import numpy as np
import pandas as pd


REQUIRED_FEATURES = [
    "us_aqi",
    "us_aqi_lag_1h",
    "pm25_pm10_ratio",
    "pm2_5",
    "pm2_5_lag_1h",
    "pm2_5_roll_mean_6h",
    "pm2_5_roll_mean_24h",
    "pm2_5_roll_std_24h",
    "pm10",
    "us_aqi_lag_24h",
    "us_aqi_lag_48h",
    "us_aqi_lag_72h",
    "us_aqi_roll_mean_24h",
    "us_aqi_roll_min_24h",
    "us_aqi_roll_max_24h",
    "us_aqi_roll_std_24h",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_u",
    "wind_v",
    "is_burning_season",
    "cos_month",
    "sin_month",
    "cos_hour",
    "sin_hour",
    "is_weekend",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["time"] = pd.to_datetime(result["time"], utc=True)
    result = (
        result.sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    wind_radians = np.radians(result["wind_direction_10m"])
    result["wind_u"] = (
        -result["wind_speed_10m"] * np.sin(wind_radians)
    )
    result["wind_v"] = (
        -result["wind_speed_10m"] * np.cos(wind_radians)
    )

    result["hour"] = result["time"].dt.hour
    result["dayofweek"] = result["time"].dt.dayofweek
    result["month"] = result["time"].dt.month

    result["sin_hour"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["cos_hour"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["sin_month"] = np.sin(2 * np.pi * result["month"] / 12)
    result["cos_month"] = np.cos(2 * np.pi * result["month"] / 12)
    result["is_weekend"] = (result["dayofweek"] >= 5).astype(int)

    result["is_burning_season"] = result["month"].isin([10, 11]).astype(int)
    result["pm25_pm10_ratio"] = (
        result["pm2_5"] / (result["pm10"] + 1e-5)
    )

    result["pm2_5_lag_1h"] = result["pm2_5"].shift(1)
    result["pm2_5_lag_24h"] = result["pm2_5"].shift(24)

    for hours in [1, 24, 48, 72]:
        result[f"us_aqi_lag_{hours}h"] = result["us_aqi"].shift(hours)

    pm25_previous = result["pm2_5"].shift(1)
    aqi_previous = result["us_aqi"].shift(1)

    result["pm2_5_roll_mean_6h"] = (
        pm25_previous.rolling(6).mean()
    )
    result["pm2_5_roll_mean_24h"] = (
        pm25_previous.rolling(24).mean()
    )
    result["pm2_5_roll_std_24h"] = (
        pm25_previous.rolling(24).std()
    )

    result["us_aqi_roll_mean_24h"] = (
        aqi_previous.rolling(24).mean()
    )
    result["us_aqi_roll_min_24h"] = (
        aqi_previous.rolling(24).min()
    )
    result["us_aqi_roll_max_24h"] = (
        aqi_previous.rolling(24).max()
    )
    result["us_aqi_roll_std_24h"] = (
        aqi_previous.rolling(24).std()
    )

    result = result.dropna().reset_index(drop=True)

    if result.empty:
        raise ValueError("Insufficient API data to create model features")

    missing = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in result.columns
    ]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return result