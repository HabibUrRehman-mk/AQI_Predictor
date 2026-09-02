from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AQI Predictor API"
    app_version: str = "1.1.0"
    model_dir: str = "/tmp/aqi_model_cache"
    request_timeout: float = 30.0
    requests_per_minute: int = 30
    allowed_origins: str = "http://localhost:8080,http://127.0.0.1:8080"


    AQ_URL: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    WEATHER_URL: str = "https://api.open-meteo.com/v1/forecast"
    hopsworks_host: str = "eu-west.cloud.hopsworks.ai"
    hopsworks_project: str = "AQI_Predictor_fsd"
    hopsworks_api_key: SecretStr
    model_names: list[str] = [
        "aqi_predictor_24h",
        "aqi_predictor_48h",
        "aqi_predictor_72h",
    ]

    FINAL_FEATURES: list[str] = [
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


settings = Settings()