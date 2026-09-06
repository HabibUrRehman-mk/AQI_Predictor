from datetime import datetime

import numpy as np
import pandas as pd

from src.backend.app.services.feature_engineering import engineer_features_for_inference


def make_raw_frame(rows: int = 100) -> pd.DataFrame:
    times = pd.date_range(datetime(2026, 1, 1), periods=rows, freq="h")
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "time": times,
            "wind_direction_10m": np.full(rows, 90.0),
            "wind_speed_10m": np.full(rows, 10.0),
            "temperature_2m": 10 + index * 0.1,
            "relative_humidity_2m": np.full(rows, 60.0),
            "surface_pressure": np.full(rows, 1000.0),
            "pm2_5": 20 + index,
            "pm10": 40 + index,
            "us_aqi": 50 + index,
        }
    )


def test_inference_features_are_complete_and_backward_looking():
    raw = make_raw_frame()
    features = engineer_features_for_inference(raw)

    assert not features.empty
    assert features["time"].is_monotonic_increasing
    assert features.iloc[-1]["us_aqi_lag_1h"] == raw.iloc[-2]["us_aqi"]
    assert features.iloc[-1]["us_aqi_roll_mean_24h"] == raw.iloc[-25:-1]["us_aqi"].mean()
    assert np.isclose(features.iloc[-1]["wind_u"], -10.0)
    assert np.isclose(features.iloc[-1]["wind_v"], 0.0, atol=1e-10)


def test_inference_requires_enough_history_for_lags():
    raw = make_raw_frame(rows=20)

    try:
        engineer_features_for_inference(raw)
    except ValueError as error:
        assert "Insufficient data" in str(error)
    else:
        raise AssertionError("Expected short input to fail due to missing lag history")
