import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import hopsworks
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

TARGET_COLS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
METADATA_COLS = ["time", "city"]
GAP = 72
TRAIN_FRAC = 0.8
VAL_FRAC = 0.9

FINAL_FEATURES = [
    "us_aqi", "us_aqi_lag_1h", "pm25_pm10_ratio",
    "pm2_5", "pm2_5_lag_1h", "pm2_5_roll_mean_6h", "pm2_5_roll_mean_24h",
    "pm2_5_roll_std_24h", "pm10",
    "us_aqi_lag_24h", "us_aqi_lag_48h", "us_aqi_lag_72h",
    "us_aqi_roll_mean_24h", "us_aqi_roll_min_24h", "us_aqi_roll_max_24h", "us_aqi_roll_std_24h",
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_u", "wind_v",
    "is_burning_season",
    "cos_month", "sin_month", "cos_hour", "sin_hour", "is_weekend",
]

# Hyperparameters found via RandomizedSearchCV in the notebook.
# Refresh these periodically (e.g. monthly) by re-running the tuning
# section of the notebook and pasting the new best_params_ in here -
# routine retrains should not re-run a 40-iteration search every time.
CATBOOST_PARAMS = {
    "24h": {"iterations": 800, "depth": 6, "learning_rate": 0.01, "l2_leaf_reg": 9, "subsample": 0.6, "border_count": 32},
    "48h": {"iterations": 300, "depth": 3, "learning_rate": 0.01, "l2_leaf_reg": 1, "subsample": 0.8, "border_count": 128},
    "72h": {"iterations": 300, "depth": 3, "learning_rate": 0.01, "l2_leaf_reg": 1, "subsample": 0.8, "border_count": 128},
}

# Primary metric used for the champion/challenger comparison.
# Lower is better for MAE - a margin avoids promoting on noise alone.
PROMOTION_METRIC = "mae"
PROMOTION_MARGIN = 0.0  # e.g. 0.02 would require a 2% improvement to promote


def get_data():
    """Connect to Hopsworks and pull the latest feature group as a sorted dataframe.

    Uses the offline store, not the online store - the online store is a
    low-latency MySQL table meant for small serving-time lookups, and reading
    the full ~17k row training set through it risks lock contention with
    whatever ingestion job is writing new rows at the same time (this is what
    caused the "Lock wait timeout exceeded" error). The offline store is the
    correct source for full historical reads like this.
    """
    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ.get("HOPSWORKS_PROJECT"),
    )
    fs = project.get_feature_store(name="aqi_predictor_fsd_featurestore")
    fg = fs.get_feature_group("weather_aqi_hourly", version=6)

    last_error = None
    for attempt in range(3):
        try:
            data = fg.read()  # offline store (default) - not online=True
            break
        except Exception as e:
            last_error = e
            wait = 30 * (attempt + 1)
            print(f"read attempt {attempt + 1} failed ({e}), retrying in {wait}s")
            time.sleep(wait)
    else:
        raise last_error

    data["time"] = pd.to_datetime(data["time"])
    data = data.sort_values("time").reset_index(drop=True)
    return project, data


def clean_data(data):
    """Drop rows with missing targets and duplicates."""
    data = data.dropna(subset=TARGET_COLS).reset_index(drop=True)
    data = data.drop_duplicates().reset_index(drop=True)
    return data


def split_data(data):
    """Time-ordered train/test split with a leakage gap, same as the notebook."""
    feature_cols = [c for c in data.columns if c not in TARGET_COLS + METADATA_COLS]
    x = data[feature_cols]
    y = data[TARGET_COLS]
    n = len(data)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * VAL_FRAC)

    train_x = x.iloc[:train_end][FINAL_FEATURES]
    train_y = y.iloc[:train_end]
    test_x = x.iloc[val_end + GAP:][FINAL_FEATURES]
    test_y = y.iloc[val_end + GAP:]
    return train_x, train_y, test_x, test_y


def score(y_true, y_pred):
    """Standard metric set, same as the notebook's score_predictions."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def train_horizon(train_x, train_y, test_x, test_y, horizon):
    """Fit a CatBoost model for one horizon using the stored tuned config."""
    model = CatBoostRegressor(**CATBOOST_PARAMS[horizon], random_state=42, verbose=False)
    model.fit(train_x, train_y[f"target_aqi_{horizon}"])
    pred = model.predict(test_x)
    metrics = score(test_y[f"target_aqi_{horizon}"], pred)
    return model, metrics


def get_current_champion(project, model_name):
    """Find the best already-registered version by scanning stored metrics directly.

    Hopsworks tags need a project-level schema to be predefined before they can
    be attached (a one-time UI step), so rather than depend on that, the current
    "production" model is just whichever registered version has the best MAE -
    computed fresh each run from data already stored on every version.

    Older manually-registered versions may have used different metric key
    names (e.g. "Mean MAE" instead of "mae") - those are skipped rather than
    treated as a fatal error, since they simply aren't comparable.
    """
    registry = project.get_model_registry()
    try:
        versions = registry.get_models(model_name)
    except Exception:
        return None, None

    comparable = []
    for m in versions:
        value = (m.training_metrics or {}).get(PROMOTION_METRIC)
        if value is not None:
            comparable.append((m, value))

    if not comparable:
        return None, None

    best_model, _ = min(comparable, key=lambda pair: pair[1])
    return best_model, best_model.training_metrics


def should_promote(new_metrics, current_metrics):
    """Champion/challenger decision - lower MAE wins, with an optional margin to avoid flapping on noise."""
    if current_metrics is None:
        return True  # nothing in production yet - first model always becomes champion
    current = current_metrics[PROMOTION_METRIC]
    new = new_metrics[PROMOTION_METRIC]
    return new <= current * (1 - PROMOTION_MARGIN)


def register_model(project, model, name, metrics, x_sample, y_sample, description, promote):
    """Register a model version unconditionally. The description records whether it won
    the comparison, purely for readability in the Hopsworks UI - the actual "which
    version is production" decision is recomputed from stored metrics each run by
    get_current_champion, not read back from this label.
    """
    model_dir = f"artifacts_{name}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))

    input_schema = Schema(x_sample)
    output_schema = Schema(y_sample)
    schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    label = "[PRODUCTION]" if promote else "[CANDIDATE]"
    full_description = f"{label} {description}"

    registry = project.get_model_registry()
    entry = registry.python.create_model(
        name=name,
        metrics=metrics,
        description=full_description,
        model_schema=schema,
        input_example=x_sample,
    )
    entry.save(model_dir, keep_original_files=True)

    if promote:
        print(f"{name}: version {entry.version} is the new best model (mae={metrics['mae']:.2f})")
    else:
        print(f"{name}: registered version {entry.version} as a candidate, did not beat the current best (mae={metrics['mae']:.2f})")

    return entry


def main():
    project, df = get_data()
    df = clean_data(df)
    train_x, train_y, test_x, test_y = split_data(df)

    summary = {}
    for horizon in ["24h", "48h", "72h"]:
        model, metrics = train_horizon(train_x, train_y, test_x, test_y, horizon)

        model_name = f"aqi_predictor_{horizon}"
        _, current_metrics = get_current_champion(project, model_name)
        promote = should_promote(metrics, current_metrics)

        register_model(
            project, model, model_name, metrics,
            x_sample=train_x.head(1),
            y_sample=train_y[[f"target_aqi_{horizon}"]].head(1),
            description=f"{horizon} ahead AQI prediction model, retrained {pd.Timestamp.utcnow()}",
            promote=promote,
        )

        summary[horizon] = {"new_mae": metrics["mae"], "new_r2": metrics["r2"],
                             "previous_production_mae": current_metrics["mae"] if current_metrics else None,
                             "promoted": promote}

    print(json.dumps(summary, indent=2))

    if not any(v["promoted"] for v in summary.values()):
        print("no horizon improved on the current production model")
        sys.exit(0)  


if __name__ == "__main__":
    main()