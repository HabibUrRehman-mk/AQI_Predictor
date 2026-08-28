from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings


class Predictor:
    def __init__(self) -> None:
        model_dir = Path(settings.model_dir)

        combined_path = model_dir / "aqi_model.joblib"
        separate_paths = [
            model_dir / "model_24h.joblib",
            model_dir / "model_48h.joblib",
            model_dir / "model_72h.joblib",
        ]

        self.model = None
        self.models = None

        if combined_path.exists():
            self.model = joblib.load(combined_path)
            self.feature_names = self._get_feature_names(self.model)
            return

        if all(path.exists() for path in separate_paths):
            self.models = [
                joblib.load(path)
                for path in separate_paths
            ]
            self.feature_names = self._get_feature_names(
                self.models[0]
            )
            return

        raise FileNotFoundError(
            "Models not found. Add either aqi_model.joblib or "
            "model_24h.joblib, model_48h.joblib, model_72h.joblib."
        )

    @staticmethod
    def _get_feature_names(model) -> list[str]:
        feature_names = getattr(model, "feature_names_in_", None)

        if feature_names is None:
            raise RuntimeError(
                "The saved model does not contain feature names. "
                "Retrain the model using a Pandas DataFrame."
            )

        return list(feature_names)

    @property
    def ready(self) -> bool:
        return self.model is not None or self.models is not None

    def predict(self, feature_df: pd.DataFrame) -> list[float]:
        missing = [
            feature
            for feature in self.feature_names
            if feature not in feature_df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing model features: {missing}"
            )

        model_input = feature_df[
            self.feature_names
        ].astype(float)

        if self.model is not None:
            predictions = np.asarray(
                self.model.predict(model_input)
            ).reshape(-1)

        else:
            predictions = np.asarray([
                model.predict(model_input)[0]
                for model in self.models
            ])

        return [
            max(0.0, round(float(value), 2))
            for value in predictions
        ]