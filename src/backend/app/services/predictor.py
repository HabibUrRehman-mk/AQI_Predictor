from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.model_registry import ModelRegistryService


class Predictor:
    def __init__(self, registry_service: ModelRegistryService | None = None) -> None:
        self.registry = registry_service or ModelRegistryService()
        self.model_names = list(self.registry.model_names)
        self.feature_names: list[str] = []
        self.ready = False

    @staticmethod
    def _get_feature_names(model: Any) -> list[str]:
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None:
            feature_names = getattr(model, "feature_names_", None)
        if feature_names is None:
            raise RuntimeError(
                "The saved model does not include feature names. Retrain the model using a DataFrame."
            )
        return list(feature_names)

    def _load_models(self) -> dict[str, Any]:
        loaded_models = self.registry.load_all_models()
        if not loaded_models:
            self.ready = False
            raise RuntimeError("No models were loaded into memory from the model registry.")

        first_model = next(iter(loaded_models.values()))
        self.feature_names = self._get_feature_names(first_model)
        self.ready = True
        return loaded_models

    def reload_models(self) -> dict[str, Any]:
        loaded_models = self.registry.refresh_models()
        if not loaded_models:
            self.ready = False
            raise RuntimeError("No models were loaded into memory from the model registry.")
        first_model = next(iter(loaded_models.values()))
        self.feature_names = self._get_feature_names(first_model)
        self.ready = True
        return loaded_models

    def predict(self, feature_df: pd.DataFrame) -> list[float]:
        if not self.ready or not self.registry.loaded_models:
            self._load_models()

        missing = [feature for feature in self.feature_names if feature not in feature_df.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")

        model_input = feature_df[self.feature_names].astype(float)
        predictions: list[float] = []

        for model_name in self.model_names:
            model = self.registry.get_model(model_name)
            if model is None:
                raise RuntimeError(f"Model '{model_name}' is not loaded in memory.")
            pred_value = float(np.asarray(model.predict(model_input)).reshape(-1)[0])
            predictions.append(max(0.0, round(pred_value, 2)))

        return predictions