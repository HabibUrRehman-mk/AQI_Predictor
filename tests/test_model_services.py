from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd

from src.backend.app.services.model_registry import ModelRegistryService, extract_model_object
from src.backend.app.services.predictor import Predictor


class FakeModel:
    def __init__(self, value, feature_names=None):
        self.value = value
        self.feature_names_in_ = np.array(feature_names or ["feature"])

    def predict(self, data):
        return np.array([self.value] * len(data))


def test_extract_model_object_accepts_direct_and_wrapped_models():
    model = FakeModel(10)

    assert extract_model_object(model, "test") is model
    assert extract_model_object(["metadata", model], "test") is model


def test_pick_best_version_uses_lowest_mae():
    candidates = [
        SimpleNamespace(version=1, metrics={"mae": 20.0}),
        SimpleNamespace(version=2, metrics={"mae": 12.5}),
    ]

    selected = ModelRegistryService._pick_best_version(candidates)

    assert selected.version == 2


def test_predictor_clamps_negative_predictions_and_keeps_model_order():
    models = {
        "aqi_predictor_24h": FakeModel(-3),
        "aqi_predictor_48h": FakeModel(42),
        "aqi_predictor_72h": FakeModel(510),
    }

    class FakeRegistry:
        model_names = list(models)
        loaded_models = models

        def get_models_snapshot(self):
            return dict(models)

    predictor = Predictor(cast(ModelRegistryService, FakeRegistry()))
    predictor.ready = True
    predictor.feature_names = ["feature"]

    predictions = predictor.predict(pd.DataFrame({"feature": [1.0]}))

    assert predictions == [0.0, 42.0, 510.0]
