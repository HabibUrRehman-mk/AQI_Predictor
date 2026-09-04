from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import hopsworks
import joblib

from app.core.config import settings


def extract_model_object(loaded_obj: Any, model_name: str) -> Any:
    """Return the estimator object even if stored inside a tuple/list."""
    if hasattr(loaded_obj, "predict"):
        return loaded_obj

    if isinstance(loaded_obj, (tuple, list)):
        for item in loaded_obj:
            if hasattr(item, "predict"):
                return item

    raise TypeError(
        f"Loaded artifact for '{model_name}' (type: {type(loaded_obj).__name__}) "
        "does not contain an object with a .predict() method."
    )


class ModelRegistryService:
    def __init__(self) -> None:
        self.model_dir = Path(settings.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_names = list(settings.model_names)
        self._active_models: dict[str, Any] = {}
        self._previous_models: dict[str, Any] = {}
        self._lock = threading.RLock()

    @property
    def loaded_models(self) -> dict[str, Any]:
        return self._active_models

    def _get_project(self) -> Any:
        if not settings.hopsworks_api_key.get_secret_value():
            raise RuntimeError(
                "HOPSWORKS_API_KEY is missing. Add it to your .env file before downloading models."
            )

        return hopsworks.login(
            host=settings.hopsworks_host,
            api_key_value=settings.hopsworks_api_key.get_secret_value(),
            project=settings.hopsworks_project,
        )

    @staticmethod
    def _metric_value(model: Any, metric_name: str) -> float | None:
        # Hopsworks often stores metrics in a dict-like object on the registry model entry.
        candidates: list[Any] = []
        for attr in ("metrics", "training_metrics"):
            value = getattr(model, attr, None)
            if value is not None:
                candidates.append(value)

        for candidate in candidates:
            if isinstance(candidate, dict):
                if metric_name in candidate:
                    value = candidate[metric_name]
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
                for nested in candidate.values():
                    if isinstance(nested, dict) and metric_name in nested:
                        try:
                            return float(nested[metric_name])
                        except (TypeError, ValueError):
                            return None
            elif hasattr(candidate, metric_name):
                try:
                    return float(getattr(candidate, metric_name))
                except (TypeError, ValueError):
                    return None

        return None

    @classmethod
    def _pick_best_version(cls, models_list: list[Any]) -> Any:
        if not models_list:
            raise ValueError("No model versions found in the registry.")

        best_model = None
        best_mae: float | None = None

        for model in models_list:
            mae = cls._metric_value(model, "mae")
            if mae is None:
                continue
            if best_mae is None or mae < best_mae:
                best_mae = mae
                best_model = model

        if best_model is not None:
            return best_model

        return max(models_list, key=lambda item: int(getattr(item, "version", 0)))

    def _persist_model_if_possible(self, model_name: str, model_object: Any) -> None:
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.model_dir / model_name / "model.pkl"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_object, cache_path)
        except (OSError, PermissionError):
            # The standard notebook flow does not require a persistent local cache.
            # Models are kept in RAM after Hopsworks download, which matches the working deployment path.
            pass

    def _download_model_version(self, model_name: str) -> tuple[str, Any, str]:
        project = self._get_project()
        registry = project.get_model_registry()
        model_versions = registry.get_models(name=model_name)

        if not model_versions:
            raise ValueError(f"No models registered under name '{model_name}'")

        selected_model = self._pick_best_version(model_versions)
        version_num = str(selected_model.version)
        download_path = selected_model.download()

        model_filepath = None
        for dirpath, _, filenames in os.walk(download_path):
            if "model.pkl" in filenames:
                model_filepath = os.path.join(dirpath, "model.pkl")
                break

        if not model_filepath or not os.path.exists(model_filepath):
            raise FileNotFoundError(
                f"Could not locate 'model.pkl' for model '{model_name}' in '{download_path}'"
            )

        raw_artifact = joblib.load(model_filepath)
        model_object = extract_model_object(raw_artifact, model_name)
        self._persist_model_if_possible(model_name, model_object)

        return model_name, model_object, version_num

    def load_from_cache(self, model_name: str) -> Any:
        cache_path = self.model_dir / model_name / "model.pkl"
        if not cache_path.exists():
            raise FileNotFoundError(f"No cached model found for '{model_name}' at '{cache_path}'")

        raw_artifact = joblib.load(cache_path)
        object_model = extract_model_object(raw_artifact, model_name)
        return object_model

    def download_model(self, model_name: str) -> Any:
        _, model_object, _ = self._download_model_version(model_name)
        with self._lock:
            self._active_models[model_name] = model_object
        return model_object

    def load_all_models(self) -> dict[str, Any]:
        """Download the best registered version of every model into RAM."""
        loaded_models: dict[str, Any] = {}

        for model_name in self.model_names:
            _, model_object, _ = self._download_model_version(model_name)
            loaded_models[model_name] = model_object

        with self._lock:
            self._previous_models = self._active_models
            self._active_models = loaded_models
            self._previous_models.clear()

        return self._active_models

    def refresh_models(self) -> dict[str, Any]:
        """Download the best registered version of every model and load it into RAM."""
        new_models: dict[str, Any] = {}

        for model_name in self.model_names:
            _, model_object, _ = self._download_model_version(model_name)
            new_models[model_name] = model_object

        with self._lock:
            self._previous_models = self._active_models
            self._active_models = new_models
            self._previous_models.clear()

        return self._active_models

    def get_model(self, model_name: str) -> Any | None:
        with self._lock:
            return self._active_models.get(model_name)

    def get_models_snapshot(self) -> dict[str, Any]:
        """Return one stable active-model snapshot for a complete prediction."""
        with self._lock:
            return dict(self._active_models)

    def clear_cached_models(self) -> None:
        with self._lock:
            self._active_models.clear()
            self._previous_models.clear()

    def clear_cache(self) -> None:
        self.clear_cached_models()
