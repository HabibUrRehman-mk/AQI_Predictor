import os
import hopsworks

HORIZONS = ["24h", "48h", "72h"]
PROMOTION_METRIC = "mae"

def connect_to_hopsworks():
    """Establish connection to the Hopsworks Model Registry."""
    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ.get("HOPSWORKS_PROJECT"),
    )
    return project.get_model_registry()

def promote_best_models_safely():
    registry = connect_to_hopsworks()

    for horizon in HORIZONS:
        model_name = f"aqi_predictor_{horizon}"
        print(f"\n--- Processing model tags for: {model_name} ---")

        try:
            models = registry.get_models(model_name)
        except Exception as e:
            print(f"No models found for {model_name}: {e}")
            continue

        if not models:
            continue

        # Extract comparable versions containing the primary metric
        comparable_models = []
        for model in models:
            metric_val = (model.training_metrics or {}).get(PROMOTION_METRIC)
            if metric_val is not None:
                comparable_models.append((model, metric_val))

        if not comparable_models:
            print(f"No versions of {model_name} contain the '{PROMOTION_METRIC}' metric.")
            continue

        # Identify the best performing version (lowest MAE)
        best_model, best_mae = min(comparable_models, key=lambda pair: pair[1])

        # Try adding tag; fallback to description update if tag schema doesn't exist
        for model, mae_val in comparable_models:
            is_best = (model.version == best_model.version)
            target_status = "production" if is_best else "candidate"

            try:
                # Attempts schema-backed tagging (works if Option 1 was performed)
                model.add_tag(name="status", value=target_status)
                print(f" -> Version {model.version}: Tagged as [{target_status}] (MAE: {mae_val:.2f})")
            except Exception:
                # Fallback: Updates model description metadata cleanly
                label = "[PRODUCTION]" if is_best else "[CANDIDATE]"
                clean_desc = (model.description or "").replace("[PRODUCTION]", "").replace("[CANDIDATE]", "").strip()
                model.description = f"{label} {clean_desc}".strip()
                
                print(f" -> Version {model.version}: Updated status to {label} in metadata description (MAE: {mae_val:.2f})")

        print(f"SUCCESS: Version {best_model.version} identified as BEST MODEL for horizon {horizon}.")

if __name__ == "__main__":
    promote_best_models_safely()