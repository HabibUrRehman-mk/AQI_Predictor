from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.services.predictor import Predictor

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AQI forecasting API with registry-backed model loading and RAM-cached inference.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.predictor = Predictor()
app.include_router(router)


@app.on_event("startup")
async def startup_event() -> None:
    try:
        app.state.predictor.reload_models()
        app.state.startup_error = None
    except Exception:
        app.state.startup_error = "Model registry cache is not available yet. Use the admin endpoint to load models."