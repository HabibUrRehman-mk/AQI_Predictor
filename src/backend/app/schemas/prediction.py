from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    city: str
    latitude: float
    longitude: float
    timestamp: datetime
    current_aqi: float = Field(ge=0)
    weather: dict[str, Any]
    predicted_aqi_24h: float = Field(ge=0)
    predicted_aqi_48h: float = Field(ge=0)
    predicted_aqi_72h: float = Field(ge=0)