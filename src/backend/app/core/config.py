from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_dir: str = "src/backend/models"
    request_timeout: float = 20.0
    allowed_origins: str = "http://localhost:8080"
    requests_per_minute: int = 30

    class Config:
        env_file = ".env"


settings = Settings()