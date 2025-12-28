import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def _pick_env_file() -> str:
    # ENV selects: dev / staging / prod
    env = os.getenv("ENV", "dev")
    return f".env.{env}"


class Settings(BaseSettings):
    env: str = "dev"
    log_level: str = "INFO"
    port: int = 8080

    kafka_broker: str = "localhost:9092"
    kafka_topic_vitals: str = "vitals"

    metrics_enabled: bool = True

    mlflow_tracking_uri: str | None = None
    gcp_project_id: str | None = None

    model_config = SettingsConfigDict(
        env_file=_pick_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
