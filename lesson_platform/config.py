"""Load settings from environment."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    anthropic_api_key: str
    anthropic_model: str
    flask_secret_key: str
    port: int


def load_settings() -> Settings:
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        flask_secret_key=os.getenv("FLASK_SECRET_KEY", "dev-secret-key"),
        port=int(os.getenv("PORT", "5000")),
    )
