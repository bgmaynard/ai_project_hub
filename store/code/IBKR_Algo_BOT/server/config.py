from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Use env="..." for pydantic-settings v2
    tws_host: str = Field("127.0.0.1", env="TWS_HOST")
    tws_port: int = Field(7497, env="TWS_PORT")        # default PAPER; script sets 7496 for LIVE
    tws_client_id: int = Field(6001, env="TWS_CLIENT_ID")

    api_key: str = Field("", env="API_KEY")

    bind_host: str = Field("127.0.0.1", env="BIND_HOST")
    bind_port: int = Field(9101, env="BIND_PORT")

    # Load .env, be case-insensitive, and ignore unknown envs/keys
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()
