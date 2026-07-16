
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    data_source_path: str = "data/raw/songs_lyrics.csv"
    sentiment_model_name: str = "distilbert-sentiment"
    recommender_model_type: str = "lightgbm"
    random_seed: int = 42
    log_level: str = "INFO"
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-2.5-flash"

    model_config = SettingsConfigDict(env_prefix="PLAYCATCH_", env_file=".env")


settings = Settings()