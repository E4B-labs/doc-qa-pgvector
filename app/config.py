from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/docqa"
    embedding_provider: str = "openai"
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    llm_provider: str = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-latest"
    max_upload_bytes: int = 5 * 1024 * 1024
    llm_timeout_seconds: float = 20.0
    similarity_threshold: float = 0.35
    embedding_dimensions: int = 1536
    rrf_k: int = 60

