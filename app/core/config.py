from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    ANTHROPIC_API_KEY: str
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "claude-sonnet-4-6"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 4

    model_config = {"env_file": ".env"}

    @property
    def DATA_DIR(self) -> Path:
        return BASE_DIR / "data"

    @property
    def INDEX_PATH(self) -> Path:
        return self.DATA_DIR / "index.faiss"

    @property
    def METADATA_PATH(self) -> Path:
        return self.DATA_DIR / "chunks.json"


settings = Settings()
settings.DATA_DIR.mkdir(exist_ok=True)
