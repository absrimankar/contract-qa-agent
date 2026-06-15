import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "claude-sonnet-4-6"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K = 4

    @property
    def ANTHROPIC_API_KEY(self) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to .env (local) or Streamlit App Secrets (cloud)."
            )
        return key

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
