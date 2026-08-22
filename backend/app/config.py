"""Runtime configuration. Everything is env-overridable so the same image runs
locally (Ollama, in-memory store) and on Cloud Run (Vertex AI, Firestore).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Google AI Studio ---
    # A plain API key, which is all a desktop user needs -- no gcloud, no
    # project, no application-default credentials.
    google_api_key: str = ""

    # --- Google Cloud ---
    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"
    # ADK reads this to route Gemini calls at Vertex AI instead of AI Studio.
    google_genai_use_vertexai: bool = True

    # --- Storage ---
    # "firestore" | "memory". Falls back to memory if Firestore is unreachable.
    store_backend: str = "memory"
    firestore_collection: str = "auditor_runs"

    # Public Cloud Run demo mode: audit HTTPS model endpoints only. This keeps
    # the free-tier web version away from local filesystem paths and uploads.
    auditor_cloud_demo: bool = False

    # --- Model under test ---
    ollama_host: str = "http://localhost:11434"
    # How long Ollama holds a model in memory after a request. Long enough that
    # the judge and the model under test both stay resident across a whole run.
    ollama_keep_alive: str = "20m"

    # --- Run defaults ---
    default_num_probes: int = 8
    # A model that has not answered in 90s has failed the probe. Waiting longer
    # only makes a stalled run look broken.
    target_timeout_s: float = 90.0
    judge_timeout_s: float = 120.0
    # Small pause between probes so the live dashboard is readable in a demo.
    probe_delay_s: float = 0.0

    @property
    def vertex_configured(self) -> bool:
        return bool(self.google_cloud_project)

    @property
    def api_key_configured(self) -> bool:
        return bool(self.google_api_key)


settings = Settings()
