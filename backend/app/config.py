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
    # "global", not a region: Gemini 3.x is only served from the global
    # endpoint. Regional endpoints list the models but 404 on generate.
    google_cloud_location: str = "global"
    # Model Garden models (Gemma, MedGemma) are served regionally, unlike
    # Gemini 3.x which is global-only.
    model_garden_location: str = "us-central1"
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

    # Gemini 3.x consumes part of the output budget on reasoning tokens, so
    # these are well above what the visible text needs. A probe set is the
    # longest single reply either agent produces.
    generator_max_tokens: int = 16384
    judge_max_tokens: int = 8192

    # --- Autonomy ---
    # After the first pass, the agent decides for itself whether any dimension
    # is weak enough to be worth investigating, and writes more probes aimed
    # at it. Nobody asks it to.
    adaptive_probing: bool = True
    # Mean score (out of 5) below which a dimension is judged weak.
    adaptive_threshold: float = 3.0
    # Follow-up rounds allowed, so a bad model cannot loop forever.
    adaptive_max_rounds: int = 2
    adaptive_probes_per_round: int = 3

    # A run is a regression if it drops this many points against the last run
    # of the same model.
    regression_drop: float = 10.0

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
