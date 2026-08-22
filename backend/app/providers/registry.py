"""The validator dropdown.

Each entry describes one model the auditing agent can run on. Adding a new
Google model is a single dict entry -- that is deliberate, since breadth of
Google model support is worth bonus points.

`adk_model` is what gets handed to ADK's LlmAgent:
  * a bare string  -> native Gemini path (AI Studio or Vertex AI)
  * "litellm:<id>" -> wrapped in LiteLlm, which covers Ollama and Vertex-hosted
                      open models such as Gemma and MedGemma.
"""
from __future__ import annotations

from pydantic import BaseModel

from ..config import settings


class ValidatorSpec(BaseModel):
    key: str
    label: str
    provider: str            # "ollama" | "vertex-ai" | "ai-studio"
    adk_model: str
    cost: str                # "free" | "paid"
    blurb: str
    #: What the entry needs before it can run:
    #:   "none"    - works out of the box (local Ollama)
    #:   "gcp"     - a Google Cloud project and application-default credentials
    #:   "api_key" - a Google AI Studio key, which is all a desktop user needs
    requires: str = "none"
    recommended_for: str = "general"

    @property
    def requires_gcp(self) -> bool:
        return self.requires == "gcp"


VALIDATORS: dict[str, ValidatorSpec] = {
    # ---- Local, $0, no credits burned. The demo default. ----
    "local-gemma": ValidatorSpec(
        key="local-gemma",
        label="Local Gemma 3 (Ollama)",
        provider="ollama",
        adk_model="litellm:ollama_chat/gemma3:12b",
        cost="free",
        blurb="Runs on this machine. No cloud credits, works offline.",
    ),
    # ---- Google Cloud validators. Each distinct model is a bonus point. ----
    "gemini-flash": ValidatorSpec(
        key="gemini-flash",
        label="Gemini 3 Flash (Vertex AI)",
        provider="vertex-ai",
        adk_model="gemini-flash-latest",
        cost="paid",
        blurb="Fast, strong reasoning. Best general-purpose judge.",
        requires="gcp",
    ),
    "gemini-pro": ValidatorSpec(
        key="gemini-pro",
        label="Gemini 3 Pro (Vertex AI)",
        provider="vertex-ai",
        adk_model="gemini-pro-latest",
        cost="paid",
        blurb="Deepest reasoning. Use for subtle correctness disputes.",
        requires="gcp",
    ),
    "gemma-vertex": ValidatorSpec(
        key="gemma-vertex",
        label="Gemma 3 27B (Vertex AI)",
        provider="vertex-ai",
        adk_model="litellm:vertex_ai/gemma-3-27b-it",
        cost="paid",
        blurb="Open-weights judge, hosted. Same family as the local option.",
        requires="gcp",
    ),
    "medgemma": ValidatorSpec(
        key="medgemma",
        label="MedGemma 27B (Vertex AI)",
        provider="vertex-ai",
        adk_model="litellm:vertex_ai/medgemma-27b-text-it",
        cost="paid",
        blurb="Clinically tuned. Use when auditing a medical model.",
        requires="gcp",
        recommended_for="medical",
    ),
    # ---- Google AI Studio. One API key, no gcloud, no project setup. This is
    # ---- the path the desktop app steers people to.
    "gemini-flash-key": ValidatorSpec(
        key="gemini-flash-key",
        label="Gemini 3 Flash (API key)",
        provider="ai-studio",
        adk_model="gemini-flash-latest",
        cost="paid",
        blurb="Fast and strong. Needs a free Google AI Studio key.",
        requires="api_key",
    ),
    "gemini-pro-key": ValidatorSpec(
        key="gemini-pro-key",
        label="Gemini 3 Pro (API key)",
        provider="ai-studio",
        adk_model="gemini-pro-latest",
        cost="paid",
        blurb="Deepest reasoning. Use for subtle correctness disputes.",
        requires="api_key",
    ),
}

DEFAULT_VALIDATOR = "local-gemma"


def get_validator(key: str) -> ValidatorSpec:
    if key not in VALIDATORS:
        raise KeyError(
            f"Unknown validator {key!r}. Options: {', '.join(VALIDATORS)}"
        )
    return VALIDATORS[key]


def _availability(spec: ValidatorSpec) -> tuple[bool, str | None]:
    if spec.requires == "gcp":
        if settings.vertex_configured:
            return True, None
        return False, "Set GOOGLE_CLOUD_PROJECT to enable Vertex AI validators."
    if spec.requires == "api_key":
        if settings.google_api_key:
            return True, None
        return False, "Add a Google AI Studio API key in Settings."
    return True, None


def list_validators() -> list[dict]:
    """Dropdown payload. `available` drives whether the option is selectable."""
    out = []
    for spec in VALIDATORS.values():
        available, reason = _availability(spec)
        payload = spec.model_dump()
        payload["requires_gcp"] = spec.requires_gcp
        out.append(
            {
                **payload,
                "available": available,
                "unavailable_reason": reason,
            }
        )
    return out
