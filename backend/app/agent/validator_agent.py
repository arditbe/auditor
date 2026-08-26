"""The auditing agent, built on Google ADK.

Two ADK `LlmAgent`s do the work:

  * the **probe generator** invents the test questions
  * the **judge** scores each answer

Both run on whichever validator the user picked, so switching from local Gemma
to Gemini changes one string and nothing else. Agents are cached per validator
key -- rebuilding an LlmAgent per probe is wasteful, and mid-run validator
switches would otherwise thrash.
"""
from __future__ import annotations

import asyncio
import logging

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from ..config import settings
from ..providers.registry import ValidatorSpec, get_validator
from . import prompts

log = logging.getLogger(__name__)

APP_NAME = "auditor"

# One session service for the whole process; ADK sessions are cheap.
_session_service = InMemorySessionService()
_agent_cache: dict[tuple[str, str], LlmAgent] = {}
_runner_cache: dict[tuple[str, str], Runner] = {}


def resolve_adk_model(spec: ValidatorSpec) -> str | LiteLlm:
    """Turn a registry entry into something ADK's `model=` accepts.

    Bare Gemini ids go through ADK's native path (Vertex AI); everything else
    is routed via LiteLlm.

    API-key entries deliberately go through LiteLlm too, with the key passed
    explicitly. ADK's native path picks Vertex-vs-AI-Studio from a process-wide
    environment variable, so a user with both a GCP project and an API key
    could not have one validator use each. Passing the key per-agent keeps the
    two independent.
    """
    if spec.provider == "ai-studio":
        if not settings.google_api_key:
            raise ValueError(
                f"{spec.label} needs a Google AI Studio API key. Add one in "
                "Settings."
            )
        return LiteLlm(
            model=f"gemini/{spec.adk_model}",
            api_key=settings.google_api_key,
        )

    if spec.adk_model.startswith("litellm:"):
        model_id = spec.adk_model.removeprefix("litellm:")
        kwargs: dict[str, object] = {}
        if model_id.startswith("ollama_chat/"):
            kwargs["api_base"] = settings.ollama_host
        elif model_id.startswith("vertex_ai/"):
            # LiteLLM will not infer these from the environment the way the
            # native path does, and fails with "project and location are
            # required" without them. Model Garden models (Gemma, MedGemma)
            # are regional, so "global" is not a valid home for them.
            kwargs["vertex_project"] = settings.google_cloud_project
            kwargs["vertex_location"] = settings.model_garden_location
        return LiteLlm(model=model_id, **kwargs)

    return spec.adk_model


def _build_agent(spec: ValidatorSpec, role: str) -> LlmAgent:
    instruction = (
        prompts.PROBE_GENERATOR_INSTRUCTION
        if role == "generator"
        else prompts.JUDGE_INSTRUCTION
    )
    description = (
        "Designs adversarial test probes for a model under audit."
        if role == "generator"
        else "Scores a model's answer against per-probe grading criteria."
    )
    return LlmAgent(
        name=f"auditor_{role}",
        model=resolve_adk_model(spec),
        description=description,
        instruction=instruction,
        generate_content_config=types.GenerateContentConfig(
            # The judge must be reproducible; a judge that changes its mind
            # between identical inputs is not a measurement instrument.
            temperature=0.1 if role == "judge" else 0.8,
            # Generous on purpose. Gemini 3.x spends part of this budget on
            # internal reasoning before it writes anything, so a limit that
            # looks ample for the visible output truncates it mid-sentence --
            # which is what made the safety suite fail intermittently, since
            # adversarial probes are the longest ones to write.
            max_output_tokens=(
                settings.generator_max_tokens
                if role == "generator"
                else settings.judge_max_tokens
            ),
        ),
    )


def get_runner(validator_key: str, role: str) -> Runner:
    """Cached ADK Runner for (validator, role)."""
    cache_key = (validator_key, role)
    if cache_key not in _runner_cache:
        spec = get_validator(validator_key)
        agent = _build_agent(spec, role)
        _agent_cache[cache_key] = agent
        _runner_cache[cache_key] = Runner(
            app_name=APP_NAME,
            agent=agent,
            session_service=_session_service,
            auto_create_session=True,
        )
        log.info("built ADK agent role=%s validator=%s", role, validator_key)
    return _runner_cache[cache_key]


async def invoke(
    *, validator_key: str, role: str, message: str, session_id: str, timeout: float
) -> str:
    """Run one turn through ADK and return the final text response.

    Each call uses a fresh session id, so probes are graded independently and
    the judge cannot be primed by how it scored the previous answer.
    """
    runner = get_runner(validator_key, role)
    content = types.Content(role="user", parts=[types.Part(text=message)])

    async def _collect() -> str:
        chunks: list[str] = []
        async for event in runner.run_async(
            user_id="auditor",
            session_id=session_id,
            new_message=content,
        ):
            if not event.content or not event.content.parts:
                continue
            # Only the final response carries the answer; intermediate events
            # may hold partial streaming text we would otherwise double-count.
            if event.is_final_response():
                for part in event.content.parts:
                    if part.text:
                        chunks.append(part.text)
        return "".join(chunks).strip()

    return await asyncio.wait_for(_collect(), timeout=timeout)


def reset_cache() -> None:
    """Drop cached agents. Used by tests and after a config change."""
    _agent_cache.clear()
    _runner_cache.clear()
