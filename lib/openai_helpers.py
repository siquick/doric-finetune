"""
OpenAI client helpers shared by dataset generation utilities.
Supports multiple OpenAI-compatible API providers.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Tuple, Dict, Any

try:  # Optional dependency; callers receive a clear error if missing.
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

try:  # Local import guard so scripts can fail fast with a friendly message.
    from openai import OpenAI, AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore

DEFAULT_MODEL = "gpt-4.1-mini"

# Provider configurations
PROVIDER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "base_url": None,  # Uses OpenAI default
        "api_key_env": "OPENAI_API_KEY",
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "api_key_env": "HF_TOKEN",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",  # Falls back to OPENAI_API_KEY
    },
}


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: Optional[str] = None
    provider: Optional[str] = None
    extra_body: Optional[Dict[str, Any]] = (
        None  # For provider-specific options (e.g., reasoning)
    )

    def resolved_model(self, override: Optional[str] = None) -> str:
        return override or self.model or DEFAULT_MODEL


def _maybe_load_dotenv() -> None:
    if load_dotenv is not None:
        load_dotenv()


def read_openai_config(default_model: str = DEFAULT_MODEL) -> OpenAIConfig:
    """
    Read OpenAI connection info keyed by repo-wide env var naming.

    Supports multiple providers via MODEL_PROVIDER env var:
    - "openai" (default): Uses OPENAI_API_KEY, base_url=None
    - "huggingface": Uses HF_TOKEN, base_url=https://router.huggingface.co/v1
    - "openrouter": Uses OPENROUTER_API_KEY (or OPENAI_API_KEY), base_url=https://openrouter.ai/api/v1

    Can also override base_url via OPENAI_BASE_URL env var.
    """
    _maybe_load_dotenv()

    # Determine provider
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    model = os.getenv("MODEL", default_model)

    # Get provider config
    provider_config = PROVIDER_CONFIGS.get(provider)
    if not provider_config:
        raise RuntimeError(
            f"Unknown MODEL_PROVIDER: {provider}. "
            f"Supported providers: {list(PROVIDER_CONFIGS.keys())}"
        )

    # Get API key (try provider-specific first, then fallback to OPENAI_API_KEY)
    api_key_env = provider_config["api_key_env"]
    api_key = os.getenv(api_key_env)

    # Fallback to OPENAI_API_KEY for OpenRouter if OPENROUTER_API_KEY not set
    if not api_key and provider == "openrouter":
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            f"{api_key_env} is required for {provider} provider. "
            f"(OpenRouter can also use OPENAI_API_KEY)"
        )

    # Get base_url (env override takes precedence, then provider default)
    base_url = os.getenv("OPENAI_BASE_URL") or provider_config["base_url"]

    # Handle provider-specific configurations
    extra_body = None
    if provider == "openrouter":
        # Check if this is a thinking/reasoning model
        if "thinking" in model.lower() or "reasoning" in model.lower():
            extra_body = {"reasoning": {"enabled": True}}

    return OpenAIConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        extra_body=extra_body,
    )


def create_openai_client(
    config: Optional[OpenAIConfig] = None,
) -> Tuple["OpenAI", OpenAIConfig]:
    """
    Instantiate the OpenAI client and return it alongside its config.

    Supports multiple providers via config.provider and config.base_url.
    """
    if config is None:
        config = read_openai_config()
    if OpenAI is None:  # pragma: no cover
        raise RuntimeError("The 'openai' package is required but not installed.")
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    return client, config


def create_async_openai_client(
    config: Optional[OpenAIConfig] = None,
) -> Tuple["AsyncOpenAI", OpenAIConfig]:
    """
    Instantiate the async OpenAI client when concurrency is needed.

    Supports multiple providers via config.provider and config.base_url.
    """
    if config is None:
        config = read_openai_config()
    if AsyncOpenAI is None:  # pragma: no cover
        raise RuntimeError(
            "The 'openai' package (async) is required but not installed."
        )
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    return client, config


def get_extra_body_for_model(
    model: str, config: Optional[OpenAIConfig] = None
) -> Optional[Dict[str, Any]]:
    """
    Get provider-specific extra_body configuration for a model.

    Currently supports OpenRouter reasoning models. Returns None for other cases.
    Can be used when calling chat.completions.create() with extra_body parameter.

    Example:
        config = read_openai_config()
        extra_body = get_extra_body_for_model(config.model, config)
        response = client.chat.completions.create(
            model=config.model,
            messages=[...],
            extra_body=extra_body,
        )
    """
    if config is None:
        config = read_openai_config()

    # Use config's extra_body if set
    if config.extra_body:
        return config.extra_body

    # Auto-detect reasoning models for OpenRouter
    if config.provider == "openrouter":
        if "thinking" in model.lower() or "reasoning" in model.lower():
            return {"reasoning": {"enabled": True}}

    return None


__all__ = [
    "OpenAIConfig",
    "create_openai_client",
    "create_async_openai_client",
    "read_openai_config",
    "get_extra_body_for_model",
    "PROVIDER_CONFIGS",
    "DEFAULT_MODEL",
]
