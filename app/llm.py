"""LLM provider seam — the one place models are constructed.

Real multi-adapter seam: OpenRouter, Anthropic, or any OpenAI-compatible
endpoint, selected by LLM_PROVIDER. Domain modules build their agents on top
of this; provider details must not leak into them.
"""
from pydantic_ai.models import Model

from app.config import get_settings


def build_model(model_name: str) -> Model:
    """Build a Model for the configured provider.

    OpenRouter has a dedicated provider/profile (handles model-prefix routing,
    reasoning fields, schema transforms). Anthropic (and Anthropic-compatible
    gateways) use AnthropicModel + AnthropicProvider. Other OpenAI-compatible
    providers fall back to OpenAIChatModel + OpenAIProvider(base_url=...).
    """
    settings = get_settings()
    if settings.llm_provider == "openrouter":
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        return OpenRouterModel(model_name, provider=OpenRouterProvider(api_key=settings.llm_api_key))

    if settings.llm_provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        # base_url is optional — pass it only when set, so an Anthropic-compatible
        # gateway can be targeted while the default (api.anthropic.com) still works.
        provider_kwargs: dict = {"api_key": settings.llm_api_key}
        if settings.llm_base_url:
            provider_kwargs["base_url"] = settings.llm_base_url
        return AnthropicModel(model_name, provider=AnthropicProvider(**provider_kwargs))

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.llm_api_key),
    )
