from langchain_deepseek import ChatDeepSeek

from orderagent.settings import Settings


def create_deepseek_model(settings: Settings) -> ChatDeepSeek:
    """Adapt existing environment names to the official LangChain integration."""
    settings.require_api_key()
    return ChatDeepSeek(
        model=settings.model_name,
        api_key=settings.api_key,
        api_base=settings.api_base_url,
        timeout=settings.timeout_seconds,
        max_retries=1,
        max_tokens=1200,
        use_responses_api=False,
    )
