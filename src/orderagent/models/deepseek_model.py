from langchain_deepseek import ChatDeepSeek

from orderagent.settings import Settings


def create_deepseek_model(settings: Settings) -> ChatDeepSeek:
    """校验密钥并创建官方ChatDeepSeek客户端，统一模型、超时及SDK重试配置。"""
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
