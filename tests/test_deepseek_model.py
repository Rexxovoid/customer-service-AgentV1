import pytest
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from orderagent.models import create_deepseek_model
from orderagent.settings import ConfigurationError, Settings


def test_configuration_aliases_and_environment_precedence(tmp_path, monkeypatch):
    for name in ("DEEP_BASE_URL", "DEEP_API_KEY", "DEEP-SEEK", "DEEP_MODEL"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text('DEEP_API_KEY="file-secret"\nDEEP-SEEK=custom-model\n', encoding="utf-8")
    settings = Settings.load(env)
    assert settings.model_name == "custom-model"
    assert settings.api_key.get_secret_value() == "file-secret"
    assert "file-secret" not in repr(settings)
    monkeypatch.setenv("DEEP_MODEL", "environment-model")
    assert Settings.load(env).model_name == "environment-model"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://api.deepseek.com", "https://api.deepseek.com"),
        ("https://api.deepseek.com/v1/", "https://api.deepseek.com/v1"),
        ("https://api.deepseek.com/v1/chat/completions", "https://api.deepseek.com/v1"),
    ],
)
def test_api_base_url_normalization(url, expected):
    assert Settings(base_url=url).api_base_url == expected


def test_missing_key_fails_before_client_creation():
    with pytest.raises(ConfigurationError):
        create_deepseek_model(Settings())


def test_model_factory_uses_chat_deepseek_without_network():
    model = create_deepseek_model(Settings(api_key=SecretStr("fake-test-key")))
    assert isinstance(model, ChatDeepSeek)
    assert model.model_name == "deepseek-chat"
    assert model.api_base == "https://api.deepseek.com"
    assert model.request_timeout == 30
    assert model.max_retries == 1
