import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigurationError(ValueError):
    """Safe-to-display configuration error; never contains environment values."""


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str = "https://api.deepseek.com"
    api_key: SecretStr = SecretStr("")
    model_name: str = "deepseek-chat"
    orders_path: Path = PROJECT_ROOT / "data" / "orders.json"
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_model_calls: int = Field(default=6, ge=1, le=20)
    max_tool_calls: int = Field(default=8, ge=1, le=30)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("DEEP_BASE_URL 必须是不含凭据的 HTTPS 地址。")
        if parsed.query or parsed.fragment:
            raise ValueError("DEEP_BASE_URL 不应包含查询参数或片段。")
        return value.rstrip("/")

    @property
    def api_base_url(self) -> str:
        parsed = urlsplit(self.base_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            path = path.removesuffix("/chat/completions")
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        # Environment variables override .env without mutating os.environ.
        file_values = dotenv_values(env_file or PROJECT_ROOT / ".env", encoding="utf-8-sig")

        def value(name: str, default: str = "") -> str:
            return os.environ.get(name, file_values.get(name) or default)

        try:
            return cls(
                base_url=value("DEEP_BASE_URL", "https://api.deepseek.com"),
                api_key=SecretStr(value("DEEP_API_KEY")),
                model_name=value("DEEP_MODEL") or value("DEEP-SEEK", "deepseek-chat"),
                orders_path=Path(value("ORDER_DATA_PATH", str(cls().orders_path))).resolve(),
            )
        except ValueError:
            raise ConfigurationError("配置格式不合法，请检查 API 地址和订单数据路径。") from None

    def require_api_key(self) -> None:
        if not self.api_key.get_secret_value().strip():
            raise ConfigurationError("缺少 DEEP_API_KEY，请在项目 .env 或环境变量中配置。")
