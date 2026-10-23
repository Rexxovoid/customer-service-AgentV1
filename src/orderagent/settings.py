import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigurationError(ValueError):
    """可展示给用户的配置异常，消息不包含环境变量值或敏感凭据。"""


class Settings(BaseModel):
    """集中保存模型、数据路径和运行限制，实例创建后配置不可变。"""

    model_config = ConfigDict(frozen=True)

    base_url: str = "https://api.deepseek.com"
    api_key: SecretStr = SecretStr("")
    model_name: str = "deepseek-chat"
    orders_path: Path = PROJECT_ROOT / "data" / "orders.json"
    products_path: Path = PROJECT_ROOT / "data" / "products.json"
    support_path: Path = PROJECT_ROOT / "data" / "support_knowledge.json"
    after_sales_db_path: Path = PROJECT_ROOT / "runtime" / "after_sales.sqlite3"
    ragflow_api_url: str = ""
    ragflow_api_key: SecretStr = SecretStr("")
    ragflow_timeout_seconds: float = Field(default=60, gt=0, le=120)
    ragflow_dataset_name: str = "售后规则知识库"
    ragflow_assistant_name: str = "售后助手"
    mock_handoff_scenario: Literal["success", "error"] = "success"
    mock_return_scenario: Literal["unknown", "eligible", "ineligible", "error"] = "unknown"
    return_self_service_entry: str = Field(
        default="示例入口：我的订单 → 订单详情 → 退货/退款；实际按钮以所用平台为准。",
        min_length=1,
        max_length=1000,
    )
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_model_calls: int = Field(default=6, ge=1, le=20)
    max_tool_calls: int = Field(default=8, ge=1, le=30)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """校验无凭据的HTTPS地址，并去除末尾斜杠。"""
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("DEEP_BASE_URL 必须是不含凭据的 HTTPS 地址。")
        if parsed.query or parsed.fragment:
            raise ValueError("DEEP_BASE_URL 不应包含查询参数或片段。")
        return value.rstrip("/")

    @field_validator("ragflow_api_url")
    @classmethod
    def validate_ragflow_url(cls, value: str) -> str:
        """兼容已有HTTP服务器，拒绝URL内凭据、查询串和片段。"""
        if not value:
            return value
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("RAGFlow地址不合法")
        return value.rstrip("/")

    @property
    def api_base_url(self) -> str:
        """将可能含chat/completions的配置地址规范为SDK所需的服务基础地址。"""
        parsed = urlsplit(self.base_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            path = path.removesuffix("/chat/completions")
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        """读取.env和系统环境变量，解析路径并校验配置，系统环境变量优先。"""
        # 只读取配置，不将.env内容写入进程环境变量。
        file_values = dotenv_values(env_file or PROJECT_ROOT / ".env", encoding="utf-8-sig")

        def value(name: str, default: str = "") -> str:
            """按系统环境、.env、默认值的优先级读取指定配置项。"""
            return os.environ.get(name, file_values.get(name) or default)

        try:
            return cls(
                base_url=value("DEEP_BASE_URL", "https://api.deepseek.com"),
                api_key=SecretStr(value("DEEP_API_KEY")),
                model_name=value("DEEP_MODEL") or value("DEEP-SEEK", "deepseek-chat"),
                orders_path=Path(value("ORDER_DATA_PATH", str(cls().orders_path))).resolve(),
                products_path=Path(value("PRODUCT_DATA_PATH", str(cls().products_path))).resolve(),
                support_path=Path(value("SUPPORT_DATA_PATH", str(cls().support_path))).resolve(),
                after_sales_db_path=Path(
                    value("AFTER_SALES_DB_PATH", str(cls().after_sales_db_path))
                ).resolve(),
                ragflow_api_url=value("RAGFLOW_API_URL"),
                ragflow_api_key=SecretStr(value("RAGFLOW_API_KEY")),
                ragflow_timeout_seconds=value("RAGFLOW_TIMEOUT_SECONDS", "60"),
                ragflow_dataset_name=value("RAGFLOW_DATASET_NAME", "售后规则知识库"),
                ragflow_assistant_name=value("RAGFLOW_ASSISTANT_NAME", "售后助手"),
                mock_handoff_scenario=value("MOCK_HANDOFF_SCENARIO", "success"),
                mock_return_scenario=value("MOCK_RETURN_SCENARIO", "unknown"),
                return_self_service_entry=value(
                    "RETURN_SELF_SERVICE_ENTRY", cls().return_self_service_entry
                ),
            )
        except ValueError:
            raise ConfigurationError(
                "配置格式不合法，请检查API地址、数据路径和模拟交接场景。"
            ) from None

    def require_api_key(self) -> None:
        """在创建模型客户端前检查密钥是否为空，缺失时抛出脱敏配置异常。"""
        if not self.api_key.get_secret_value().strip():
            raise ConfigurationError("缺少 DEEP_API_KEY，请在项目 .env 或环境变量中配置。")
