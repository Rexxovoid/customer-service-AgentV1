from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolContext(BaseModel):
    """应用注入的可信用户上下文，不允许由模型工具参数替换。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    user_id: str = Field(min_length=1, max_length=100)


class OrderQuery(BaseModel):
    """订单查询参数，只接受非空订单号并拒绝额外身份字段。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
    order_id: str = Field(min_length=1, max_length=100)


class OrderItem(BaseModel):
    """订单内单项商品的标识、名称、型号及购买数量。"""

    product_id: str
    source_sku: str
    name: str
    model: str
    quantity: int = Field(gt=0)


class StoredOrder(BaseModel):
    """校验后的订单存储结构，不含无需交给模型的原始归档字段。"""

    id: str
    user_id: str
    synthetic: bool
    order_status: str
    order_status_label: str
    shipping_status: str
    logistics_status: str
    tracking_number: str | None
    items: list[OrderItem] = Field(min_length=1)
    snapshot_at: datetime
    delivered_at: datetime | None = None
    after_sales_status: str = "未售后"


class OrderSummary(BaseModel):
    """允许返回给模型的订单摘要，包含快照时间及模拟数据标记。"""

    order_id: str
    order_status: str
    order_status_label: str
    shipping_status: str
    logistics_status: str
    tracking_number: str | None
    items: list[OrderItem]
    snapshot_at: datetime
    synthetic: bool
    data_source: Literal["local_demo_json"] = "local_demo_json"
    after_sales_status: str = "未售后"


class ToolError(BaseModel):
    """业务工具失败时统一返回的错误码和可展示消息。"""

    code: str
    message: str


class ToolResult(BaseModel):
    """工具统一结果结构，通过ok区分成功与失败。"""

    ok: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None

    @classmethod
    def failure(cls, code: str, message: str) -> "ToolResult":
        """构造不带业务数据的失败结果，统一各工具的错误返回形式。"""
        return cls(ok=False, error=ToolError(code=code, message=message))
