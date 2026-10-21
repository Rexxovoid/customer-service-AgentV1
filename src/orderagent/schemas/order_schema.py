from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolContext(BaseModel):
    """Trusted caller context, injected by the application, never by model arguments."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    user_id: str = Field(min_length=1, max_length=100)


class OrderQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
    order_id: str = Field(min_length=1, max_length=100)


class OrderItem(BaseModel):
    product_id: str
    source_sku: str
    name: str
    model: str
    quantity: int = Field(gt=0)


class StoredOrder(BaseModel):
    """Validated input record; archival fields are intentionally not passed to the model."""

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


class OrderSummary(BaseModel):
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


class ToolError(BaseModel):
    code: str
    message: str


class ToolResult(BaseModel):
    ok: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None

    @classmethod
    def failure(cls, code: str, message: str) -> "ToolResult":
        return cls(ok=False, error=ToolError(code=code, message=message))
