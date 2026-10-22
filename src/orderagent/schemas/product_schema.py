from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProductQuery(BaseModel):
    """商品实体关键词参数，不接受身份信息或其他额外字段。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)

    query: str = Field(
        min_length=1,
        max_length=200,
        description="品类、品牌、型号、商品名称或ID，例如鼠标、罗技 M185、MS-P-0002。",
    )


class ParameterEvidence(BaseModel):
    """商品参数的原文证据分组；空分组不能解释为商品不支持该功能。"""

    evidence: list[str]
    note: str
    complete: bool = False


class ProductSource(BaseModel):
    """商品资料的来源标识与可追溯链接。"""

    source_id: str
    url: str


class StoredProduct(BaseModel):
    """仅保留商品查询所需字段，不将归档文本和图片等原始信息传给模型。"""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    brand: str = Field(min_length=1)
    model: str = Field(min_length=1)
    category: Literal["mice", "keyboards", "headphones"]
    category_name: str
    source_sku: str
    description: str
    evidence_groups: dict[str, ParameterEvidence]
    source: ProductSource
    source_last_updated: str
    current_price: dict[str, Any] | None = None
    stock: Any = None
