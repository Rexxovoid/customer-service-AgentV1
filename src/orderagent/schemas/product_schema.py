from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProductQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)

    query: str = Field(
        min_length=1,
        max_length=200,
        description="品类、品牌、型号、商品名称或ID，例如鼠标、罗技 M185、MS-P-0002。",
    )


class ParameterEvidence(BaseModel):
    evidence: list[str]
    note: str
    complete: bool = False


class ProductSource(BaseModel):
    source_id: str
    url: str


class StoredProduct(BaseModel):
    """Only source-backed fields; archival text and images are not sent to the model."""

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
