from typing import Literal, NotRequired

from deepagents.graph import DeepAgentState
from pydantic import BaseModel, ConfigDict, Field


class SupportQuery(BaseModel):
    """质量知识查询参数，包含品类、具体症状及用户明确提供的环境。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)

    category: Literal["headphones", "mice", "keyboards"]
    issue: str = Field(min_length=1, max_length=500)
    environment: str = Field(default="", max_length=200, description="用户已提供的系统、品牌或型号")
    order_id: str = Field(default="", max_length=100, description="用户已提供的本人订单号；可留空")


class HandoffQuery(BaseModel):
    """模型可填写的交接原因，不允许指定用户身份或接口成功状态。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)

    reason: str = Field(min_length=1, max_length=500)
    topic: Literal["auto", "quality", "return"] = "auto"
    request_human: bool = False


class SupportKnowledge(BaseModel):
    """可用于基础排查的知识记录，保留来源性质、适用条件和症状别名。"""

    id: str = Field(min_length=1)
    category: Literal["headphones", "mice", "keyboards"]
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    applicability: str = Field(min_length=1)
    source: dict[str, str]
    data_origin: str
    aliases: list[str] = Field(default_factory=list)
    # 每组至少一个关键词须命中用户提供的环境，全部条件组都满足才适用。
    required_context: list[list[str]] = Field(default_factory=list)


class SupportAgentState(DeepAgentState):
    """扩展官方会话状态，记录一次方案及人工交接结果，由检查点按线程保存。"""

    support_phase: NotRequired[Literal["advised", "transferred", "transfer_failed"] | None]
    support_issue: NotRequired[str]
    support_advice: NotRequired[dict]
    support_handoff: NotRequired[dict]
    support_order_id: NotRequired[str]
