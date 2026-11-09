from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RESULT_STORAGE_VERSION = 3


class Check(BaseModel):
    """机器检查与自然语言规格分离，禁止将未检查项冒充通过。"""

    model_config = ConfigDict(extra="forbid")
    kind: Literal[
        "state", "tool", "tool_arg", "creates", "operation", "tool_error", "answer_contains"
    ]
    key: str = ""
    value: str | int | bool | None = None


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user: str
    expect: list[str]
    forbid: list[str]
    checks: list[Check] = Field(default_factory=list)
    # 初版分支：检查失败/成功分别可提前结束；默认成功进入下一轮。
    on_pass: Literal["continue", "stop"] = "continue"
    on_fail: Literal["stop", "continue"] = "stop"


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^E\d{2}$")
    category: str
    user_id: str
    mode: Literal["live_agent", "controlled_agent", "live_rag"]
    fixture: str | None = None
    turns: list[Turn] = Field(default_factory=list)
    query: dict | None = None
    expect: list[str] = Field(default_factory=list)
    forbid: list[str] = Field(default_factory=list)
    reference: str
    review_status: Literal["pending", "approved"] = "pending"
    content_upload_approved: bool = False
    retrieval_only: bool = False
    # 真实检索金标由人工填写，不根据Agent当前输出自动产生。
    relevant_documents: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_route(self):
        """防止错模式或缺失夹具静默变成真实业务调用。"""
        allowed = {
            "basic_headphone_rule",
            "no_applicable_rule",
            "rag_timeout",
            "applicable_return_rule",
            "trusted_refunded_order",
            "handoff_error_and_no_rule",
        }
        if self.retrieval_only and self.mode != "live_rag":
            raise ValueError("retrieval_only仅适用于live_rag")
        if self.mode == "controlled_agent" and self.fixture not in allowed:
            raise ValueError("受控案例必须指定已知fixture")
        if self.mode == "live_rag" and (not self.query or self.turns):
            raise ValueError("RAG案例必须包含query且不包含turns")
        if self.mode != "live_rag" and not self.turns:
            raise ValueError("Agent案例需要turns")
        return self


class Dimension(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: int | None = Field(ge=0, le=4)
    reason: str
    evidence: str


class QualityScore(BaseModel):
    """分维度评分；null表示证据不足，与0分不同。"""

    model_config = ConfigDict(extra="forbid")
    factual_correctness: Dimension
    faithfulness: Dimension
    relevance: Dimension
    actionability: Dimension
    expression: Dimension
