from typing import Literal, NotRequired

from deepagents.graph import DeepAgentState
from pydantic import BaseModel, ConfigDict, Field


class ReportedFact(BaseModel):
    """用户陈述的字段及原文引用；不是可信平台证据，来源由程序补充。"""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    field: Literal[
        "symptom",
        "purchase_date",
        "receipt_date",
        "notice_date",
        "condition",
        "exception",
        "product_model",
        "activation",
        "wear",
        "exception_acknowledgement",
        "store_promise",
        "other",
    ]
    value: str = Field(max_length=300)
    status: Literal["explicit", "missing", "ambiguous", "conflict"]
    quote: str = Field(default="", max_length=500)
    normalized_description: str = Field(
        default="", max_length=300, description="仅症状检索用同义描述，不是新增事实或资格条件"
    )
    relation: Literal["supplement", "correction", "new_problem", "uncertain"] = Field(
        default="supplement", description="相对已有陈述的关系；新增症状不等于新的独立问题"
    )
    replaces_quote: str = Field(
        default="", max_length=500, description="纠正多条同字段陈述时，引用被纠正的历史原文"
    )


class SupportQuery(BaseModel):
    """质量知识查询参数，包含品类、具体症状及用户明确提供的环境。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)

    category: Literal["headphones", "mice", "keyboards"]
    issue: str = Field(min_length=1, max_length=500)
    environment: str = Field(default="", max_length=200, description="用户已提供的系统、品牌或型号")
    order_id: str = Field(default="", max_length=100, description="用户已提供的本人订单号；可留空")
    is_question: bool = True
    new_issue: bool = False
    attempted_solutions: list[str] = Field(default_factory=list)
    task: Literal["quality", "return", "return_policy"] = "quality"
    reported_facts: list[dict] = Field(default_factory=list)
    original_statements: list[str] = Field(default_factory=list)
    search_description: str = Field(default="", max_length=500)


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
    customer_reply: str = Field(
        default="", max_length=1800, description="用户可见的方案；证据原文另存answer"
    )
    applicability: str = Field(min_length=1)
    source: dict[str, str]
    data_origin: str
    aliases: list[str] = Field(default_factory=list)
    # 每组至少一个关键词须命中用户提供的环境，全部条件组都满足才适用。
    required_context: list[list[str]] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    clarify_on_partial_context: bool = False
    incompatible_context: list[str] = Field(default_factory=list)


class SupportAgentState(DeepAgentState):
    """质量与退货各自维护问题计数；咨询状态仅保存在官方线程检查点。"""

    support_phase: NotRequired[
        Literal["advised", "awaiting_information", "transferred", "transfer_failed"] | None
    ]
    support_issue: NotRequired[str]
    support_advice: NotRequired[dict]
    support_handoff: NotRequired[dict]
    support_order_id: NotRequired[str]
    support_question_count: NotRequired[int]
    support_question_history: NotRequired[dict]
    support_question_scope: NotRequired[str]
    support_problem_key: NotRequired[str]
    support_advice_history: NotRequired[dict]
    support_result: NotRequired[dict]
    support_handoff_history: NotRequired[dict]
    support_handoff_topic: NotRequired[str]
    question_processed_messages: NotRequired[dict]
    active_support_topic: NotRequired[str]
    consultation_id: NotRequired[str]
    support_error_code: NotRequired[str]
