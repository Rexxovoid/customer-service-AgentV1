from datetime import datetime
from typing import Literal, NotRequired

from pydantic import BaseModel, ConfigDict, Field

from orderagent.schemas.support_schema import SupportAgentState

RETURN_HANDOFF_THRESHOLD = 4


class ReturnQuery(BaseModel):
    """退货工具参数允许缺项，由业务流程提示补充；不接受用户自报资格或链接。"""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    order_id: str = Field(default="", max_length=100)
    reason: str = Field(default="", max_length=500)
    request_human: bool = False
    is_return_request: bool = True


class ReturnLinkRequest(BaseModel):
    """仅请求申请入口，不执行正式提交、审批或退款。"""

    request_id: str
    order_id: str
    user_id: str
    reason: str


class ReturnLinkResponse(BaseModel):
    """申请入口响应必须与当前请求及订单对应，并声明模拟性质与有效期。"""

    request_id: str
    order_id: str
    user_id: str
    status: Literal["ready", "error"]
    simulated: bool
    url: str | None = None
    expires_at: datetime | None = None


class ReturnOperation(BaseModel):
    """独立于普通售后记录的幂等台账；终态只能由可信业务适配器写入。"""

    model_config = ConfigDict(extra="forbid")
    user_id: str
    order_id: str
    request_id: str
    status: Literal["creating", "link_ready", "processing", "uncertain", "returned", "refunded"]
    simulated: bool = True
    response: ReturnLinkResponse | None = None


class ReturnAgentState(SupportAgentState):
    """复用官方会话状态，记录退货诉求计数、当前阶段和工具结果。"""

    return_order_id: NotRequired[str]
    return_reason: NotRequired[str]
    return_stage: NotRequired[
        Literal[
            "collecting", "guided", "awaiting_confirmation", "link_ready", "handoff", "rejected"
        ]
        | None
    ]
    return_result: NotRequired[dict]
    return_rule_source: NotRequired[dict]
    return_request_count: NotRequired[int]
    return_request_history: NotRequired[dict[str, dict]]
