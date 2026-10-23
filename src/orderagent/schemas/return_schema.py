from datetime import datetime
from typing import Literal, NotRequired

from pydantic import BaseModel, ConfigDict, Field

from orderagent.schemas.support_schema import SupportAgentState


class ReturnQuery(BaseModel):
    """退货工具参数允许缺项，由业务流程提示补充；不接受用户自报资格或链接。"""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    order_id: str = Field(default="", max_length=100)
    reason: str = Field(default="", max_length=500)
    request_human: bool = False


class ReturnEvidence(BaseModel):
    """可信适配器提供的资格证据；未知保持None，禁止以缺省值替代审核结论。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    order_id: str
    user_id: str
    checked_at: datetime
    simulated: bool
    goods_intact: bool | None = None
    no_reason_supported: bool | None = None
    excluded_by_exception: bool | None = None
    purchase_confirmation_checked: bool | None = None
    applicable_policy_checked: bool | None = None


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


class ReturnAgentState(SupportAgentState):
    """复用官方会话状态，记录当前退货问题和一次解释结果。"""

    return_order_id: NotRequired[str]
    return_reason: NotRequired[str]
    return_stage: NotRequired[
        Literal["collecting", "guided", "awaiting_confirmation", "link_ready", "handoff"] | None
    ]
    return_result: NotRequired[dict]
    return_rule_source: NotRequired[dict]
