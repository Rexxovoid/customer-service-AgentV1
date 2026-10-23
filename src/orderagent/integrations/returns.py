from datetime import datetime, timedelta
from typing import Literal, Protocol
from urllib.parse import urlencode

from orderagent.schemas.order_schema import StoredOrder
from orderagent.schemas.return_schema import ReturnEvidence, ReturnLinkRequest, ReturnLinkResponse


class ReturnGateway(Protocol):
    """退货平台契约；真实实现需查询有效证据，限定链接域名并按请求ID幂等。"""

    allowed_hosts: frozenset[str]

    def get_evidence(self, order: StoredOrder, now: datetime) -> ReturnEvidence | None:
        """查询当前订单的完整资格证据，未查询到时返回None。"""
        ...

    def create_link(self, request: ReturnLinkRequest, now: datetime) -> ReturnLinkResponse:
        """返回申请入口；不把生成入口当作已经提交或批准。"""
        ...


class MockReturnGateway:
    """模拟平台：默认资格未知；eligible仅用于明确配置的成功路径演示。"""

    allowed_hosts = frozenset({"returns.example.invalid"})

    def __init__(self, scenario: Literal["unknown", "eligible", "ineligible", "error"] = "unknown"):
        """固定模拟场景，不允许用户通过工具参数改变资格。"""
        if scenario not in {"unknown", "eligible", "ineligible", "error"}:
            raise ValueError("不支持的模拟退货场景。")
        self.scenario = scenario

    def get_evidence(self, order: StoredOrder, now: datetime) -> ReturnEvidence | None:
        """模拟资格服务返回证据；默认不为基础订单补造完好或政策确认。"""
        if self.scenario == "error":
            raise RuntimeError("模拟资格接口错误。")
        if self.scenario == "unknown":
            return None
        return ReturnEvidence(
            order_id=order.id,
            user_id=order.user_id,
            checked_at=now,
            simulated=True,
            goods_intact=True,
            no_reason_supported=self.scenario == "eligible",
            excluded_by_exception=False,
            purchase_confirmation_checked=True,
            applicable_policy_checked=True,
        )

    def create_link(self, request: ReturnLinkRequest, now: datetime) -> ReturnLinkResponse:
        """返回保留测试域名下的模拟入口，不访问平台、不创建真实售后申请。"""
        ready = self.scenario == "eligible"
        return ReturnLinkResponse(
            request_id=request.request_id,
            order_id=request.order_id,
            user_id=request.user_id,
            status="ready" if ready else "error",
            simulated=True,
            url="https://returns.example.invalid/apply?"
            + urlencode({"request_id": request.request_id})
            if ready
            else None,
            expires_at=now + timedelta(minutes=30) if ready else None,
        )
