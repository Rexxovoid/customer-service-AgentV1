from datetime import datetime, timedelta
from typing import Literal, Protocol
from urllib.parse import urlencode

from orderagent.schemas.return_schema import ReturnLinkRequest, ReturnLinkResponse


class ReturnGateway(Protocol):
    """退货入口契约；模拟/真实标记由适配器提供，入口不代表退货完成。"""

    allowed_hosts: frozenset[str]
    simulated: bool

    def create_link(self, request: ReturnLinkRequest, now: datetime) -> ReturnLinkResponse:
        """返回申请入口；不把生成入口当作已经提交或批准。"""
        ...


class MockReturnGateway:
    """仅模拟入口生成结果，不再提供或判断商品质量资格证据。"""

    allowed_hosts = frozenset({"returns.example.invalid"})
    simulated = True

    def __init__(self, scenario: Literal["unknown", "eligible", "ineligible", "error"] = "unknown"):
        """兼容旧配置：unknown/eligible均成功生成，ineligible/error仅模拟接口失败。"""
        if scenario not in {"unknown", "eligible", "ineligible", "error"}:
            raise ValueError("不支持的模拟退货场景。")
        self.scenario = scenario

    def create_link(self, request: ReturnLinkRequest, now: datetime) -> ReturnLinkResponse:
        """返回保留测试域名下的模拟入口，不访问平台、不创建真实售后申请。"""
        if self.scenario == "error":
            raise RuntimeError("模拟入口接口错误")
        ready = self.scenario in {"unknown", "eligible"}
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
