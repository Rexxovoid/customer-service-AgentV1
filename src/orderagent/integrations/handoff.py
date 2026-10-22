from typing import Literal, Protocol

from pydantic import BaseModel, Field


class HandoffRequest(BaseModel):
    """人工交接请求；真实平台适配器需使用request_id作为幂等键。"""

    request_id: str
    user_id: str
    issue: str
    reason: str
    attempted_solution: dict = Field(default_factory=dict)


class HandoffResponse(BaseModel):
    """平台交接响应，区分成功、失败、模拟标记及实际返回的客服工号。"""

    request_id: str
    status: Literal["connected", "error"]
    simulated: bool
    agent_id: str | None = None


class HandoffGateway(Protocol):
    """客服平台适配契约，真实实现需设置网络超时并确保请求幂等。"""

    def transfer(self, request: HandoffRequest) -> HandoffResponse:
        """向客服平台发起交接并返回状态，不由模型决定是否成功。"""
        ...


class MockHandoffGateway:
    """固定场景的模拟客服接口，不连接真人或外部平台。"""

    def __init__(self, scenario: Literal["success", "error"] = "success"):
        """校验并保存由程序配置选择的成功或失败测试场景。"""
        if scenario not in {"success", "error"}:
            raise ValueError("不支持的模拟交接场景。")
        self.scenario = scenario

    def transfer(self, request: HandoffRequest) -> HandoffResponse:
        """按配置返回模拟结果；成功分配工号001，失败不返回客服工号。"""
        success = self.scenario == "success"
        return HandoffResponse(
            request_id=request.request_id,
            status="connected" if success else "error",
            simulated=True,
            agent_id="001" if success else None,
        )
