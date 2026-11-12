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


class HumanMessageResponse(BaseModel):
    """模拟消息结果；关闭必须是结构化事件，不能从回复文本判断。"""

    status: Literal["received", "error"]
    event: Literal["message", "conversation_closed"] = "message"
    message: str = ""
    waiting_user: bool = False


class MockHumanChatGateway:
    """已有模拟交接后的固定消息接口，无LLM、网络或真实售后副作用。"""

    def __init__(self, scenario="reply", development=False):
        if scenario not in {"reply", "error", "close"}:
            raise ValueError("不支持的模拟人工消息场景")
        if scenario != "reply" and not development:
            raise ValueError("人工测试场景须显式开启开发模式")
        self.scenario = scenario

    def send(self, *, handoff_id, agent_id, user_id, message_id, text):
        """仅已连接的持久交接可发消息；场景来自后端配置而非用户正文。"""
        if not all((handoff_id, agent_id, user_id, message_id, text)):
            raise ValueError("缺少人工消息绑定")
        if self.scenario == "error":
            return HumanMessageResponse(status="error")
        if self.scenario == "close":
            return HumanMessageResponse(
                status="received",
                event="conversation_closed",
                message=f"【模拟客服{agent_id}】本次模拟服务准备结束。",
            )
        return HumanMessageResponse(
            status="received",
            waiting_user=True,
            message=f"【模拟客服{agent_id}】已收到你的消息，请补充当前情况或需要协助的内容。",
        )
