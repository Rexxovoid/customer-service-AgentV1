from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.config import get_config
from pydantic import ValidationError

from orderagent.schemas.order_schema import OrderQuery, ToolResult
from orderagent.schemas.product_schema import ProductQuery
from orderagent.schemas.support_schema import HandoffQuery, SupportAgentState, SupportQuery
from orderagent.tools.support_tools import support_reply, transfer_support


class BusinessToolPolicyMiddleware(AgentMiddleware):
    """限制可用业务工具、校验调用参数，并控制一次质量方案后的人工交接流程。"""

    state_schema = SupportAgentState
    argument_schemas = {
        "query_order": OrderQuery,
        "query_products": ProductQuery,
        "query_quality_support": SupportQuery,
        "handoff_to_human": HandoffQuery,
    }

    def __init__(self, handoff_gateway):
        """注入与工具共用的客服接口，供明确反馈的快捷交接路径使用。"""
        self.handoff_gateway = handoff_gateway

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        """在模型生成前展示已完成的质量结果，或处理明确未解决及人工请求。"""
        last = state["messages"][-1]
        # 质量或交接结果直接展示并结束本轮，避免模型额外生成另一套方案。
        if (
            isinstance(last, ToolMessage)
            and last.name in {"query_quality_support", "handoff_to_human"}
            and last.artifact
            and (last.artifact.get("data") or {}).get("action") in {"advice", "handoff"}
            and (state.get("support_handoff") or state.get("support_advice"))
        ):
            return {"messages": [AIMessage(content=support_reply(state))], "jump_to": "end"}
        if not isinstance(last, HumanMessage):
            return None
        text = last.text.replace(" ", "")
        explicit = any(word in text for word in ("转人工", "人工客服", "找客服"))
        unsuccessful = state.get("support_phase") and any(
            word in text
            for word in (
                "未解决",
                "没有解决",
                "没解决",
                "不管用",
                "没用",
                "无效",
                "还是没声音",
                "仍然没声音",
                "还是没有声音",
                "还是不行",
                "还是不响",
            )
        )
        if any(word in text for word in ("还没试", "还没有试", "还没用", "还没尝试", "还没有尝试")):
            unsuccessful = False
        if not (explicit or unsuccessful):
            return None
        issue_state = {**state, "support_issue": state.get("support_issue", last.text)}
        update = transfer_support(
            issue_state,
            runtime.context,
            get_config().get("configurable", {}).get("thread_id", ""),
            "用户明确要求人工" if explicit else "用户反馈一次方案后仍未解决",
            self.handoff_gateway,
        )
        if not state.get("support_issue"):
            update["support_issue"] = last.text
        return {
            **update,
            "messages": [AIMessage(content=support_reply({**state, **update}))],
            "jump_to": "end",
        }

    def wrap_model_call(self, request, handler):
        """仅向模型提供白名单内的业务工具，再交给下一层模型调用处理。"""
        tools = [t for t in request.tools if getattr(t, "name", None) in self.argument_schemas]
        return handler(request.override(tools=tools))

    def wrap_tool_call(self, request, handler):
        """执行前校验工具名、参数和并行限制，拒绝时返回结构化工具错误。"""
        schema = self.argument_schemas.get(request.tool_call["name"])
        if schema is None:
            result = ToolResult.failure(
                "UNKNOWN_TOOL", "当前仅支持订单、商品、质量查询和人工交接。"
            )
        else:
            try:
                schema.model_validate(request.tool_call["args"])
                calls = request.state["messages"][-1].tool_calls
                if len(calls) > 1 and any(
                    call["name"] in {"query_quality_support", "handoff_to_human"} for call in calls
                ):
                    raise ValueError("质量与交接操作不能并行执行。")
            except (ValidationError, TypeError, ValueError):
                result = ToolResult.failure(
                    "INVALID_ARGUMENTS", "参数不合法，请按工具格式提供查询条件。"
                )
            else:
                return handler(request)
        return ToolMessage(
            content=result.model_dump_json(),
            artifact=result.model_dump(mode="json"),
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )
