from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from orderagent.schemas.order_schema import OrderQuery, ToolResult


class OrderToolPolicyMiddleware(AgentMiddleware):
    """Offer only query_order; reject hidden built-in tool calls."""

    def wrap_model_call(self, request, handler):
        tools = [t for t in request.tools if getattr(t, "name", None) == "query_order"]
        return handler(request.override(tools=tools))

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] != "query_order":
            result = ToolResult.failure("UNKNOWN_TOOL", "当前仅支持订单查询工具。")
        else:
            try:
                OrderQuery.model_validate(request.tool_call["args"])
                return handler(request)
            except (ValidationError, TypeError):
                result = ToolResult.failure("INVALID_ARGUMENTS", "参数不合法，请只提供有效订单号。")
        return ToolMessage(
            content=result.model_dump_json(),
            artifact=result.model_dump(mode="json"),
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )
