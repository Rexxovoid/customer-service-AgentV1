from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from orderagent.schemas.order_schema import OrderQuery, ToolResult
from orderagent.schemas.product_schema import ProductQuery


class BusinessToolPolicyMiddleware(AgentMiddleware):
    """Offer business tools only; validate their arguments before execution."""

    argument_schemas = {"query_order": OrderQuery, "query_products": ProductQuery}

    def wrap_model_call(self, request, handler):
        tools = [t for t in request.tools if getattr(t, "name", None) in self.argument_schemas]
        return handler(request.override(tools=tools))

    def wrap_tool_call(self, request, handler):
        schema = self.argument_schemas.get(request.tool_call["name"])
        if schema is None:
            result = ToolResult.failure("UNKNOWN_TOOL", "当前仅支持订单和商品查询工具。")
        else:
            try:
                schema.model_validate(request.tool_call["args"])
                return handler(request)
            except (ValidationError, TypeError):
                result = ToolResult.failure(
                    "INVALID_ARGUMENTS", "参数不合法，请按工具格式提供查询条件。"
                )
        return ToolMessage(
            content=result.model_dump_json(),
            artifact=result.model_dump(mode="json"),
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )
