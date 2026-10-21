from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool

from orderagent.repositories.order_repository import OrderRepository
from orderagent.schemas.order_schema import OrderQuery, OrderSummary, ToolContext, ToolResult


class OrderQueryTool:
    name = "query_order"
    description = (
        "根据订单号查询当前用户的订单状态、发货物流与商品信息。"
        "用户身份由程序注入，不要传入user_id。未提供订单号时先询问。"
    )

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    def execute(self, arguments: OrderQuery, context: ToolContext) -> ToolResult:
        order = self.repository.find_owned(arguments.order_id, context.user_id)
        if order is None:
            # Identical response for absent and unauthorized orders prevents enumeration.
            return ToolResult.failure("ORDER_NOT_FOUND", "订单不存在或不属于当前用户。")
        summary = OrderSummary(
            order_id=order.id,
            order_status=order.order_status,
            order_status_label=order.order_status_label,
            shipping_status=order.shipping_status,
            logistics_status=order.logistics_status,
            tracking_number=order.tracking_number,
            items=order.items,
            snapshot_at=order.snapshot_at,
            synthetic=order.synthetic,
        )
        return ToolResult(ok=True, data=summary.model_dump(mode="json"))

    def as_langchain_tool(self) -> BaseTool:
        @tool(
            self.name,
            description=self.description,
            response_format="content_and_artifact",
        )
        def query_order(order_id: str, runtime: ToolRuntime[ToolContext]) -> tuple[str, dict]:
            # Runtime is injected by LangGraph, absent from model-facing schema.
            context = runtime.context
            if not isinstance(context, ToolContext):
                result = ToolResult.failure("CONTEXT_REQUIRED", "缺少可信用户上下文。")
            else:
                result = self.execute(OrderQuery(order_id=order_id), context)
            return result.model_dump_json(), result.model_dump(mode="json")

        return query_order
