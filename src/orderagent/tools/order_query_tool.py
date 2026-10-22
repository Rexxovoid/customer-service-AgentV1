from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool

from orderagent.repositories.order_repository import OrderRepository
from orderagent.schemas.order_schema import OrderQuery, OrderSummary, ToolContext, ToolResult


def query_order(
    repository: OrderRepository, arguments: OrderQuery, context: ToolContext
) -> ToolResult:
    """校验订单归属并筛选状态、物流和商品明细，供CLI与Agent工具共同使用。"""
    order = repository.find_owned(arguments.order_id, context.user_id)
    if order is None:
        # 不存在和无权限统一返回相同结果，避免泄露其他用户的订单是否存在。
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


def create_order_query_tool(repository: OrderRepository) -> BaseTool:
    """绑定订单仓储并返回官方工具，由LangGraph注入上下文和执行调用。"""

    @tool(
        "query_order",
        description=(
            "当用户询问某个已购订单的付款状态、是否发货、物流单号或订单商品明细时调用。"
            "order_id为用户提供或当前会话已确认的完整订单号，例如DEMO-HP-O-0001；"
            "没有订单号先询问，禁止编造、枚举或用商品型号代替。用户身份由程序注入，"
            "不要传user_id或runtime。返回当前用户所属订单的状态、商品型号数量和快照时间；"
            "订单不存在或不属于当前用户统一返回ORDER_NOT_FOUND，不推测其他用户信息。"
            "结果是本地模拟快照，不是平台实时数据。商品参数咨询用query_products，"
            "质量故障用query_quality_support；本工具不修改订单、不退货、不退款。"
        ),
        response_format="content_and_artifact",
    )
    def order_query(order_id: str, runtime: ToolRuntime[ToolContext]) -> tuple[str, dict]:
        """从框架上下文获取可信身份，返回模型可读JSON及程序可读artifact。"""
        # runtime由框架注入，不出现在模型可填写的参数列表中。
        context = runtime.context
        if not isinstance(context, ToolContext):
            result = ToolResult.failure("CONTEXT_REQUIRED", "缺少可信用户上下文。")
        else:
            result = query_order(repository, OrderQuery(order_id=order_id), context)
        return result.model_dump_json(), result.model_dump(mode="json")

    return order_query
