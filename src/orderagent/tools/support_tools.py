from uuid import NAMESPACE_URL, uuid5

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from orderagent.integrations.handoff import HandoffGateway, HandoffRequest, HandoffResponse
from orderagent.repositories.support_repository import JsonSupportRepository
from orderagent.schemas.order_schema import ToolContext, ToolResult
from orderagent.schemas.support_schema import HandoffQuery, SupportQuery


def transfer_support(
    state: dict, context: ToolContext | None, thread_id: str, reason: str, gateway: HandoffGateway
) -> dict:
    """复用会话交接结果或请求客服接口，返回状态增量供官方检查点保存。"""
    if state.get("support_handoff"):
        return {}  # 成功与失败结果都复用，不在同一会话中静默重试。
    user_id = context.user_id if isinstance(context, ToolContext) else ""
    scope = state.get("return_order_id", "")
    request_id = "HANDOFF-" + uuid5(NAMESPACE_URL, f"{thread_id}:{user_id}:{scope}").hex[:16]
    if not isinstance(context, ToolContext) or not thread_id:
        response = HandoffResponse(request_id=request_id, status="error", simulated=True)
        message = "【模拟客服】转人工错误：缺少可信用户身份或会话标识。"
    else:
        request = HandoffRequest(
            request_id=request_id,
            user_id=context.user_id,
            issue=state.get("support_issue", reason),
            reason=reason,
            attempted_solution=state.get("support_advice", {}),
        )
        try:
            response = HandoffResponse.model_validate(gateway.transfer(request))
            if response.request_id != request_id:
                raise ValueError("Unexpected handoff request identity")
        except Exception:
            # 平台异常统一转为失败，不暴露凭据、请求内容或异常细节。
            response = HandoffResponse(request_id=request_id, status="error", simulated=False)
        prefix = "【模拟客服】" if response.simulated else ""
        if response.status == "connected" and response.agent_id:
            message = f"{prefix}对接人工客服成功，由工号{response.agent_id}为你服务。"
        else:
            response = response.model_copy(update={"status": "error", "agent_id": None})
            message = f"{prefix}转人工错误，请稍后重试或联系平台客服。"
    result = response.model_dump(mode="json")
    result.update({"message": message, "reason": reason})
    return {
        "support_phase": "transferred" if response.status == "connected" else "transfer_failed",
        "support_handoff": result,
    }


def support_reply(state: dict) -> str:
    """优先展示交接结果，否则展示一次知识方案、适用条件和来源。"""
    if state.get("support_handoff"):
        return state["support_handoff"]["message"]
    advice = state["support_advice"]
    source = advice["source"]
    reference = source.get("url") or source.get("reference_text", "项目模拟案例")
    return (
        f"可先进行这一次基础排查：{advice['answer']}\n"
        f"适用条件：{advice['applicability']}\n"
        f"知识依据：{advice['id']}，{reference}\n"
        "如果仍未解决，请直接告诉我，我将转人工，不再继续提供其他排查方案。"
    )


def create_support_tools(
    repository: JsonSupportRepository, gateway: HandoffGateway, *, order_repository=None
) -> list[BaseTool]:
    """绑定统一知识仓储与客服接口，创建质量查询和人工交接两个官方工具。"""

    def command(runtime: ToolRuntime, update: dict, result: ToolResult) -> Command:
        """把状态增量与对应ToolMessage交给官方Command统一更新。"""
        return Command(
            update={
                **update,
                "messages": [
                    ToolMessage(
                        content=result.model_dump_json(),
                        artifact=result.model_dump(mode="json"),
                        tool_call_id=runtime.tool_call_id,
                        name=next(
                            call["name"]
                            for call in runtime.state["messages"][-1].tool_calls
                            if call["id"] == runtime.tool_call_id
                        ),
                        status="success" if result.ok else "error",
                    )
                ],
            }
        )

    def handoff(runtime: ToolRuntime, reason: str, issue: str = "", order_id: str = "") -> Command:
        """补齐故障上下文并执行交接，将接口结果转换为工具状态更新。"""
        state = dict(runtime.state)
        if order_id:
            state["support_order_id"] = order_id
        if state.get("return_stage"):
            state["support_issue"] = (
                f"退货订单={state.get('return_order_id') or '未提供'}；"
                f"原因={state.get('return_reason') or '未提供'}"
            )
        if issue and not state.get("support_issue"):
            state["support_issue"] = issue
        update = transfer_support(
            state,
            runtime.context,
            runtime.config.get("configurable", {}).get("thread_id", ""),
            reason,
            gateway,
        )
        if issue:
            update["support_issue"] = state["support_issue"]
        if state.get("support_order_id"):
            update["support_order_id"] = state["support_order_id"]
        if state.get("return_stage"):
            update["support_issue"] = state["support_issue"]
            update["return_stage"] = "handoff"
        combined = {**state, **update}
        result = ToolResult(
            ok=combined["support_phase"] == "transferred",
            data={"action": "handoff", **combined["support_handoff"]},
        )
        return command(runtime, update, result)

    @tool("query_quality_support")
    def query_quality_support(
        category: str,
        issue: str,
        runtime: ToolRuntime[ToolContext],
        environment: str = "",
        order_id: str = "",
    ) -> Command:
        """用户首次描述耳机、鼠标或键盘的实际使用故障时，查询有依据的一次基础方案。

        适用示例：耳机左耳没有声音、无线鼠标没反应、USB键盘无响应。
        category必须为headphones（耳机）、mice（鼠标）或keyboards（键盘）；
        品类不明确时先询问。issue填写具体症状，不要只填商品名或泛称质量问题。
        environment只填用户已明确提供的系统、品牌或型号，例如Windows 11、AirPods iPhone；
        未提供时留空，禁止猜测环境以套用特定方案。无需先提供订单号。
        order_id只填用户已提供的本人订单号，提供后核验归属并绑定售后状态；无订单只作咨询。
        品类须与订单商品一致；订单快照未签收时交人工核实，不编造到货事实。
        返回匹配知识的答案、适用条件、来源和会话阶段；无适用知识自动调用人工接口。
        同一会话最多给一次自动方案，再次查询不会给新方案，而是转人工。
        已给方案后用户反馈未解决、已经自行排查无效或明确要求人工，
        应直接调用handoff_to_human，不能继续尝试本工具或凭常识补充排查步骤。
        查询商品性能参数请用query_products，查订单物流请用query_order。
        本工具必须单独调用，不与其他工具并行；最终方案和交接回复由程序输出。
        """
        query = SupportQuery(
            category=category, issue=issue, environment=environment, order_id=order_id
        )
        binding = order_id or runtime.state.get("support_order_id", "")
        if binding and order_repository is not None:
            supplied = any(
                isinstance(message, HumanMessage) and binding in message.text
                for message in runtime.state["messages"]
            )
            owned = (
                order_repository.find_owned(binding, runtime.context.user_id)
                if isinstance(runtime.context, ToolContext) and supplied
                else None
            )
            if owned is None:
                return command(
                    runtime,
                    {},
                    ToolResult.failure(
                        "ORDER_NOT_FOUND",
                        "订单不存在、不属于当前用户或并非用户提供，请核对订单号。",
                    ),
                )
            prefix = {"headphones": "HP-P-", "mice": "MS-P-", "keyboards": "KB-P-"}[category]
            if not any(item.product_id.startswith(prefix) for item in owned.items):
                return command(
                    runtime,
                    {},
                    ToolResult.failure(
                        "ORDER_PRODUCT_MISMATCH", "订单中的商品与所述品类不符，请确认订单或商品。"
                    ),
                )
            if owned.order_status != "delivered":
                return handoff(
                    runtime,
                    "订单快照未签收，与到货故障场景不一致，请人工核实",
                    issue=query.issue,
                    order_id=binding,
                )
            if getattr(order_repository, "after_sales", None) is not None:
                store = order_repository.after_sales
                started = store.start_issue(
                    runtime.context.user_id,
                    binding,
                    f"{runtime.config.get('configurable', {}).get('thread_id')}:"
                    f"{runtime.tool_call_id}:reported",
                )
                if not started:
                    return handoff(
                        runtime,
                        "该订单已有售后处理记录，交人工核实，避免重复排查",
                        issue=query.issue,
                        order_id=binding,
                    )
        if runtime.state.get("support_phase") or runtime.state.get("return_stage"):
            return handoff(runtime, "当前问题已解释或给出一次方案，停止自动排查", order_id=binding)
        try:
            knowledge = repository.find(query)
        except Exception:
            return handoff(
                runtime, "知识库服务异常，无法核实适用规则", issue=query.issue, order_id=binding
            )
        if knowledge is None or knowledge.source.get("handling", "basic_troubleshooting") != (
            "basic_troubleshooting"
        ):
            return handoff(runtime, "未查询到适用的故障知识", issue=query.issue, order_id=binding)
        advice = knowledge.model_dump(mode="json")
        update = {
            "support_phase": "advised",
            "support_issue": query.issue,
            "support_advice": advice,
            "support_order_id": binding,
        }
        return command(runtime, update, ToolResult(ok=True, data={"action": "advice", **advice}))

    @tool("handoff_to_human")
    def handoff_to_human(reason: str, runtime: ToolRuntime[ToolContext]) -> Command:
        """用户要求人工、反馈一次方案后仍未解决，或缺少适用知识时立即请求人工交接。

        适用示例：用户说没有解决、试过仍无效、故障依旧、请找人工客服。
        reason填写简短交接原因，可概括故障和已尝试方案，但不编造未发生的处理记录。
        用户身份、会话标识和已有方案由程序提供；不要传user_id、工号、成功状态或runtime。
        一旦需要交接就停止自动排查，不再调用质量查询或商品工具寻找另一套维修方案。
        返回接口状态、请求ID及客服工号（成功时）；成功和失败均以接口结果为准，
        失败不能说成功。当前默认接口为模拟客服，不表示已连接真人。
        同一会话重复交接会复用已记录的成功或失败结果，不自动重试。
        必须单独调用，不与其他工具并行，不发起退款或批准退换货。
        """
        arguments = HandoffQuery(reason=reason)
        return handoff(runtime, arguments.reason)

    return [query_quality_support, handoff_to_human]
