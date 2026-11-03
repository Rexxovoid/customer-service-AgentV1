from uuid import NAMESPACE_URL, uuid5

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from orderagent.integrations.handoff import HandoffGateway, HandoffRequest, HandoffResponse
from orderagent.integrations.ragflow import RAG_ERROR_MESSAGES
from orderagent.schemas.order_schema import ToolContext, ToolResult
from orderagent.schemas.support_schema import HandoffQuery, SupportQuery
from orderagent.tools.session_policy import (
    AUTO_HANDOFF_FROM,
    HANDOFF_SUGGESTION,
    SUGGEST_HANDOFF_FROM,
    advance_question,
    reset_consultation,
)


def transfer_support(state, context, thread_id, reason, gateway, *, topic="quality") -> dict:
    """按主题和问题复用交接结果，不写订单状态，不静默重试失败交接。"""
    prefix = "return" if topic == "return" else "support"
    scope = state.get(f"{prefix}_problem_key") or state.get(f"{prefix}_order_id", "")
    key = f"{topic}:{scope}"
    history = dict(state.get("support_handoff_history", {}))
    result = history.get(key)
    if result is None:
        user_id = context.user_id if isinstance(context, ToolContext) else ""
        epoch = state.get("consultation_id", "initial")
        request_id = (
            "HANDOFF-" + uuid5(NAMESPACE_URL, f"{thread_id}:{user_id}:{epoch}:{key}").hex[:16]
        )
        response = HandoffResponse(request_id=request_id, status="error", simulated=True)
        if isinstance(context, ToolContext) and thread_id:
            try:
                response = HandoffResponse.model_validate(
                    gateway.transfer(
                        HandoffRequest(
                            request_id=request_id,
                            user_id=user_id,
                            issue=state.get("support_issue") or reason,
                            reason=reason,
                            attempted_solution=state.get("support_advice", {})
                            if topic == "quality"
                            else {},
                        )
                    )
                )
                if response.request_id != request_id:
                    raise ValueError("交接身份不匹配")
            except Exception:
                response = HandoffResponse(request_id=request_id, status="error", simulated=False)
        connected = response.status == "connected" and bool(response.agent_id)
        if not connected:
            response = response.model_copy(update={"status": "error", "agent_id": None})
        label = "【模拟客服】" if response.simulated else ""
        message = (
            f"{label}对接人工客服成功，由工号{response.agent_id}为你服务。"
            if connected
            else f"{label}转人工错误，请稍后重试或联系平台客服。"
        )
        result = {**response.model_dump(mode="json"), "message": message, "reason": reason}
        result["code"] = (
            state.get("return_error_code" if topic == "return" else "support_error_code") or None
        )
        if not isinstance(context, ToolContext) or not thread_id:
            result["message"] = "【模拟客服】转人工错误：缺少可信用户身份或会话标识。"
        history[key] = result
    return {
        "support_phase": "transferred" if result["status"] == "connected" else "transfer_failed",
        "support_handoff": result,
        "support_handoff_topic": topic,
        "support_handoff_history": history,
    }


def support_reply(state) -> str:
    """展示当前主题的确定性回复，不混用其他问题的旧结果。"""
    if state.get("support_handoff"):
        return state["support_handoff"]["message"]
    if state.get("support_result"):
        return state["support_result"]["message"]
    advice = state["support_advice"]
    # 不将原始RAG摘录回退为用户答案；诊断证据保存在state与工具artifact。
    return advice.get("customer_reply") or (
        advice["answer"]
        if advice.get("data_origin") != "ragflow_retrieval"
        else "暂时无法整理可靠的排查方案，请联系人工客服。"
    )


def create_support_tools(
    repository, gateway: HandoffGateway, *, order_repository=None
) -> list[BaseTool]:
    """通过官方工具和Command实现咨询、交接与结束，不持久化质量登记。"""

    def command(runtime, update, result):
        """提交官方状态增量和对应的工具消息。"""
        name = next(
            c["name"]
            for c in runtime.state["messages"][-1].tool_calls
            if c["id"] == runtime.tool_call_id
        )
        visible_result = result
        if result.data and result.data.get("action") in {"advice", "clarification"}:
            visible_result = result.model_copy(
                update={
                    "data": {
                        k: result.data[k]
                        for k in ("action", "message", "request_count", "missing_information")
                        if k in result.data
                    }
                }
            )
        return Command(
            update={
                **update,
                "messages": [
                    ToolMessage(
                        content=visible_result.model_dump_json(),
                        artifact=result.model_dump(mode="json"),
                        tool_call_id=runtime.tool_call_id,
                        name=name,
                        status="success" if result.ok else "error",
                    )
                ],
            }
        )

    def handoff(runtime, state, reason, *, topic="quality"):
        """业务已确定交接条件后请求客服，保留原因供测评核查。"""
        update = transfer_support(
            state,
            runtime.context,
            runtime.config.get("configurable", {}).get("thread_id", ""),
            reason,
            gateway,
            topic=topic,
        )
        if topic == "return":
            update["return_stage"] = "handoff"
        return command(
            runtime,
            {**state, **update},
            ToolResult(
                ok=update["support_phase"] == "transferred",
                data={
                    "action": "handoff",
                    "topic": topic,
                    "code": state.get(
                        "return_error_code" if topic == "return" else "support_error_code"
                    )
                    or None,
                    **update["support_handoff"],
                },
            ),
        )

    @tool("query_quality_support")
    def query_quality_support(
        category: str,
        issue: str,
        runtime: ToolRuntime[ToolContext],
        environment: str = "",
        order_id: str = "",
        is_question: bool = True,
        new_issue: bool = False,
    ) -> Command:
        """查询耳机、鼠标、键盘故障规则，允许多次有依据排查，不登记订单售后。

        category只能headphones/mice/keyboards；issue写具体症状；environment只填用户明确提供
        的型号、系统、连接方式，禁止猜测。订单号可选，提供时必须是用户已说过的本人订单。
        同一故障未解决时继续用本工具，不能仅因已给方案就人工；不能用商品工具找维修方案。
        首次问题及同问题继续求解is_question=true；只补信息或查询进度为false。
        明确换成新的独立故障才令new_issue=true，不清楚先询问，禁止为规避次数伪造新问题。
        程序计数：第4、5次建议人工，第6次自动人工；补信息不增加次数。
        相关规则缺必要信息时只问缺项；无适用规则、冲突或异常立即人工，不凭常识编方案。
        用户只看整理后的步骤和必要提醒；原文、文档名、分块ID及内部流程仅供诊断，不向用户展示。
        明确要人工用handoff_to_human(request_human=true)；明确退货用handle_return。
        已解决或结束咨询用end_support_session。本工具单独调用；已无效的方案不重复推荐。
        """
        query = SupportQuery(
            category=category,
            issue=issue,
            environment=environment,
            order_id=order_id,
            is_question=is_question,
            new_issue=new_issue,
        )
        state = dict(runtime.state)
        # 保留最近原始表述供规则选择核对，不另起问题分析模型。
        query = query.model_copy(
            update={
                "original_statements": [
                    m.text for m in state["messages"] if isinstance(m, HumanMessage)
                ][-2:]
            }
        )
        binding = order_id or ("" if new_issue else state.get("support_order_id", ""))
        if binding and order_repository is not None:
            supplied = any(
                isinstance(m, HumanMessage) and binding in m.text for m in state["messages"]
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
                    ToolResult.failure("ORDER_NOT_FOUND", "订单不存在或不属于当前用户。"),
                )
            prefix = {"headphones": "HP-P-", "mice": "MS-P-", "keyboards": "KB-P-"}[category]
            if not any(item.product_id.startswith(prefix) for item in owned.items):
                return command(
                    runtime,
                    {},
                    ToolResult.failure("ORDER_PRODUCT_MISMATCH", "订单与商品品类不符。"),
                )
        state.update(
            advance_question(
                state,
                "quality",
                f"{category}:{binding or '__unbound__'}",
                new_issue=new_issue,
                countable=is_question,
            )
        )
        state.update(
            support_order_id=binding,
            support_issue=issue,
            support_handoff={},
            support_result={},
        )
        key = state["support_problem_key"]
        if key != runtime.state.get("support_problem_key"):
            state["support_advice"] = {}
        if state.get("support_handoff_history", {}).get(f"quality:{key}"):
            return handoff(runtime, state, "当前问题已转人工，复用交接结果")
        if state["support_question_count"] >= AUTO_HANDOFF_FROM:
            return handoff(runtime, state, "同一质量问题第6次有效求解仍未解决，自动转人工")
        history = {k: list(v) for k, v in state.get("support_advice_history", {}).items()}
        attempts = history.get(key, [])
        query = query.model_copy(update={"attempted_solutions": attempts})
        state["support_error_code"] = ""
        try:
            knowledge = repository.find(query)
        except Exception as exc:
            code = (
                str(exc)
                if isinstance(exc, ValueError) and str(exc) in RAG_ERROR_MESSAGES
                else "KNOWLEDGE_SERVICE_ERROR"
            )
            state["support_error_code"] = code
            reason = RAG_ERROR_MESSAGES.get(code, "知识服务异常，无法核实适用规则")
            return handoff(runtime, state, f"{code}：{reason}")
        if knowledge is None:
            state["support_error_code"] = "KNOWLEDGE_NO_RULE"
            return handoff(runtime, state, "未查询到适用的故障知识")
        handling = knowledge.source.get("handling", "basic_troubleshooting")
        if handling not in {"basic_troubleshooting", "clarify"}:
            return handoff(runtime, state, "规则冲突或知识不支持自动排查")
        advice = knowledge.model_dump(mode="json")
        if handling == "clarify":
            if not knowledge.missing_information:
                return handoff(runtime, state, "相关规则缺少可核实的澄清条件")
            message = "还需要确认：" + "、".join(knowledge.missing_information) + "。"
            phase = "awaiting_information"
        elif knowledge.answer in attempts:
            message = (
                "这些步骤已经提供过，不重复推荐。"
                "请告诉我已尝试的步骤和结果，以便继续核查；也可选择人工。"
            )
            phase = "awaiting_information"
        else:
            message = knowledge.customer_reply.strip()
            if not message:
                if knowledge.data_origin == "ragflow_retrieval":
                    state["support_error_code"] = "RAG_REPLY_INVALID"
                    return handoff(runtime, state, "知识缺少可可靠展示的用户排查方案")
                # 本地摘要已有可直接使用的条件/安全步骤，不额外调用模型。
                message = knowledge.answer
            history[key] = [*attempts, knowledge.answer]
            phase = "advised"
        if SUGGEST_HANDOFF_FROM <= state["support_question_count"] < AUTO_HANDOFF_FROM:
            message += HANDOFF_SUGGESTION
        result = {
            "action": "advice" if phase == "advised" else "clarification",
            "message": message,
            "request_count": state["support_question_count"],
            **advice,
        }
        state.update(
            support_advice=advice,
            support_advice_history=history,
            support_phase=phase,
            support_result=result,
        )
        return command(runtime, state, ToolResult(ok=True, data=result))

    @tool("handoff_to_human")
    def handoff_to_human(
        reason: str,
        runtime: ToolRuntime[ToolContext],
        topic: str = "auto",
        request_human: bool = False,
    ) -> Command:
        """用户明确要求人工时立即交接，或复用当前问题已触发的交接。

        reason概括真实交接原因；topic为quality/return/auto，auto优先当前活动主题。
        明确要求人工或同意先前建议才request_human=true；未解决、坚持退货不等于同意。
        第6次阈值与无规则交接由质量/退货工具核实，不能仅填原因绕过业务工具。
        交接仅复用相同主题/问题；接口失败不说成功，不自动业务重试；单独调用、不执行退款。
        """
        args = HandoffQuery(reason=reason, topic=topic, request_human=request_human)
        state = dict(runtime.state)
        selected = (
            args.topic
            if args.topic != "auto"
            else state.get("active_support_topic")
            or ("return" if state.get("return_stage") else "quality")
        )
        count = state.get(
            "return_request_count" if selected == "return" else "support_question_count", 0
        )
        prefix = "return" if selected == "return" else "support"
        scope = state.get(f"{prefix}_problem_key") or state.get(f"{prefix}_order_id", "")
        key = f"{selected}:{scope}"
        if (
            not args.request_human
            and count < AUTO_HANDOFF_FROM
            and not state.get("support_handoff_history", {}).get(key)
        ):
            return command(
                runtime,
                {},
                ToolResult.failure(
                    "HANDOFF_NOT_READY",
                    "尚未满足人工条件，请调用质量/退货工具核实规则；未解决不等于请求人工。",
                ),
            )
        if selected == "return":
            state["support_issue"] = (
                f"退货订单={state.get('return_order_id', '')}；"
                f"原因={state.get('return_reason', '')}"
            )
        return handoff(runtime, state, args.reason, topic=selected)

    @tool("end_support_session")
    def end_support_session(runtime: ToolRuntime[ToolContext]) -> Command:
        """仅在用户明确表示已解决、结束咨询或不再需要帮助时结束当前咨询。

        引用、假设、否定的结束话术不能触发；单纯沉默、断网不能调用。
        清空问题计数、咨询上下文和待确认引导，绝不删除退货台账、不标记退款完成、不撤销平台申请。
        """
        return command(
            runtime,
            reset_consultation(),
            ToolResult(
                ok=True,
                data={
                    "action": "session_ended",
                    "message": "本次咨询已结束，提问次数已清零；已有退货记录保持不变。",
                },
            ),
        )

    return [query_quality_support, handoff_to_human, end_support_session]
