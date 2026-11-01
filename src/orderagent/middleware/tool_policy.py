from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, ModelResponse, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.config import get_config
from pydantic import BaseModel, ConfigDict, ValidationError

from orderagent.schemas.order_schema import OrderQuery, ToolResult
from orderagent.schemas.product_schema import ProductQuery
from orderagent.schemas.return_schema import ReturnAgentState, ReturnQuery
from orderagent.schemas.support_schema import HandoffQuery, SupportQuery
from orderagent.tools.session_policy import (
    AUTO_HANDOFF_FROM,
    HANDOFF_SUGGESTION,
    SUGGEST_HANDOFF_FROM,
    advance_question,
    reset_consultation,
)
from orderagent.tools.support_tools import support_reply, transfer_support


class EndSessionQuery(BaseModel):
    """结束会话无业务输入，不允许模型携带订单状态或退款参数。"""

    model_config = ConfigDict(extra="forbid")


class BusinessToolPolicyMiddleware(AgentMiddleware):
    """复用官方生命周期做工具边界校验与确定性结果展示，不登记质量咨询。"""

    state_schema = ReturnAgentState
    argument_schemas = {
        "query_order": OrderQuery,
        "query_products": ProductQuery,
        "query_quality_support": SupportQuery,
        "handoff_to_human": HandoffQuery,
        "handle_return": ReturnQuery,
        "end_support_session": EndSessionQuery,
    }

    def __init__(self, handoff_gateway, order_repository=None, after_sales=None):
        self.handoff_gateway = handoff_gateway
        self.order_repository = order_repository
        self.after_sales = after_sales

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        """展示工具回复；完整结束短句清零，完整人工请求交接，其余语义交给模型工具。"""
        last = state["messages"][-1]
        if isinstance(last, ToolMessage) and last.artifact:
            data = last.artifact.get("data") or {}
            if data.get("action") in {
                "return_result",
                "advice",
                "clarification",
                "handoff",
                "session_ended",
            }:
                message = data.get("message") or support_reply(state)
                return {"messages": [AIMessage(content=message)], "jump_to": "end"}
        if not isinstance(last, HumanMessage):
            return None
        text = last.text.strip().rstrip("。！？.!?")
        if text in {
            "解决了",
            "已解决",
            "问题解决了",
            "不用了",
            "结束咨询",
            "结束本次咨询",
            "不需要帮助了",
        }:
            return {
                **reset_consultation(),
                "messages": [
                    AIMessage(content="本次咨询已结束，提问次数已清零；已有退货记录保持不变。")
                ],
                "jump_to": "end",
            }
        if text in {"不退了", "不用退货了", "不退货了", "取消退货", "取消退货申请"}:
            order_id = state.get("return_order_id")
            if order_id and runtime.context and self.order_repository and self.after_sales:
                if self.order_repository.find_owned(order_id, runtime.context.user_id):
                    self.after_sales.record(
                        runtime.context.user_id, order_id, "退货引导已取消", str(last.id)
                    )
            return {
                **reset_consultation(),
                "messages": [
                    AIMessage(
                        content="已结束本次机器人退货引导，咨询次数已清零；未撤销或删除已有平台/模拟申请记录。"
                    )
                ],
                "jump_to": "end",
            }
        if (
            state.get("active_support_topic") == "quality"
            and state.get("support_question_scope")
            and text
            in {
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
                "试了没用",
                "方案无效",
            }
        ):
            cached = state.get("support_handoff_history", {}).get(
                f"quality:{state.get('support_problem_key', '')}"
            )
            if cached:
                return {"messages": [AIMessage(content=cached["message"])], "jump_to": "end"}
            progress = advance_question(state, "quality", state["support_question_scope"])
            combined = {**state, **progress}
            if progress["support_question_count"] >= AUTO_HANDOFF_FROM:
                update = transfer_support(
                    combined,
                    runtime.context,
                    get_config().get("configurable", {}).get("thread_id", ""),
                    "同一质量问题第6次明确未解决反馈，自动转人工",
                    self.handoff_gateway,
                )
                return {
                    **progress,
                    **update,
                    "messages": [AIMessage(content=update["support_handoff"]["message"])],
                    "jump_to": "end",
                }
            # 先记次数再交模型处理；后续工具通过同一消息ID去重，直接补问也不会漏数。
            return progress
        if text not in {"转人工", "人工客服", "找客服", "请转人工"}:
            return None
        topic = state.get("active_support_topic") or (
            "return" if state.get("return_stage") else "quality"
        )
        issue = (
            f"退货订单={state.get('return_order_id', '')}；原因={state.get('return_reason', '')}"
            if topic == "return"
            else state.get("support_issue") or last.text
        )
        update = transfer_support(
            {**state, "support_issue": issue},
            runtime.context,
            get_config().get("configurable", {}).get("thread_id", ""),
            "用户明确要求人工",
            self.handoff_gateway,
            topic=topic,
        )
        if topic == "return":
            update["return_stage"] = "handoff"
        return {
            **update,
            "messages": [AIMessage(content=update["support_handoff"]["message"])],
            "jump_to": "end",
        }

    def after_model(self, state, runtime):
        """模型直接补问而无工具时，仍补齐已计数第4—5次的人工建议。"""
        last = state["messages"][-1]
        latest = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        topic = state.get("active_support_topic")
        count = state.get(
            "support_question_count" if topic == "quality" else "return_request_count", 0
        )
        counted = bool(
            latest and state.get("question_processed_messages", {}).get(latest.id) == topic
        )
        if (
            isinstance(last, AIMessage)
            and not last.tool_calls
            and counted
            and SUGGEST_HANDOFF_FROM <= count < AUTO_HANDOFF_FROM
            and "建议转人工" not in last.text
        ):
            return {
                "messages": [last.model_copy(update={"content": last.text + HANDOFF_SUGGESTION})]
            }
        return None

    def wrap_model_call(self, request, handler):
        """严格待确认指令用官方ModelResponse短路；其余调用白名单模型，工具仍经原节点。"""
        state = request.state
        last = state["messages"][-1]
        if isinstance(last, HumanMessage):
            # 仅确定性确认省去主模型；执行仍经官方tools节点和现有业务工具。
            confirmation = last.text.strip()
            if confirmation.endswith(("。", ".", "！", "!")):
                confirmation = confirmation[:-1]
            order_id = state.get("return_order_id", "")
            product_id = state.get("return_target_product_id", "")
            reason = state.get("return_reason", "")
            facts = state.get("return_reported_facts", {})
            valid_facts = isinstance(facts, dict) and all(
                isinstance(fact, dict)
                and isinstance(fact.get("value"), str)
                and fact.get("status") in {"explicit", "missing", "ambiguous", "conflict"}
                for fact in facts.values()
            )
            binding = {
                "order_id": order_id,
                "product_id": product_id,
                "reason": reason,
                "facts": {
                    key: {"value": fact["value"], "status": fact["status"]}
                    for key, fact in (facts.items() if valid_facts else [])
                },
            }
            if (
                state.get("return_stage") == "awaiting_confirmation"
                and state.get("return_has_request")
                and order_id
                and product_id
                and reason
                and valid_facts
                and state.get("return_confirmation_payload") == binding
                and confirmation == f"确认退货 {order_id}"
            ):
                return ModelResponse(
                    result=[
                        AIMessage(
                            content="",
                            name="deterministic_return_confirmation",
                            tool_calls=[
                                {
                                    "name": "handle_return",
                                    "args": {
                                        "order_id": order_id,
                                        "reason": reason,
                                        "target_product_id": product_id,
                                        "turn_kind": "confirm",
                                        "is_return_request": False,
                                    },
                                    "id": f"confirm-{uuid4().hex}",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ]
                )
        tools = [t for t in request.tools if getattr(t, "name", None) in self.argument_schemas]
        return handler(request.override(tools=tools))

    def wrap_tool_call(self, request, handler):
        """参数、工具白名单及业务操作不并行边界保持原校验。"""
        schema = self.argument_schemas.get(request.tool_call["name"])
        if schema is None:
            result = ToolResult.failure("UNKNOWN_TOOL", "当前仅支持已注册的订单、商品和售后工具。")
        else:
            try:
                schema.model_validate(request.tool_call["args"])
                calls = request.state["messages"][-1].tool_calls
                if len(calls) > 1 and any(
                    c["name"]
                    in {
                        "query_quality_support",
                        "handoff_to_human",
                        "handle_return",
                        "end_support_session",
                    }
                    for c in calls
                ):
                    raise ValueError("业务状态操作不能并行")
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
