from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.config import get_config
from pydantic import ValidationError

from orderagent.schemas.order_schema import OrderQuery, ToolResult
from orderagent.schemas.product_schema import ProductQuery
from orderagent.schemas.return_schema import ReturnAgentState, ReturnQuery
from orderagent.schemas.support_schema import HandoffQuery, SupportQuery
from orderagent.tools.return_tools import return_handoff
from orderagent.tools.support_tools import support_reply, transfer_support


class BusinessToolPolicyMiddleware(AgentMiddleware):
    """限制可用业务工具、校验调用参数，并控制一次质量方案后的人工交接流程。"""

    state_schema = ReturnAgentState
    argument_schemas = {
        "query_order": OrderQuery,
        "query_products": ProductQuery,
        "query_quality_support": SupportQuery,
        "handoff_to_human": HandoffQuery,
        "handle_return": ReturnQuery,
    }

    def __init__(self, handoff_gateway, order_repository=None, after_sales=None):
        """注入与工具共用的客服接口，供明确反馈的快捷交接路径使用。"""
        self.handoff_gateway = handoff_gateway
        self.order_repository = order_repository
        self.after_sales = after_sales

    def _record(self, state, context, event_id, *, quality=False):
        """只为已核验归属的订单记录实际处理结果，匿名咨询不修改订单。"""
        order_id = (
            state.get("support_order_id")
            if quality
            else (state.get("return_order_id") or state.get("support_order_id"))
        )
        if not order_id or not context or not self.after_sales or not self.order_repository:
            return
        if self.order_repository.find_owned(order_id, context.user_id) is None:
            return
        status = {
            "advised": "已提供一次排查",
            "transferred": "已转人工（模拟）",
            "transfer_failed": "转人工失败",
        }.get(state.get("support_phase"))
        if not quality and state.get("return_stage") == "link_ready":
            status = "已提供退货测试入口"
        if status:
            self.after_sales.record(context.user_id, order_id, status, event_id)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        """在模型生成前展示已完成的质量结果，或处理明确未解决及人工请求。"""
        last = state["messages"][-1]
        if isinstance(last, ToolMessage) and last.name in {
            "query_quality_support",
            "handoff_to_human",
            "handle_return",
        }:
            if last.name != "handle_return" or state.get("return_stage") in {
                "handoff",
                "link_ready",
            }:
                data = (last.artifact or {}).get("data") or {}
                self._record(
                    state,
                    runtime.context,
                    str(last.id),
                    quality=last.name == "query_quality_support" or data.get("topic") == "quality",
                )
        if (
            isinstance(last, ToolMessage)
            and last.name == "handle_return"
            and last.artifact
            and (last.artifact.get("data") or {}).get("action") == "return_result"
        ):
            return {
                "messages": [AIMessage(content=last.artifact["data"]["message"])],
                "jump_to": "end",
            }
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
        # 只对完整且无歧义的短句快捷交接，复杂表达交模型判断。
        text = last.text.strip().rstrip("。！？.!?")
        if state.get("return_stage") and text in {
            "不退了",
            "不用退货了",
            "不退货了",
            "取消退货",
            "取消退货申请",
        }:
            return {
                "return_stage": None,
                "return_order_id": "",
                "return_reason": "",
                "return_result": {},
                "return_rule_source": {},
                "messages": [AIMessage(content="已取消本次机器人退货引导，未提交平台申请。")],
                "jump_to": "end",
            }
        explicit = text in ("转人工", "人工客服", "找客服", "请转人工")
        unsuccessful = state.get("support_phase") == "advised" and text in (
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
        )
        if not (explicit or unsuccessful):
            return None
        if state.get("return_stage") and explicit:
            update = return_handoff(
                state,
                runtime.context,
                get_config().get("configurable", {}).get("thread_id", ""),
                self.handoff_gateway,
            )
            self._record({**state, **update}, runtime.context, str(last.id))
            return {
                **update,
                "messages": [AIMessage(content=support_reply({**state, **update}))],
                "jump_to": "end",
            }
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
        self._record({**state, **update}, runtime.context, str(last.id), quality=True)
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
                "UNKNOWN_TOOL", "当前仅支持订单、商品、质量查询、退货引导和人工交接。"
            )
        else:
            try:
                schema.model_validate(request.tool_call["args"])
                calls = request.state["messages"][-1].tool_calls
                if len(calls) > 1 and any(
                    call["name"] in {"query_quality_support", "handoff_to_human", "handle_return"}
                    for call in calls
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
