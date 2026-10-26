import json
from datetime import datetime, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from orderagent.agents.agent_factory import create_order_agent
from orderagent.integrations.returns import MockReturnGateway
from orderagent.repositories.after_sales_repository import AfterSalesRepository
from orderagent.repositories.support_repository import JsonSupportRepository
from orderagent.schemas.order_schema import ToolContext
from orderagent.schemas.return_schema import ReturnQuery
from orderagent.schemas.support_schema import SupportKnowledge
from orderagent.settings import ConfigurationError, Settings
from tests.fake_chat_model import ScriptedChatModel, tool_call
from tests.test_support_tools import RecordingGateway

NOW = datetime.fromisoformat("2026-10-23T12:00:00+08:00")
USER = ToolContext(user_id="RETURN-U-001")
ORDER = "RETURN-HP-O-0001"


@pytest.fixture
def order_rows():
    """小型订单集隔离退货规则，不依赖批量数据中的偶然状态。"""
    base = {
        "id": ORDER,
        "user_id": USER.user_id,
        "synthetic": True,
        "order_status": "delivered",
        "order_status_label": "已签收",
        "shipping_status": "已发货",
        "logistics_status": "已签收",
        "tracking_number": "RETURN-TRACK-001",
        "snapshot_at": NOW.isoformat(),
        "delivered_at": "2026-10-22T09:00:00+08:00",
        "items": [
            {
                "product_id": "HP-P-0001",
                "source_sku": "1019184",
                "name": "模拟耳机",
                "model": "TEST",
                "quantity": 1,
            }
        ],
    }
    return [
        base,
        {**base, "id": "RETURN-HP-O-0002"},
        {**base, "id": "RETURN-HP-O-0003", "user_id": "OTHER"},
    ]


class TestReturnRules:
    """仅供工具路径测试的显式假规则，不代表真实售后政策。"""

    __test__ = False

    def find(self, query):
        if query.issue.startswith("退货规则与适用条件"):
            return SupportKnowledge(
                id="TEST-RETURN-RULE",
                category=query.category,
                question=query.issue,
                answer="测试规则：用户描述的问题支持提供模拟申请入口。",
                applicability="自动化测试夹具，不用于真实业务",
                source={"handling": "return_eligible", "reference_text": "测试规则"},
                data_origin="synthetic_test_fixture",
            )
        return JsonSupportRepository([Settings().support_path]).find(query)


@pytest.fixture
def graph_builder(tmp_path, order_rows):
    """用已有官方图、假模型、临时订单及固定时钟验证真实工具路径。"""

    def build(
        responses,
        *,
        scenario="unknown",
        gateway=None,
        handoff=None,
        knowledge=None,
        clock=None,
        **settings,
    ):
        path = tmp_path / "orders.json"
        path.write_text(json.dumps(order_rows), encoding="utf-8")
        model = ScriptedChatModel(responses=responses)
        graph = create_order_agent(
            Settings(
                orders_path=path,
                mock_return_scenario=scenario,
                after_sales_db_path=tmp_path / "state.sqlite3",
                **settings,
            ),
            model=model,
            return_gateway=gateway,
            handoff_gateway=handoff,
            return_clock=clock if clock is not None else lambda: NOW,
            support_repository=knowledge if knowledge is not None else TestReturnRules(),
        )
        return graph, model

    return build


def call(order_id=ORDER, reason="不喜欢佩戴感受", **kwargs):
    """构造模型提出的退货工具调用。"""
    return tool_call("handle_return", {"order_id": order_id, "reason": reason, **kwargs})


def ask(graph, text, thread="return", context=USER):
    """提交同一会话的一轮消息。"""
    return graph.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread}},
        context=context,
    )


@pytest.mark.parametrize(
    "arguments,missing",
    [
        ({}, ["订单号", "退货原因"]),
        ({"order_id": ORDER}, ["退货原因"]),
        ({"reason": "不喜欢"}, ["订单号"]),
        ({"order_id": " ", "reason": " "}, ["订单号", "退货原因"]),
    ],
)
def test_missing_information_returns_guidance_without_model_rewrite(
    graph_builder, arguments, missing
):
    graph, model = graph_builder([tool_call("handle_return", arguments)])
    state = ask(graph, "我要退货")
    assert state["return_stage"] == "collecting"
    assert state["return_result"]["missing_fields"] == missing
    assert "我的订单" in state["messages"][-1].text
    assert all(field in state["messages"][-1].text for field in missing)
    assert len(model._requests) == 1


def test_multi_turn_information_completion_is_not_mistaken_for_insistence(graph_builder):
    graph, model = graph_builder(
        [
            tool_call("handle_return", {}),
            tool_call("handle_return", {"order_id": ORDER}),
            tool_call("handle_return", {"reason": "不喜欢"}),
        ]
    )
    ask(graph, "我要退货")
    ask(graph, f"订单号是{ORDER}")
    state = ask(graph, "原因是不喜欢")
    assert state["return_stage"] == "awaiting_confirmation"
    assert state["return_order_id"] == ORDER and state["return_reason"] == "不喜欢"
    assert "support_handoff" not in state
    assert len(model._requests) == 3


@pytest.mark.parametrize("reason", ["不喜欢", "左耳没有声音"])
def test_eligibility_not_quality_keyword_controls_link(graph_builder, reason):
    graph, model = graph_builder([call(reason=reason), call(reason=reason)], scenario="eligible")
    pending = ask(graph, "订单" + ORDER + "我要退货，原因是" + reason)
    assert pending["return_stage"] == "awaiting_confirmation"
    assert "url" not in pending["return_result"]
    state = ask(graph, f"确认退货 {ORDER}")
    assert state["return_stage"] == "link_ready"
    assert state["return_result"]["url"].startswith("https://returns.example.invalid/")
    assert state["return_result"]["simulated"]
    assert "不可提交真实" in state["messages"][-1].text
    assert "不代表申请已提交" in state["messages"][-1].text
    assert len(model._requests) == 2


@pytest.mark.parametrize("scenario", ["unknown", "eligible"])
def test_mock_evidence_scenario_does_not_gate_rag_supported_return(graph_builder, scenario):
    graph, _ = graph_builder([call(), call()], scenario=scenario)
    pending = ask(graph, "我要退货")
    assert pending["return_stage"] == "awaiting_confirmation"
    state = ask(graph, f"确认退货 {ORDER}")
    assert state["return_stage"] == "link_ready"


@pytest.mark.parametrize("text", ["我还是要退货", "不想去后台，你来办", "我坚持售后", "请转人工"])
def test_return_repetition_waits_until_fourth_request_or_explicit_human(graph_builder, text):
    handoff = RecordingGateway()
    graph, model = graph_builder([call()], handoff=handoff)
    ask(graph, "我要退货")
    if text == "请转人工":
        state = ask(graph, text)
        assert len(model._requests) == 1
    else:
        for count in (2, 3):
            state = ask(graph, text)
            assert state["return_request_count"] == count
            assert state["return_stage"] == "awaiting_confirmation"
            assert not handoff.requests
        state = ask(graph, text)
        assert state["return_request_count"] == 4
    assert state["return_stage"] == "handoff"
    assert state["support_phase"] == "transferred"
    assert "工号001" in state["messages"][-1].text
    assert ORDER in handoff.requests[0].issue
    assert "不喜欢佩戴感受" in handoff.requests[0].issue
    again = ask(graph, "还是要退货")
    assert again["support_handoff"] == state["support_handoff"]
    assert len(handoff.requests) == 1


def test_changed_reason_does_not_reset_request_count(graph_builder):
    graph, _ = graph_builder([call(), call(reason="换一个理由")])
    ask(graph, "我要退货")
    state = ask(graph, "请再处理一次")
    assert state["return_stage"] == "awaiting_confirmation"
    assert state["return_request_count"] == 2
    ask(graph, "我仍要退货")
    state = ask(graph, "我仍要退货")
    assert state["return_stage"] == "handoff"
    assert state["return_request_count"] == 4


def test_unusual_insistence_is_routed_by_model_flag(graph_builder):
    graph, _ = graph_builder([call(), call(request_human=True)])
    ask(graph, "我要退货")
    state = ask(graph, "这件事请让人工客服处理")
    assert state["return_stage"] == "handoff"


def test_missing_information_repeated_waits_for_four_requests(graph_builder):
    graph, _ = graph_builder([call(order_id="", reason=""), call(order_id="", reason="")])
    ask(graph, "我要退货")
    state = ask(graph, "订单号我找不到了，你帮忙处理")
    assert state["return_stage"] == "collecting"
    assert state["return_request_count"] == 2
    ask(graph, "我仍要求退货")
    state = ask(graph, "我仍要求退货")
    assert state["return_stage"] == "handoff"


def test_cancelled_request_does_not_trigger_handoff(graph_builder):
    graph, model = graph_builder([call(), AIMessage(content="好的。")])
    ask(graph, "我要退货")
    state = ask(graph, "不用退货了")
    assert state["return_stage"] is None
    assert "support_handoff" not in state
    assert len(model._requests) == 1


def test_new_order_clears_previous_reason_and_checks_ownership(graph_builder):
    graph, _ = graph_builder([call(), call(order_id="RETURN-HP-O-0002", reason="")])
    ask(graph, "我要退货")
    state = ask(graph, "换一个订单RETURN-HP-O-0002退货")
    assert state["return_stage"] == "collecting"
    assert state["return_reason"] == ""
    assert state["return_order_id"] == "RETURN-HP-O-0002"


@pytest.mark.parametrize("order_id", ["RETURN-HP-O-0003", "RETURN-HP-O-9999"])
def test_absent_and_foreign_orders_share_response(graph_builder, order_id):
    graph, _ = graph_builder([call(order_id=order_id)], scenario="eligible")
    state = ask(graph, "我要退货")
    assert state["return_result"]["code"] == "ORDER_NOT_FOUND"
    assert "url" not in state["return_result"]
    assert "OTHER" not in state["messages"][-1].text


def test_missing_trusted_context_is_rejected(graph_builder):
    graph, _ = graph_builder([call()], scenario="eligible")
    state = ask(graph, "我要退货", context=None)
    assert "缺少可信用户身份" in state["messages"][-1].text
    assert "url" not in state["return_result"]


def test_handoff_failure_not_success(graph_builder):
    graph, _ = graph_builder([call()], handoff=RecordingGateway(fail=True))
    ask(graph, "我要退货")
    state = ask(graph, "请转人工")
    assert state["support_phase"] == "transfer_failed"
    assert "转人工错误" in state["messages"][-1].text


def test_quality_advice_followed_by_explicit_return_starts_return_counter(graph_builder):
    graph, model = graph_builder(
        [
            tool_call("query_quality_support", {"category": "headphones", "issue": "左耳没有声音"}),
            call(),
        ]
    )
    ask(graph, "耳机左耳没有声音")
    state = ask(graph, "我仍要求售后退货")
    assert state["support_phase"] == "advised"
    assert state["return_stage"] == "awaiting_confirmation"
    assert state["return_request_count"] == 1
    assert len(model._requests) == 2


def test_first_quality_question_after_return_guidance_still_gets_one_advice(graph_builder):
    graph, _ = graph_builder(
        [
            call(),
            tool_call("query_quality_support", {"category": "headphones", "issue": "左耳没有声音"}),
        ]
    )
    ask(graph, "我要退货")
    state = ask(graph, "再给个解决办法")
    assert state["return_stage"] == "awaiting_confirmation"
    assert state["support_phase"] == "advised"
    assert state["support_advice"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "http://returns.example.invalid/x"),
        ("url", "https://evil.example/x"),
        ("url", "https://user:pass@returns.example.invalid/x"),
        ("order_id", "OTHER"),
        ("user_id", "OTHER"),
        ("request_id", "OTHER"),
        ("expires_at", NOW - timedelta(seconds=1)),
        ("expires_at", NOW.replace(tzinfo=None)),
        ("status", "error"),
    ],
)
def test_invalid_link_response_rejected(graph_builder, field, value):
    class BadGateway(MockReturnGateway):
        def create_link(self, request, now):
            return super().create_link(request, now).model_copy(update={field: value})

    graph, _ = graph_builder([call(), call()], gateway=BadGateway("eligible"))
    ask(graph, "我要退货")
    state = ask(graph, f"确认退货 {ORDER}")
    assert state["return_stage"] == "guided"
    assert "入口生成失败" in state["messages"][-1].text
    assert "url" not in state["return_result"]


def test_link_exception_is_sanitized(graph_builder):
    class BrokenGateway(MockReturnGateway):
        def create_link(self, request, now):
            raise RuntimeError("private-return-platform-secret")

    graph, _ = graph_builder([call(), call()], gateway=BrokenGateway("eligible"))
    ask(graph, "我要退货")
    state = ask(graph, f"确认退货 {ORDER}")
    assert "private-return-platform-secret" not in str(state)
    assert state["return_stage"] == "guided"


@pytest.mark.parametrize(
    "extra", [{"user_id": "OTHER"}, {"eligible": True}, {"url": "https://evil.example"}]
)
def test_model_cannot_supply_identity_eligibility_or_link(graph_builder, extra):
    graph, _ = graph_builder(
        [
            tool_call("handle_return", {"order_id": ORDER, "reason": "不喜欢", **extra}),
            AIMessage(content="参数错误。"),
        ]
    )
    state = ask(graph, "我要退货")
    message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert message.artifact["error"]["code"] == "INVALID_ARGUMENTS"
    assert "return_result" not in state


def test_return_parameters_are_strict():
    assert ReturnQuery().order_id == ""
    for arguments in ({"order_id": 1}, {"reason": "x" * 501}, {"request_human": "true"}):
        with pytest.raises(ValidationError):
            ReturnQuery.model_validate(arguments)


def test_return_configuration_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("MOCK_RETURN_SCENARIO", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "MOCK_RETURN_SCENARIO=eligible\nRETURN_SELF_SERVICE_ENTRY=Orders / Return\n",
        encoding="utf-8",
    )
    assert Settings.load(env).mock_return_scenario == "eligible"
    assert Settings.load(env).return_self_service_entry == "Orders / Return"
    monkeypatch.setenv("MOCK_RETURN_SCENARIO", "unknown")
    assert Settings.load(env).mock_return_scenario == "unknown"
    monkeypatch.setenv("MOCK_RETURN_SCENARIO", "bad")
    with pytest.raises(ConfigurationError):
        Settings.load(env)


def test_parallel_return_calls_are_rejected(graph_builder):
    first = call()
    second = call()
    second.tool_calls[0]["id"] = "other"
    first.tool_calls.extend(second.tool_calls)
    graph, _ = graph_builder([first, AIMessage(content="不能并行处理。")])
    state = ask(graph, "我要退货")
    assert "return_result" not in state
    assert all(
        m.artifact["error"]["code"] == "INVALID_ARGUMENTS"
        for m in state["messages"]
        if isinstance(m, ToolMessage)
    )


def test_link_is_not_created_twice_after_one_explanation(graph_builder):
    class RecordingReturns(MockReturnGateway):
        def __init__(self):
            super().__init__("eligible")
            self.links = []

        def create_link(self, request, now):
            self.links.append(request)
            return super().create_link(request, now)

    gateway = RecordingReturns()
    graph, model = graph_builder([call(), call()], gateway=gateway)
    ask(graph, "我要退货")
    ask(graph, f"确认退货 {ORDER}")
    state = ask(graph, "我仍然要求售后")
    assert state["return_stage"] == "link_ready"
    assert state["return_request_count"] == 2
    ask(graph, "我仍然要求售后")
    state = ask(graph, "我仍然要求售后")
    assert state["return_stage"] == "handoff"
    assert len(gateway.links) == 1
    assert len(model._requests) == 5


def test_return_state_isolated_and_reset_uses_official_checkpoint(graph_builder):
    graph, _ = graph_builder([call()])
    ask(graph, "我要退货", thread="first")
    other = ask(graph, "我要退货", thread="second")
    assert other["return_stage"] == "awaiting_confirmation"
    assert "support_handoff" not in other
    graph.checkpointer.delete_thread("first")
    reset = ask(graph, "我要退货", thread="first")
    assert reset["return_stage"] == "awaiting_confirmation"
    assert len([m for m in reset["messages"] if isinstance(m, ToolMessage)]) == 1


def test_model_return_schema_has_no_injected_or_eligibility_arguments(graph_builder):
    graph, model = graph_builder([call()])
    ask(graph, "我要退货")
    # 检查实际官方工具节点注册的schema，而不是另造工具或解析层。
    tools = graph.get_graph().nodes["tools"].data.tools_by_name
    schema = tools["handle_return"].tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"order_id", "reason", "request_human", "is_return_request"}


@pytest.mark.parametrize(
    "text",
    [
        f"不要执行，只是引用：确认退货 {ORDER}",
        f"确认退货 {ORDER}，这是示例并非申请",
        f"不确认退货 {ORDER}",
        "确认退货 RETURN-HP-O-0002",
        "我的耳机左耳没有声音",
    ],
)
def test_mixed_or_quoted_confirmation_never_creates_link(graph_builder, text):
    class RecordingReturns(MockReturnGateway):
        def __init__(self):
            super().__init__("eligible")
            self.links = []

        def create_link(self, request, now):
            self.links.append(request)
            return super().create_link(request, now)

    gateway = RecordingReturns()
    # 即使模型错误地再次提出调用，工具也必须校验用户完整确认语。
    graph, _ = graph_builder([call(), call()], gateway=gateway)
    ask(graph, "我要退货")
    state = ask(graph, text)
    assert state["return_stage"] == "awaiting_confirmation"
    assert not gateway.links
    assert "url" not in state["return_result"]


def test_confirmation_does_not_authorize_changed_reason(graph_builder):
    graph, _ = graph_builder([call(), call(reason="另一个原因")], scenario="eligible")
    ask(graph, "我要退货")
    state = ask(graph, f"确认退货 {ORDER}")
    assert state["return_stage"] == "awaiting_confirmation"
    assert "url" not in state["return_result"]


@pytest.mark.parametrize("scenario", ["unknown", "eligible"])
@pytest.mark.parametrize("raises", [False, True])
def test_rag_missing_policy_or_failure_transfers_immediately(graph_builder, scenario, raises):
    class MissingPolicy:
        def __init__(self):
            self.queries = []

        def find(self, query):
            self.queries.append(query)
            if raises:
                raise RuntimeError("private-rag-error")
            return None

    repository = MissingPolicy()
    gateway = RecordingGateway()
    graph, _ = graph_builder(
        [call()],
        scenario=scenario,
        handoff=gateway,
        knowledge=repository,
        ragflow_api_url="http://localhost",
        ragflow_api_key="test-key",
    )
    state = ask(graph, f"订单{ORDER}我要退货，原因是不喜欢")
    assert repository.queries
    assert state["return_stage"] == "handoff"
    assert state["return_request_count"] == 1
    assert not state["return_result"].get("url")
    assert len(gateway.requests) == 1
    assert "private-rag-error" not in str(state)


def test_cancellation_invalidates_pending_confirmation(graph_builder):
    graph, _ = graph_builder([call(), call()], scenario="eligible")
    ask(graph, "我要退货")
    cancelled = ask(graph, "取消退货")
    assert cancelled["return_stage"] is None
    state = ask(graph, f"确认退货 {ORDER}")
    assert state["return_stage"] == "awaiting_confirmation"
    assert "url" not in state["return_result"]


def test_supplement_and_confirmation_do_not_increment_request_counter(graph_builder):
    graph, _ = graph_builder(
        [
            call(order_id="", reason=""),
            call(reason="", is_return_request=False),
            call(is_return_request=False),
            call(is_return_request=False),
        ],
        scenario="eligible",
    )
    first = ask(graph, "我要退货")
    assert first["return_request_count"] == 1
    bound = ask(graph, f"订单号{ORDER}")
    assert bound["return_request_count"] == 1
    pending = ask(graph, "原因是不喜欢佩戴感受")
    assert pending["return_request_count"] == 1
    assert pending["return_stage"] == "awaiting_confirmation"
    linked = ask(graph, f"确认退货 {ORDER}")
    assert linked["return_request_count"] == 1
    assert linked["return_stage"] == "link_ready"


def test_same_human_message_is_counted_only_once(graph_builder):
    graph, _ = graph_builder([call()])
    for _ in range(4):
        state = graph.invoke(
            {"messages": [HumanMessage(content="我要退货", id="one-user-message")]},
            config={"configurable": {"thread_id": "dedup"}},
            context=USER,
        )
        assert state["return_request_count"] == 1
        assert state["return_stage"] == "awaiting_confirmation"


def test_switching_orders_keeps_separate_counts_and_restores_previous_count(graph_builder):
    second = "RETURN-HP-O-0002"
    gateway = RecordingGateway()
    orders = [ORDER, second, ORDER, second, ORDER, ORDER]
    graph, _ = graph_builder([call(order_id=value) for value in orders], handoff=gateway)
    for value, expected in zip(orders, [1, 1, 2, 2, 3, 4], strict=True):
        state = ask(graph, f"我要退货，订单{value}")
        assert state["return_request_count"] == expected
        assert state["return_stage"] == ("handoff" if expected == 4 else "awaiting_confirmation")
    assert len(gateway.requests) == 1


def test_cancelling_does_not_erase_same_order_request_history(graph_builder):
    graph, _ = graph_builder([call()])
    ask(graph, "我要退货")
    ask(graph, "取消退货")
    state = ask(graph, f"我又要退货，订单{ORDER}")
    assert state["return_request_count"] == 2
    assert state["return_stage"] == "awaiting_confirmation"


def test_model_cannot_use_return_handoff_tool_to_skip_counter(graph_builder):
    gateway = RecordingGateway()
    graph, model = graph_builder(
        [
            call(),
            tool_call("handoff_to_human", {"reason": "用户坚持退货", "topic": "return"}),
            call(),
        ],
        handoff=gateway,
    )
    ask(graph, "我要退货")
    state = ask(graph, "我还是要退货")
    assert state["return_request_count"] == 2
    assert state["return_stage"] == "awaiting_confirmation"
    assert not gateway.requests
    errors = [
        message
        for message in state["messages"]
        if isinstance(message, ToolMessage) and message.status == "error"
    ]
    assert any(
        message.artifact.get("error", {}).get("code") == "RETURN_HANDOFF_NOT_READY"
        for message in errors
    )
    assert len(model._requests) == 3


def test_first_simple_return_request_cannot_skip_counter_with_auto_handoff(graph_builder):
    gateway = RecordingGateway()
    graph, _ = graph_builder(
        [
            tool_call("handoff_to_human", {"reason": "用户要退货"}),
            call(),
        ],
        handoff=gateway,
    )
    state = ask(graph, "我要退货")
    assert state["return_request_count"] == 1
    assert state["return_stage"] == "awaiting_confirmation"
    assert not gateway.requests


def test_complex_explicit_human_request_uses_model_classification(graph_builder):
    gateway = RecordingGateway()
    graph, _ = graph_builder(
        [
            call(),
            tool_call(
                "handoff_to_human",
                {
                    "reason": "用户明确要求人工客服介入",
                    "topic": "return",
                    "request_human": True,
                },
            ),
        ],
        handoff=gateway,
    )
    ask(graph, "我要退货")
    state = ask(graph, "这个退货能否安排一位人工客服来处理？")
    assert state["return_stage"] == "handoff"
    assert state["return_request_count"] == 1
    assert len(gateway.requests) == 1


@pytest.mark.parametrize("status", ["已提供一次排查", "已转人工（模拟）"])
def test_persisted_status_alone_does_not_trigger_immediate_return_handoff(
    graph_builder, tmp_path, status
):
    gateway = RecordingGateway()
    graph, _ = graph_builder([call()], scenario="eligible", handoff=gateway)
    AfterSalesRepository(tmp_path / "state.sqlite3").record(
        USER.user_id, ORDER, status, "old-event"
    )
    state = ask(graph, f"我要退货，订单{ORDER}")
    assert state["return_request_count"] == 1
    assert state["return_stage"] == ("awaiting_confirmation")
    assert not gateway.requests


def test_different_order_after_handoff_does_not_reuse_previous_issue(graph_builder):
    gateway = RecordingGateway()
    graph, _ = graph_builder([call(), call(order_id="RETURN-HP-O-0002")], handoff=gateway)
    ask(graph, "我要退货")
    ask(graph, "请转人工")
    state = ask(graph, "换一单RETURN-HP-O-0002退货，原因是不喜欢")
    assert state["return_stage"] == "awaiting_confirmation"
    assert not state["support_handoff"]
    ask(graph, "请转人工")
    assert len(gateway.requests) == 2
    assert gateway.requests[0].request_id != gateway.requests[1].request_id
    assert "RETURN-HP-O-0002" in gateway.requests[1].issue
