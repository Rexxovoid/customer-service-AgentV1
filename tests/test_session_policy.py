import json
import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from orderagent.agents.agent_factory import create_order_agent
from orderagent.repositories.after_sales_repository import AfterSalesRepository
from orderagent.repositories.order_repository import JsonOrderRepository
from orderagent.repositories.support_repository import JsonSupportRepository
from orderagent.schemas.order_schema import ToolContext
from orderagent.schemas.support_schema import SupportKnowledge, SupportQuery
from orderagent.settings import Settings
from orderagent.tools.session_policy import advance_question, reset_consultation
from tests.fake_chat_model import ScriptedChatModel, tool_call
from tests.test_return_tools import ORDER, USER, ask, call
from tests.test_return_tools import graph_builder as graph_builder
from tests.test_return_tools import order_rows as order_rows
from tests.test_support_tools import RecordingGateway


class Rules:
    """显式受控规则，不代表真实RAG服务结果。"""

    def __init__(self, handling="basic_troubleshooting", missing=None):
        self.handling = handling
        self.missing = missing or []
        self.queries = []

    def find(self, query):
        self.queries.append(query)
        return SupportKnowledge(
            id="POLICY-FIXTURE",
            category=query.category,
            question=query.issue,
            answer="有声道平衡设置时检查是否居中。",
            applicability="仅供受控测试",
            source={"handling": self.handling, "reference_text": "测试规则"},
            data_origin="synthetic_test_fixture",
            missing_information=self.missing,
        )


def quality(**kwargs):
    return tool_call(
        "query_quality_support", {"category": "headphones", "issue": "左耳没有声音", **kwargs}
    )


def build_quality(tmp_path, responses, *, rules=None, gateway=None):
    settings = Settings(after_sales_db_path=tmp_path / "quality.sqlite3")
    graph = create_order_agent(
        settings,
        model=ScriptedChatModel(responses=responses),
        support_repository=rules if rules is not None else Rules(),
        handoff_gateway=gateway,
    )
    return graph, AfterSalesRepository(settings.after_sales_db_path)


def consult(graph, text, message_id=None):
    message = HumanMessage(content=text, id=message_id)
    return graph.invoke(
        {"messages": [message]},
        config={"configurable": {"thread_id": "session-policy"}},
        context=ToolContext(user_id="DEMO-U-004"),
    )


def test_quality_four_five_suggest_six_handoffs_and_no_order_registration(tmp_path):
    gateway = RecordingGateway()
    rules = Rules()
    graph, store = build_quality(
        tmp_path, [quality(order_id="DEMO-HP-O-0004")], rules=rules, gateway=gateway
    )
    for count in range(1, 7):
        state = consult(
            graph, "订单 DEMO-HP-O-0004 左耳无声" if count == 1 else "仍未解决，请继续处理"
        )
        assert state["support_question_count"] == count
        if count in (4, 5):
            assert "建议转人工" in state["messages"][-1].text
        if count < 6:
            assert not gateway.requests
        else:
            assert state["support_phase"] == "transferred"
    assert len(rules.queries) == 5
    assert len(gateway.requests) == 1
    assert store.get_status("DEMO-U-004", "DEMO-HP-O-0004") == "未售后"
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM after_sales_events").fetchone()[0] == 0


def test_quality_missing_conditions_clarifies_and_supplement_does_not_count(tmp_path):
    rules = Rules("clarify", ["连接方式和设备系统"])
    graph, _ = build_quality(
        tmp_path, [quality(), quality(environment="Windows USB", is_question=False)], rules=rules
    )
    first = consult(graph, "左耳无声")
    assert first["support_phase"] == "awaiting_information"
    assert first["support_question_count"] == 1
    assert "连接方式" in first["messages"][-1].text
    rules.handling = "basic_troubleshooting"
    second = consult(graph, "Windows电脑，USB连接")
    assert second["support_question_count"] == 1
    assert second["support_phase"] == "advised"


def test_new_quality_problem_after_handoff_does_not_reuse_old_handoff(tmp_path):
    gateway = RecordingGateway()
    graph, _ = build_quality(
        tmp_path, [quality(), quality(issue="右耳无声", new_issue=True)], gateway=gateway
    )
    consult(graph, "左耳无声")
    consult(graph, "转人工")
    state = consult(graph, "另一个新问题：右耳也不响了")
    assert state["support_question_count"] == 1
    assert state["support_phase"] == "advised"
    assert not state["support_handoff"]
    assert len(gateway.requests) == 1


@pytest.mark.parametrize("ending", ["问题解决了", "结束咨询", "不用了"])
def test_explicit_end_resets_both_counters_but_keeps_persistent_return(
    graph_builder, tmp_path, ending
):
    graph, _ = graph_builder([call()])
    ask(graph, "我要退货")
    ask(graph, f"确认退货 {ORDER}")
    store = AfterSalesRepository(tmp_path / "state.sqlite3")
    original = store.get_return_operation(USER.user_id, ORDER)
    ended = ask(graph, ending)
    assert ended["return_request_count"] == ended["support_question_count"] == 0
    assert store.get_return_operation(USER.user_id, ORDER) == original
    replay = ask(graph, f"确认退货 {ORDER}")
    # 原入口可复用，但不会再创建；最终操作ID不变。
    assert store.get_return_operation(USER.user_id, ORDER) == original
    assert replay["return_request_count"] == 0


def test_end_tool_for_complex_resolved_message_clears_pending_confirmation(graph_builder):
    graph, _ = graph_builder([call(), tool_call("end_support_session", {}), call()])
    ask(graph, "我要退货")
    ended = ask(graph, "后来我检查后解决了，这次咨询就到这里")
    assert ended["return_stage"] is None
    assert ended["return_request_count"] == 0
    replay = ask(graph, f"确认退货 {ORDER}")
    assert replay["return_stage"] == "awaiting_confirmation"
    assert "url" not in replay["return_result"]


@pytest.mark.parametrize(
    "text", ["不要结束咨询", "我在引用‘问题解决了’，其实没有", "如果解决了就结束咨询"]
)
def test_quoted_or_negated_end_is_not_a_shortcut_reset(tmp_path, text):
    graph, _ = build_quality(tmp_path, [quality(), AIMessage(content="请继续说明。")])
    consult(graph, "左耳无声")
    state = consult(graph, text)
    assert state["support_question_count"] == 1


def test_quality_and_return_counts_do_not_share_or_block_each_other(graph_builder):
    graph, _ = graph_builder([quality(), call(), quality()])
    ask(graph, "耳机左耳无声，请排查")
    returned = ask(graph, "另外，我明确要求退货")
    assert returned["support_question_count"] == returned["return_request_count"] == 1
    continued = ask(graph, "先继续排查之前的耳机问题")
    assert continued["support_question_count"] == 2
    assert continued["return_request_count"] == 1
    assert not continued["support_handoff"]


def test_shared_counter_deduplicates_across_tools_and_new_problem_retries():
    state = {"messages": [HumanMessage(content="故障仍未解决", id="one-message")]}
    first = advance_question(state, "quality", "headphones:__unbound__")
    same = advance_question({**state, **first}, "return", "__unbound__")
    assert first["support_question_count"] == 1
    assert same["return_request_count"] == 0
    retry = advance_question(
        {**state, **first}, "quality", "headphones:__unbound__", new_issue=True
    )
    assert retry["support_question_count"] == 1
    assert retry["support_problem_key"] == first["support_problem_key"]


def test_anonymous_quality_binding_preserves_count_and_attempt_history(tmp_path):
    graph, _ = build_quality(
        tmp_path, [quality(), quality(order_id="DEMO-HP-O-0004", is_question=False)]
    )
    first = consult(graph, "左耳无声")
    second = consult(graph, "补充订单 DEMO-HP-O-0004")
    assert second["support_question_count"] == 1
    assert first["support_problem_key"] == second["support_problem_key"]
    assert "不重复推荐" in second["messages"][-1].text


def test_trusted_completion_is_the_only_after_sales_flag(tmp_path):
    settings = Settings(after_sales_db_path=tmp_path / "flags.sqlite3")
    store = AfterSalesRepository(settings.after_sales_db_path)
    orders = JsonOrderRepository(settings.orders_path, store)
    for progress in ("问题已反馈", "退货申请待确认", "退货引导已取消", "退货入口生成失败"):
        store.record("DEMO-U-004", "DEMO-HP-O-0004", progress, progress)
        assert orders.find_owned("DEMO-HP-O-0004", "DEMO-U-004").after_sales_status == "未售后"
    store.record("DEMO-U-004", "DEMO-HP-O-0004", "refunded", "trusted-completion")
    order = orders.find_owned("DEMO-HP-O-0004", "DEMO-U-004")
    assert order.after_sales_status == "已售后"
    assert order.after_sales_progress == "refunded"
    reset_consultation()
    assert orders.find_owned("DEMO-HP-O-0004", "DEMO-U-004").after_sales_status == "已售后"


def test_return_rule_missing_information_does_not_create_or_handoff(graph_builder):
    graph, _ = graph_builder([call()], knowledge=Rules("clarify", ["具体使用环境"]))
    state = ask(graph, "我要退货")
    assert state["return_stage"] == "collecting"
    assert "具体使用环境" in state["messages"][-1].text
    assert not state["support_handoff"]


def test_handoff_tool_cannot_skip_quality_rules_under_six(tmp_path):
    gateway = RecordingGateway()
    graph, _ = build_quality(
        tmp_path,
        [
            quality(),
            tool_call("handoff_to_human", {"topic": "quality", "reason": "之前给过一次方案"}),
            quality(),
        ],
        gateway=gateway,
    )
    consult(graph, "左耳无声")
    state = consult(graph, "仍未解决")
    assert state["support_question_count"] == 2
    assert not gateway.requests
    errors = [m for m in state["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
    assert errors[0].artifact["error"]["code"] == "HANDOFF_NOT_READY"


def test_end_then_new_consultation_uses_new_handoff_request_id(tmp_path):
    gateway = RecordingGateway()
    graph, _ = build_quality(tmp_path, [quality()], gateway=gateway)
    consult(graph, "左耳没有声音")
    consult(graph, "转人工")
    consult(graph, "结束咨询")
    consult(graph, "左耳没有声音，需要重新咨询")
    consult(graph, "转人工")
    assert len(gateway.requests) == 2
    assert gateway.requests[0].request_id != gateway.requests[1].request_id


def test_explicit_feedback_counts_even_when_model_only_asks_for_information(tmp_path):
    gateway = RecordingGateway()
    graph, _ = build_quality(
        tmp_path, [quality(), AIMessage(content="请补充型号。")], gateway=gateway
    )
    consult(graph, "左耳没有声音")
    for count in range(2, 7):
        state = consult(graph, "没有解决")
        assert state["support_question_count"] == count
        if count in (4, 5):
            assert "建议转人工" in state["messages"][-1].text
        if count < 6:
            assert not gateway.requests
    assert len(gateway.requests) == 1


def test_new_unrelated_message_does_not_inherit_quality_suggestion(tmp_path):
    graph, _ = build_quality(
        tmp_path,
        [quality(), quality(), quality(), quality(), AIMessage(content="商品名称待确认。")],
    )
    consult(graph, "左耳没有声音")
    for _ in range(3):
        consult(graph, "没有解决")
    state = consult(graph, "另外问一下，你有哪些商品？")
    assert "建议转人工" not in state["messages"][-1].text


def test_explicit_human_only_rule_is_not_downgraded_to_clarification(tmp_path):
    query = SupportQuery(category="headphones", issue="左耳没有声音")
    knowledge = (
        Rules("human")
        .find(query)
        .model_copy(
            update={
                "required_context": [["Windows"]],
                "applicability": "Windows only",
            }
        )
    )
    path = tmp_path / "knowledge.json"
    path.write_text(json.dumps([knowledge.model_dump()], ensure_ascii=False), encoding="utf-8")
    result = JsonSupportRepository([path]).find(query)
    assert result is None or result.source["handling"] == "human"
