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


def test_multiline_information_does_not_count_and_preserves_core_reason(graph_builder):
    fact = {
        "field": "receipt_date",
        "value": "2026-10-22",
        "status": "explicit",
        "quote": "2026-10-22签收",
    }
    graph, _ = graph_builder(
        [
            call(reason="质量问题"),
            call(reason="质量问题以及日期", turn_kind="provide_information", reported_facts=[fact]),
        ]
    )
    ask(graph, "质量问题我要退货")
    result = ask(graph, "2026-10-22签收")
    assert result["return_request_count"] == 1
    assert result["return_reason"] == "质量问题"
    assert result["return_reported_facts"]["receipt_date"]["verified"] is False
    assert result["return_stage"] == "awaiting_confirmation"


@pytest.mark.parametrize("status", ["ambiguous", "conflict"])
def test_uncertain_user_fact_clarifies_not_handoff(graph_builder, status):
    graph, _ = graph_builder(
        [
            call(
                reported_facts=[
                    {
                        "field": "receipt_date",
                        "value": "前几天",
                        "status": status,
                        "quote": "前几天收到",
                    }
                ]
            )
        ]
    )
    result = ask(graph, "前几天收到，想退货")
    assert result["return_stage"] == "collecting"
    assert not result.get("support_handoff")


def test_explicit_fact_cannot_invent_receipt_from_purchase(graph_builder):
    graph, _ = graph_builder(
        [
            call(
                reported_facts=[
                    {
                        "field": "receipt_date",
                        "value": "2026-10-22",
                        "status": "explicit",
                        "quote": "购买不到七天",
                    }
                ]
            )
        ]
    )
    result = ask(graph, "购买不到七天，我想退货")
    assert result["return_result"]["code"] == "UNSUPPORTED_USER_FACT"
    assert result["return_stage"] == "collecting"


def test_multi_product_selection_binds_confirmation(graph_builder, order_rows):
    order_rows[0]["items"].append(
        {**order_rows[0]["items"][0], "product_id": "HP-P-0002", "name": "另一款耳机"}
    )
    graph, _ = graph_builder(
        [
            call(),
            call(target_product_id="HP-P-0001", turn_kind="provide_information"),
            call(target_product_id="HP-P-0002", turn_kind="confirm"),
        ]
    )
    first = ask(graph, "我要退货")
    assert first["return_result"]["missing_fields"] == ["目标商品"]
    ask(graph, "退第一款HP-P-0001")
    switched = ask(graph, f"确认退货 {ORDER}，但目标改为HP-P-0002")
    assert switched["return_stage"] == "awaiting_confirmation"
    assert switched["return_target_product_id"] == "HP-P-0002"


def test_policy_consultation_does_not_create_or_count_request(graph_builder):
    graph, _ = graph_builder(
        [
            call(reason="退货期限如何计算", turn_kind="policy_question"),
            call(reason="退货期限如何计算", turn_kind="confirm", is_return_request=False),
        ]
    )
    result = ask(graph, "退货期限怎么计算")
    assert result["return_request_count"] == 0
    assert result["return_has_request"] is False
    assert "未申请退货" in result["messages"][-1].text
    result = ask(graph, f"确认退货 {ORDER}")
    assert "尚未收到明确退货申请" in result["messages"][-1].text


@pytest.mark.parametrize("category", ["headphones", "mice", "keyboards"])
def test_policy_category_reaches_tool_and_finishes_without_application(
    graph_builder, category, tmp_path
):
    rules = TestReturnRules()
    original = rules.find
    queries = []
    rules.find = lambda query: (queries.append(query), original(query))[1]
    gateway = MockReturnGateway()
    gateway.create_link = lambda request, now: pytest.fail("规则咨询不应创建入口")
    graph, model = graph_builder(
        [
            call(
                order_id="",
                reason="退货需要什么条件",
                category=category,
                turn_kind="policy_question",
                is_return_request=False,
            ),
        ],
        knowledge=rules,
        gateway=gateway,
    )
    result = ask(graph, "退货需要什么条件")
    assert len(model._requests) == 1
    assert len(queries) == 1 and queries[0].category == category
    assert result["return_request_count"] == 0
    assert result["return_has_request"] is False
    assert result["return_stage"] == "guided"
    assert not result["return_result"].get("url")
    assert "未申请退货" in result["messages"][-1].text
    store = AfterSalesRepository(tmp_path / "state.sqlite3")
    assert store.get_return_operation(USER.user_id, ORDER) is None
    assert store.get_status(USER.user_id, ORDER) == "未售后"


def test_policy_missing_category_clarifies_without_query_or_count(graph_builder):
    rules = TestReturnRules()
    rules.find = lambda query: pytest.fail("缺品类不应调用知识库")
    graph, model = graph_builder(
        [
            call(order_id="", reason="退货条件", turn_kind="policy_question"),
        ],
        knowledge=rules,
    )
    result = ask(graph, "退货需要什么条件")
    assert len(model._requests) == 1
    assert "请说明" in result["messages"][-1].text
    assert result["return_request_count"] == 0 and result["return_has_request"] is False
    assert not result["return_result"].get("url")


def test_policy_owned_order_overrides_user_category(graph_builder):
    rules = TestReturnRules()
    original = rules.find
    queries = []
    rules.find = lambda query: (queries.append(query), original(query))[1]
    graph, _ = graph_builder(
        [
            call(turn_kind="policy_question", category="mice"),
        ],
        knowledge=rules,
    )
    result = ask(graph, "查询这个订单的退货规则")
    assert len(queries) == 1 and queries[0].category == "headphones"
    assert result["return_has_request"] is False
    assert result["return_request_count"] == 0


def test_policy_unowned_order_cannot_use_user_category_to_query(graph_builder):
    rules = TestReturnRules()
    rules.find = lambda query: pytest.fail("越权订单不应调用知识库")
    graph, _ = graph_builder(
        [
            call(order_id="RETURN-HP-O-0003", turn_kind="policy_question", category="mice"),
        ],
        knowledge=rules,
    )
    result = ask(graph, "查询该订单的退货规则")
    assert result["return_result"]["code"] == "ORDER_NOT_FOUND"
    assert result["return_has_request"] is False


@pytest.mark.parametrize("extra", [{"category": "phones"}, {"category": 1}, {"unexpected": True}])
def test_policy_invalid_category_or_extra_is_still_rejected(graph_builder, extra):
    with pytest.raises(ValidationError):
        ReturnQuery(turn_kind="policy_question", **extra)
    graph, _ = graph_builder(
        [
            call(order_id="", turn_kind="policy_question", **extra),
            AIMessage(content="请提供有效品类"),
        ]
    )
    result = ask(graph, "咨询退货规则")
    errors = [m.artifact for m in result["messages"] if isinstance(m, ToolMessage)]
    assert errors[0]["error"]["code"] == "INVALID_ARGUMENTS"
    assert not result.get("return_has_request")


def test_e34_policy_quality_then_return_completes_each_turn(graph_builder):
    graph, model = graph_builder(
        [
            call(
                order_id="",
                reason="耳机退货条件",
                category="headphones",
                turn_kind="policy_question",
                is_return_request=False,
            ),
            tool_call("query_quality_support", {"category": "headphones", "issue": "左耳没有声音"}),
            call(reason="左耳没有声音"),
        ]
    )
    policy = ask(graph, "耳机退货一般需要什么条件")
    assert policy["return_has_request"] is False
    assert policy["return_request_count"] == 0
    ask(graph, "耳机左耳没有声音，怎么排查")
    final = ask(graph, f"订单{ORDER}，左耳没有声音，我申请退货")
    assert len(model._requests) == 3
    assert final["return_stage"] == "awaiting_confirmation"
    assert not final["return_result"].get("url")


def test_future_iso_date_cannot_be_used_for_return(graph_builder):
    graph, _ = graph_builder(
        [
            call(
                reported_facts=[
                    {
                        "field": "receipt_date",
                        "value": "2026-10-29",
                        "status": "explicit",
                        "quote": "2026-10-29签收",
                    }
                ]
            )
        ]
    )
    result = ask(graph, "2026-10-29签收，我要退货")
    assert result["return_result"]["code"] == "DATE_CONFLICT"
    assert result["return_stage"] == "collecting"


@pytest.mark.parametrize("ending", ["", "。", ".", "！", "!", "。 "])
def test_confirmation_allows_one_terminal_mark_and_rechecks_rule(graph_builder, ending):
    rules = TestReturnRules()
    original = rules.find
    queries = []
    rules.find = lambda query: (queries.append(query), original(query))[1]
    graph, _ = graph_builder(
        [call(), call(turn_kind="confirm", is_return_request=False)], knowledge=rules
    )
    first = ask(graph, "我要退货")
    assert f"确认退货 {ORDER}。" not in first["messages"][-1].text
    result = ask(graph, f"  确认退货 {ORDER}{ending}  ")
    assert result["return_stage"] == "link_ready"
    assert result["return_request_count"] == 1
    assert len(queries) == 2


@pytest.mark.parametrize(
    "text",
    [
        f"确认退货 {ORDER}。。",
        f"确认退货 {ORDER}!!",
        f"不要确认退货 {ORDER}",
        f"‘确认退货 {ORDER}’是什么意思",
        f"确认退货 {ORDER}，修改金额",
        "好的",
        "帮我退吧",
        "确认退货 RETURN-HP-O-0002",
    ],
)
def test_invalid_confirmation_does_not_query_rule_or_increment(graph_builder, text):
    rules = TestReturnRules()
    original = rules.find
    queries = []
    rules.find = lambda query: (queries.append(query), original(query))[1]
    graph, _ = graph_builder(
        [call(), call(turn_kind="confirm", is_return_request=False)], knowledge=rules
    )
    ask(graph, "我要退货")
    result = ask(graph, text)
    assert result["return_stage"] == "awaiting_confirmation"
    assert result["return_request_count"] == 1
    assert result["return_result"]["code"] == "RETURN_CONFIRMATION_INVALID"
    assert len(queries) == 1
    assert not result.get("support_handoff")


def test_confirmation_with_changed_reason_needs_new_binding(graph_builder):
    graph, _ = graph_builder(
        [
            call(),
            call(
                reason="改为另一原因",
                turn_kind="confirm",
                is_return_request=False,
                reason_update="replace",
                reason_quote="原因改为另一原因",
            ),
        ]
    )
    ask(graph, "我要退货")
    result = ask(graph, f"确认退货 {ORDER}，原因改为另一原因")
    assert result["return_stage"] == "awaiting_confirmation"
    assert result["return_reason"] == "改为另一原因"


@pytest.mark.parametrize(
    "correction_text",
    [
        "修改这次申请：退货原因改为左耳无声。先不要执行，请重新让我确认。",
        "我刚才说得不准确，这次请按照左耳无声这个原因办理，先让我确认。",
        "把这次申请换一下，用左耳无声作为原因。",
    ],
)
def test_explicit_reason_correction_rebinds_without_conflict_or_execution(
    graph_builder, correction_text
):
    graph, _ = graph_builder(
        [
            call(
                reason="佩戴不舒服",
                reported_facts=[
                    {
                        "field": "symptom",
                        "value": "佩戴不舒服",
                        "status": "explicit",
                        "quote": "佩戴不舒服",
                    }
                ],
            ),
            call(
                reason="左耳无声",
                turn_kind="provide_information",
                is_return_request=False,
                reason_update="replace",
                reason_quote=correction_text,
                reported_facts=[
                    {
                        "field": "symptom",
                        "value": "左耳无声",
                        "status": "explicit",
                        "quote": "左耳无声",
                        "relation": "correction",
                    }
                ],
            ),
        ]
    )
    initial = ask(graph, "佩戴不舒服，我申请退货")
    old_binding = initial["return_confirmation_payload"]
    changed = ask(graph, correction_text)
    assert changed["return_stage"] == "awaiting_confirmation"
    assert changed["return_reason"] == "左耳无声"
    assert changed["return_reported_facts"]["symptom"]["status"] == "explicit"
    assert changed["return_reported_facts"]["symptom"]["verified"] is False
    assert changed["return_confirmation_payload"] != old_binding
    assert changed["return_confirmation_payload"]["reason"] == "左耳无声"
    assert (
        changed["return_confirmation_payload"]["revision"] == changed["return_application_revision"]
    )
    assert changed["return_request_count"] == 1
    assert not changed.get("return_result", {}).get("url")
    final = ask(graph, f"确认退货 {ORDER}")
    assert final["return_stage"] == "link_ready"
    assert final["return_reason"] == "左耳无声"


@pytest.mark.parametrize(
    "text",
    [
        "不要把退货原因改为左耳无声",
        "他说“退货原因改为左耳无声”",
        "现在左耳无声",
    ],
)
def test_additional_symptom_coexists_and_keeps_application_reason(graph_builder, text):
    graph, _ = graph_builder(
        [
            call(
                reason="佩戴不舒服",
                reported_facts=[
                    {
                        "field": "symptom",
                        "value": "佩戴不舒服",
                        "status": "explicit",
                        "quote": "佩戴不舒服",
                    }
                ],
            ),
            call(
                reason="左耳无声",
                turn_kind="provide_information",
                reported_facts=[
                    {
                        "field": "symptom",
                        "value": "左耳无声",
                        "status": "explicit",
                        "quote": "左耳无声",
                    }
                ],
            ),
        ]
    )
    ask(graph, "佩戴不舒服，我申请退货")
    result = ask(graph, text)
    assert result["return_stage"] == "awaiting_confirmation"
    assert {f["value"] for f in result["return_reported_facts"].values()} == {
        "佩戴不舒服",
        "左耳无声",
    }
    assert all(f["status"] == "explicit" for f in result["return_reported_facts"].values())
    assert result["return_reason"] == "佩戴不舒服"


def test_relation_uncertain_blocks_confirmation_until_user_clarifies(graph_builder):
    graph, _ = graph_builder(
        [
            call(reason="佩戴不舒服"),
            call(
                reason="左耳无声",
                turn_kind="provide_information",
                reason_update="uncertain",
                new_issue=True,
            ),
            call(turn_kind="confirm", is_return_request=False),
            call(turn_kind="provide_information", reason_quote="只是补充，不改退货原因"),
        ]
    )
    initial = ask(graph, "佩戴不舒服，我申请退货")
    uncertain = ask(graph, "左耳还没声音")
    assert uncertain["return_stage"] == "collecting"
    assert uncertain["return_confirmation_payload"] == {}
    assert uncertain["return_problem_key"] == initial["return_problem_key"]
    blocked = ask(graph, f"确认退货 {ORDER}")
    assert blocked["return_stage"] == "collecting"
    assert not blocked["return_result"].get("url")
    clarified = ask(graph, "只是补充，不改退货原因")
    assert clarified["return_stage"] == "awaiting_confirmation"
    assert clarified["return_reason"] == "佩戴不舒服"


def test_reason_change_requires_current_quote_and_keeps_history(graph_builder):
    graph, _ = graph_builder(
        [
            call(reason="佩戴不舒服"),
            call(
                reason="左耳无声",
                turn_kind="provide_information",
                reason_update="replace",
                reason_quote="不存在的修改意图",
            ),
            call(
                reason="左耳无声",
                turn_kind="provide_information",
                reason_update="replace",
                reason_quote="这次用左耳无声办理",
                new_issue=True,
            ),
        ]
    )
    initial = ask(graph, "佩戴不舒服，我申请退货")
    invalid = ask(graph, "左耳无声")
    assert invalid["return_result"]["code"] == "UNSUPPORTED_USER_FACT"
    assert invalid["return_reason"] == "佩戴不舒服"
    revised = ask(graph, "这次用左耳无声办理")
    assert revised["return_stage"] == "awaiting_confirmation"
    assert revised["return_reason"] == "左耳无声"
    assert revised["return_application_revision"] > initial["return_application_revision"]
    assert revised["return_problem_key"] == initial["return_problem_key"]
    event = next(e for e in revised["return_information_history"] if e["kind"] == "reason_update")
    assert event["before"] == "佩戴不舒服" and event["after"] == "左耳无声"
    assert event["verified"] is False


def test_multiple_symptom_correction_requires_target_and_keeps_other_symptoms(graph_builder):
    first = {"field": "symptom", "value": "佩戴不舒服", "status": "explicit", "quote": "佩戴不舒服"}
    second = {"field": "symptom", "value": "左耳无声", "status": "explicit", "quote": "左耳无声"}
    changed = {
        "field": "symptom",
        "value": "右耳无声",
        "status": "explicit",
        "quote": "右耳无声",
        "relation": "correction",
    }
    graph, _ = graph_builder(
        [
            call(reason="佩戴不舒服", reported_facts=[first]),
            call(turn_kind="provide_information", reported_facts=[second], new_issue=True),
            call(turn_kind="provide_information", reported_facts=[changed]),
            call(
                turn_kind="provide_information",
                reason_quote="只纠正刚才的描述",
                reported_facts=[{**changed, "replaces_quote": "左耳无声"}],
            ),
        ]
    )
    initial = ask(graph, "佩戴不舒服，我申请退货")
    supplemented = ask(graph, "还有左耳无声")
    assert supplemented["return_problem_key"] == initial["return_problem_key"]
    uncertain = ask(graph, "我说错了，是右耳无声")
    assert uncertain["return_stage"] == "collecting"
    final = ask(graph, "只纠正刚才的描述，左耳无声说错了，是右耳无声")
    assert final["return_stage"] == "awaiting_confirmation"
    assert {f["value"] for f in final["return_reported_facts"].values()} == {
        "佩戴不舒服",
        "右耳无声",
    }
    assert any(
        (e.get("before") or {}).get("value") == "左耳无声"
        for e in final["return_information_history"]
        if e["kind"] == "fact"
    )


def test_explicit_critical_fact_correction_is_unverified_and_rechecked(graph_builder):
    first = {
        "field": "receipt_date",
        "value": "2026-10-21",
        "status": "explicit",
        "quote": "2026-10-21",
    }
    second = {
        "field": "receipt_date",
        "value": "2026-10-22",
        "status": "explicit",
        "quote": "2026-10-22",
        "relation": "correction",
    }
    graph, _ = graph_builder(
        [
            call(reported_facts=[first]),
            call(turn_kind="provide_information", reported_facts=[second]),
        ]
    )
    initial = ask(graph, "2026-10-21签收，我要退货")
    final = ask(graph, "之前日期说错了，2026-10-22才签收")
    assert final["return_stage"] == "awaiting_confirmation"
    assert final["return_reported_facts"]["receipt_date"]["value"] == "2026-10-22"
    assert final["return_reported_facts"]["receipt_date"]["verified"] is False
    assert final["return_confirmation_payload"] != initial["return_confirmation_payload"]


def test_nonidentical_critical_text_is_ambiguous_not_proven_conflict(graph_builder):
    graph, _ = graph_builder(
        [
            call(
                reported_facts=[
                    {
                        "field": "receipt_date",
                        "value": "今天",
                        "status": "explicit",
                        "quote": "今天",
                    }
                ]
            ),
            call(
                turn_kind="provide_information",
                reported_facts=[
                    {
                        "field": "receipt_date",
                        "value": "2026-10-23",
                        "status": "explicit",
                        "quote": "2026-10-23",
                    }
                ],
            ),
        ]
    )
    ask(graph, "今天签收，我要退货")
    final = ask(graph, "2026-10-23签收")
    assert final["return_reported_facts"]["receipt_date"]["status"] == "ambiguous"
    assert final["return_stage"] == "collecting"
    assert not final["return_result"].get("url")


def test_explicit_new_problem_starts_new_context_without_deleting_history(graph_builder):
    graph, _ = graph_builder(
        [
            call(
                reason="佩戴不舒服",
                reported_facts=[
                    {
                        "field": "symptom",
                        "value": "佩戴不舒服",
                        "status": "explicit",
                        "quote": "佩戴不舒服",
                    }
                ],
            ),
            call(
                reason="左耳无声",
                new_issue=True,
                reported_facts=[
                    {
                        "field": "symptom",
                        "value": "左耳无声",
                        "status": "explicit",
                        "quote": "左耳无声",
                        "relation": "new_problem",
                    }
                ],
            ),
        ]
    )
    initial = ask(graph, "佩戴不舒服，我要退货")
    final = ask(graph, "这是独立的新问题：左耳无声，我申请退货")
    assert final["return_problem_key"] != initial["return_problem_key"]
    assert final["return_request_count"] == 1
    assert final["return_reason"] == "左耳无声"
    assert len(final["return_information_history"]) >= 2
    assert final["return_stage"] == "awaiting_confirmation"


def test_reset_clears_statement_context_and_application_revision_only():
    from orderagent.tools.session_policy import reset_consultation

    reset = reset_consultation()
    assert reset["return_information_history"] == []
    assert reset["return_application_revision"] == 0
    assert reset["return_information_pending"] == {}
    assert reset["return_reason_update_pending"] is False
    assert "return_operations" not in reset


def test_strict_pending_confirmation_skips_llm_but_rechecks_rule_and_limits(graph_builder):
    rules = TestReturnRules()
    original = rules.find
    queries = []
    rules.find = lambda query: (queries.append(query), original(query))[1]
    graph, model = graph_builder([call()], knowledge=rules)
    ask(graph, "我要退货")
    state = ask(graph, f"  确认退货 {ORDER}。  ")
    assert len(model._requests) == 1
    assert len(queries) == 2
    assert state["return_stage"] == "link_ready"
    routed = [
        m
        for m in state["messages"]
        if getattr(m, "name", "") == "deterministic_return_confirmation"
    ]
    assert len(routed) == 1
    assert routed[0].tool_calls[0]["args"]["turn_kind"] == "confirm"
    checkpoint = graph.get_state({"configurable": {"thread_id": "return"}}).values
    assert checkpoint["thread_tool_call_count"]["__all__"] == 2


@pytest.mark.parametrize(
    "text", ["好的", f"确认退货 {ORDER}！！", f"不要确认退货 {ORDER}", "确认退货 RETURN-HP-O-0002"]
)
def test_ambiguous_or_invalid_confirmation_still_uses_main_model(graph_builder, text):
    graph, model = graph_builder([call(), call(turn_kind="confirm", is_return_request=False)])
    ask(graph, "我要退货")
    state = ask(graph, text)
    assert len(model._requests) == 2
    assert state["return_stage"] == "awaiting_confirmation"
    assert not any(
        getattr(m, "name", "") == "deterministic_return_confirmation" for m in state["messages"]
    )


def test_fast_confirmation_does_not_bypass_order_ownership(graph_builder):
    graph, model = graph_builder([call()])
    ask(graph, "我要退货")
    state = ask(graph, f"确认退货 {ORDER}", context=ToolContext(user_id="OTHER"))
    assert len(model._requests) == 1
    assert state["return_result"]["code"] == "ORDER_NOT_FOUND"
    assert state["return_stage"] != "link_ready"


def test_fast_confirmation_rule_failure_still_hands_off(graph_builder):
    rules = TestReturnRules()
    graph, model = graph_builder([call()], knowledge=rules)
    ask(graph, "我要退货")

    def fail(query):
        raise ValueError("RAG_SELECTION_ERROR")

    rules.find = fail
    state = ask(graph, f"确认退货 {ORDER}")
    assert len(model._requests) == 1
    assert state["return_stage"] == "handoff"
    assert state["return_error_code"] == "RAG_SELECTION_ERROR"


def test_changed_pending_binding_does_not_take_shortcut(graph_builder):
    graph, model = graph_builder([call(), AIMessage(content="申请条件已变化，请重新核对。")])
    ask(graph, "我要退货")
    graph.update_state(
        {"configurable": {"thread_id": "return"}}, {"return_confirmation_payload": {}}
    )
    state = ask(graph, f"确认退货 {ORDER}")
    assert len(model._requests) == 2
    assert not any(
        getattr(m, "name", "") == "deterministic_return_confirmation" for m in state["messages"]
    )


def test_confirmation_without_pending_application_cannot_create(graph_builder):
    graph, _ = graph_builder([call(turn_kind="confirm", is_return_request=False)])
    result = ask(graph, f"确认退货 {ORDER}。")
    assert result["return_has_request"] is False
    assert result["return_stage"] == "guided"


def test_conflicting_fact_values_require_confirmation(graph_builder):
    facts = [
        {
            "field": "receipt_date",
            "value": "2026-10-21",
            "status": "explicit",
            "quote": "2026-10-21签收",
        },
        {
            "field": "receipt_date",
            "value": "2026-10-22",
            "status": "explicit",
            "quote": "2026-10-22签收",
        },
    ]
    graph, _ = graph_builder(
        [
            call(reported_facts=[facts[0]]),
            call(turn_kind="provide_information", reported_facts=[facts[1]]),
        ]
    )
    ask(graph, "2026-10-21签收，要退货")
    result = ask(graph, "2026-10-22签收")
    assert result["return_reported_facts"]["receipt_date"]["status"] == "conflict"
    assert result["return_stage"] == "collecting"
    assert result["return_request_count"] == 1
    assert "签收时间的两次描述不同" in result["messages"][-1].text
    assert "2026-10-21" in result["messages"][-1].text
    assert "2026-10-22" in result["messages"][-1].text
    assert "receipt_date" not in result["messages"][-1].text


def test_distinct_other_facts_are_not_a_false_conflict(graph_builder):
    facts = [
        {"field": "other", "value": "今天刚收到", "quote": "今天刚收到", "status": "explicit"},
        {"field": "other", "value": "TEST鼠标", "quote": "TEST鼠标", "status": "explicit"},
    ]
    graph, _ = graph_builder(
        [
            call(reported_facts=[facts[0]]),
            call(turn_kind="provide_information", reported_facts=[facts[1]]),
            call(turn_kind="provide_information", reported_facts=[facts[1]]),
        ]
    )
    ask(graph, "今天刚收到，我要退货")
    state = ask(graph, "TEST鼠标")
    assert state["return_stage"] == "awaiting_confirmation"
    assert len(state["return_reported_facts"]) == 2
    assert all(f["status"] == "explicit" for f in state["return_reported_facts"].values())
    old_binding = state["return_confirmation_payload"]
    repeated = ask(graph, "型号还是TEST鼠标")
    assert len(repeated["return_reported_facts"]) == 2
    assert repeated["return_confirmation_payload"] == old_binding
    assert repeated["return_request_count"] == 1


def test_model_and_receipt_information_are_independent_fields(graph_builder):
    graph, _ = graph_builder(
        [
            call(
                reported_facts=[
                    {
                        "field": "receipt_date",
                        "value": "今天",
                        "status": "explicit",
                        "quote": "今天收到",
                    }
                ]
            ),
            call(
                turn_kind="provide_information",
                reported_facts=[
                    {
                        "field": "product_model",
                        "value": "TEST",
                        "status": "explicit",
                        "quote": "型号TEST",
                    }
                ],
            ),
        ]
    )
    ask(graph, "今天收到，我要退货")
    state = ask(graph, "型号TEST")
    assert state["return_stage"] == "awaiting_confirmation"
    assert set(state["return_reported_facts"]) == {"receipt_date", "product_model"}


def test_historical_quote_cannot_silently_clear_conflict(graph_builder):
    first = {
        "field": "receipt_date",
        "value": "2026-10-21",
        "status": "explicit",
        "quote": "2026-10-21签收",
    }
    second = {
        "field": "receipt_date",
        "value": "2026-10-22",
        "status": "explicit",
        "quote": "2026-10-22签收",
    }
    graph, _ = graph_builder(
        [
            call(reported_facts=[first]),
            call(turn_kind="provide_information", reported_facts=[second]),
            call(turn_kind="provide_information", reported_facts=[second]),
            call(turn_kind="provide_information", reported_facts=[second]),
        ]
    )
    ask(graph, "2026-10-21签收，我要退货")
    ask(graph, "2026-10-22签收")
    unresolved = ask(graph, "商品HP-P-0001")
    assert unresolved["return_stage"] == "collecting"
    assert unresolved["return_reported_facts"]["receipt_date"]["status"] == "conflict"
    resolved = ask(graph, "请以2026-10-22签收为准")
    assert resolved["return_reported_facts"]["receipt_date"]["status"] == "explicit"
    assert resolved["return_stage"] == "awaiting_confirmation"


def test_wrong_target_id_returns_owned_candidates(graph_builder):
    graph, _ = graph_builder([call(target_product_id="MS-P-NOT-OWNED")])
    state = ask(graph, "我要退货，商品MS-P-NOT-OWNED")
    assert state["return_stage"] == "collecting"
    assert "HP-P-0001" in state["messages"][-1].text
    assert any(d["change"] == "not_matched" for d in state["return_fact_diagnostics"])


def test_repeated_return_intent_does_not_replace_existing_symptom(graph_builder):
    questions = [
        "左耳无声，我申请退货",
        "同一个订单左耳无声，我仍然要求退货",
        "还是这个订单，我坚持退货",
        "同一问题，我再次要求退货",
        "同一问题，仍然没有解决，我要求退货",
        "同一问题，继续退货",
    ]
    graph, _ = graph_builder(
        [
            call(
                reason="左耳无声",
                turn_kind="request",
                is_return_request=True,
                reported_facts=[
                    {
                        "field": "symptom",
                        "value": "左耳无声",
                        "status": "explicit",
                        "quote": text,
                        "normalized_description": "耳机左耳无声音输出",
                    }
                ],
            )
            for text in questions
        ]
    )
    original_quote = None
    for number, text in enumerate(questions, 1):
        state = ask(graph, text)
        assert state["return_request_count"] == number
        fact = state["return_reported_facts"]["symptom"]
        assert fact["value"] == "左耳无声"
        assert fact["status"] == "explicit"
        if number == 2:
            original_quote = fact["quote"]
        if number >= 3:
            assert fact["quote"] == original_quote
        if number < 6:
            assert state["return_stage"] == "awaiting_confirmation"
        if number in (4, 5):
            assert "建议转人工" in state["messages"][-1].text
        if number == 6:
            assert state["return_stage"] == "handoff"


def test_symptom_summary_is_search_hint_not_new_user_fact(graph_builder):
    graph, _ = graph_builder(
        [
            call(
                reported_facts=[
                    {
                        "field": "symptom",
                        "value": "耳机左耳无声",
                        "quote": "左耳无声",
                        "status": "explicit",
                    }
                ]
            )
        ]
    )
    result = ask(graph, "左耳无声，我申请退货")
    fact = result["return_reported_facts"]["symptom"]
    assert fact["value"] == "左耳无声"
    assert fact["normalized_description"] == "耳机左耳无声"
    assert fact["verified"] is False
    assert result["return_confirmation_payload"]["facts"]["symptom"] == {
        "value": "左耳无声",
        "status": "explicit",
    }


def test_repeating_same_fact_with_new_source_preserves_business_binding(graph_builder):
    first = {
        "field": "receipt_date",
        "value": "2026-10-22",
        "quote": "2026-10-22签收",
        "status": "explicit",
    }
    second = {**first, "quote": "签收日期2026-10-22"}
    graph, _ = graph_builder(
        [
            call(reported_facts=[first]),
            call(turn_kind="provide_information", reported_facts=[second]),
            call(turn_kind="confirm", is_return_request=False),
        ]
    )
    a = ask(graph, "2026-10-22签收，我要退货")
    b = ask(graph, "再补充，签收日期2026-10-22")
    assert (
        a["return_reported_facts"]["receipt_date"]["message_id"]
        != b["return_reported_facts"]["receipt_date"]["message_id"]
    )
    assert a["return_confirmation_payload"] == b["return_confirmation_payload"]
    c = ask(graph, f"确认退货 {ORDER}。")
    assert c["return_stage"] == "link_ready"
    assert c["return_request_count"] == 1


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
            tool_call("handle_return", {"order_id": ORDER, "is_return_request": False}),
            tool_call("handle_return", {"reason": "不喜欢", "is_return_request": False}),
        ]
    )
    ask(graph, "我要退货")
    ask(graph, f"订单号是{ORDER}")
    state = ask(graph, "原因是不喜欢")
    assert state["return_stage"] == "awaiting_confirmation"
    assert state["return_order_id"] == ORDER and state["return_reason"] == "不喜欢"
    assert not state.get("support_handoff")
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
    assert len(model._requests) == 1


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
        for count in (2, 3, 4, 5):
            state = ask(graph, text)
            assert state["return_request_count"] == count
            assert state["return_stage"] == "awaiting_confirmation"
            assert not handoff.requests
        state = ask(graph, text)
        assert state["return_request_count"] == 6
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
    for count in range(2, 7):
        state = ask(graph, "还是这个退货问题，继续帮我处理")
        assert state["return_request_count"] == count
        assert state["return_stage"] == ("handoff" if count == 6 else "awaiting_confirmation")
        if count in (4, 5):
            assert "建议转人工" in state["messages"][-1].text


def test_unusual_insistence_is_routed_by_model_flag(graph_builder):
    graph, _ = graph_builder([call(), call(request_human=True)])
    ask(graph, "我要退货")
    state = ask(graph, "这件事请让人工客服处理")
    assert state["return_stage"] == "handoff"


def test_missing_information_repeated_request_hands_off_at_six(graph_builder):
    graph, _ = graph_builder([call(order_id="", reason="")])
    for count in range(1, 7):
        state = ask(graph, "我仍然要求退货，请继续帮我处理")
        assert state["return_request_count"] == count
        assert state["return_stage"] == ("handoff" if count == 6 else "collecting")


def test_cancelled_request_does_not_trigger_handoff(graph_builder):
    graph, model = graph_builder([call(), AIMessage(content="好的。")])
    ask(graph, "我要退货")
    state = ask(graph, "不用退货了")
    assert state["return_stage"] is None
    assert not state.get("support_handoff")
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


def test_link_is_not_created_twice_after_repeated_questions(graph_builder):
    class RecordingReturns(MockReturnGateway):
        def __init__(self):
            super().__init__("eligible")
            self.links = []

        def create_link(self, request, now):
            self.links.append(request)
            return super().create_link(request, now)

    gateway = RecordingReturns()
    graph, model = graph_builder([call()], gateway=gateway)
    ask(graph, "我要退货")
    ask(graph, f"确认退货 {ORDER}")
    for count in range(2, 7):
        state = ask(graph, "我仍然要求售后")
        assert state["return_request_count"] == count
        assert state["return_stage"] == ("handoff" if count == 6 else "link_ready")
    assert len(gateway.links) == 1
    assert len(model._requests) == 6


def test_return_state_isolated_and_reset_uses_official_checkpoint(graph_builder):
    graph, _ = graph_builder([call()])
    ask(graph, "我要退货", thread="first")
    other = ask(graph, "我要退货", thread="second")
    assert other["return_stage"] == "awaiting_confirmation"
    assert not other.get("support_handoff")
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
    assert set(schema["properties"]) == {
        "order_id",
        "reason",
        "request_human",
        "is_return_request",
        "new_issue",
        "target_product_id",
        "turn_kind",
        "reported_facts",
        "reason_update",
        "reason_quote",
        "category",
    }


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
    state = ask(graph, f"确认退货 {ORDER}，原因改为另一个原因")
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
        assert state["return_stage"] == "awaiting_confirmation"
    assert not gateway.requests


def test_cancelling_clears_session_question_history(graph_builder):
    graph, _ = graph_builder([call()])
    ask(graph, "我要退货")
    ended = ask(graph, "取消退货")
    assert ended["return_request_count"] == 0
    state = ask(graph, f"我又要退货，订单{ORDER}")
    assert state["return_request_count"] == 1
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
        message.artifact.get("error", {}).get("code") == "HANDOFF_NOT_READY" for message in errors
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
