from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event, Lock

import pytest

from orderagent.integrations.returns import MockReturnGateway
from orderagent.repositories.after_sales_repository import AfterSalesRepository
from orderagent.schemas.order_schema import StoredOrder, ToolContext
from orderagent.schemas.return_schema import ReturnOperation
from orderagent.tools.return_tools import check_return_operation
from tests import test_return_tools
from tests.test_return_tools import (
    NOW,
    ORDER,
    USER,
    TestReturnRules,
    ask,
    call,
)

graph_builder = test_return_tools.graph_builder
order_rows = test_return_tools.order_rows


class CountingGateway(MockReturnGateway):
    """只统计入口调用，旧模拟证据方法若被调用则让测试失败。"""

    def __init__(self, fail=False):
        super().__init__()
        self.calls = []
        self.lock = Lock()
        self.fail = fail

    def get_evidence(self, *args):
        raise AssertionError("不应读取模拟订单资格证据")

    def create_link(self, request, now):
        with self.lock:
            self.calls.append(request)
        if self.fail:
            raise RuntimeError("private-platform-key")
        return super().create_link(request, now)


@pytest.mark.parametrize("status", ["pending_payment", "paid_unshipped", "shipped", "cancelled"])
def test_mock_order_status_or_signature_date_does_not_prove_quality(
    graph_builder, order_rows, status
):
    order_rows[0].update(order_status=status, delivered_at=None)
    gateway = CountingGateway()
    graph, _ = graph_builder([call(), call()], gateway=gateway)
    assert ask(graph, "耳机左耳无声，我要退货")["return_stage"] == "awaiting_confirmation"
    state = ask(graph, f"确认退货 {ORDER}")
    assert state["return_stage"] == "link_ready"
    assert len(gateway.calls) == 1


@pytest.mark.parametrize(
    "status,stage",
    [
        ("已退货", "rejected"),
        ("已退款", "rejected"),
        ("returned", "rejected"),
        ("refunded", "rejected"),
        ("退货处理中", "guided"),
        ("退款处理中", "guided"),
    ],
)
def test_existing_terminal_or_processing_record_prevents_new_operation(
    graph_builder, tmp_path, status, stage
):
    gateway = CountingGateway()
    graph, _ = graph_builder([call()], gateway=gateway)
    store = AfterSalesRepository(tmp_path / "state.sqlite3")
    store.record(USER.user_id, ORDER, status, "trusted-fixture")
    state = ask(graph, "我要退货")
    assert state["return_stage"] == stage
    assert "url" not in state["return_result"]
    assert not gateway.calls
    # 人工或普通查询事件不能擦掉可信的终态/处理中状态。
    store.record(USER.user_id, ORDER, "已转人工（模拟）", "later-handoff")
    assert store.get_status(USER.user_id, ORDER) == status


@pytest.mark.parametrize("status", ["已转人工（模拟）", "转人工失败", "已提供一次排查"])
def test_ordinary_after_sales_record_does_not_mean_already_returned(
    graph_builder, tmp_path, status
):
    gateway = CountingGateway()
    graph, _ = graph_builder([call(), call()], gateway=gateway)
    store = AfterSalesRepository(tmp_path / "state.sqlite3")
    store.record(USER.user_id, ORDER, status, "old-consultation")
    ask(graph, "耳机左耳没声音，我要退货")
    state = ask(graph, f"确认退货 {ORDER}")
    assert state["return_stage"] == "link_ready"
    assert len(gateway.calls) == 1
    assert store.get_status(USER.user_id, ORDER) == "已提供退货测试入口"


def test_restart_and_reason_change_reuse_persisted_link(graph_builder):
    gateway = CountingGateway()
    graph, _ = graph_builder([call(), call()], gateway=gateway)
    ask(graph, "我要退货")
    first = ask(graph, f"确认退货 {ORDER}")
    fresh, _ = graph_builder([call(reason="另一个退货原因")], gateway=gateway)
    second = ask(fresh, "现在我又要退这笔订单", thread="fresh-thread")
    assert second["return_stage"] == "link_ready"
    assert first["return_result"]["request_id"] == second["return_result"]["request_id"]
    assert first["return_result"]["url"] == second["return_result"]["url"]
    assert len(gateway.calls) == 1


@pytest.mark.parametrize("status", ["returned", "refunded"])
def test_completed_operation_is_checked_before_reusing_session_link(
    graph_builder, tmp_path, status
):
    gateway = CountingGateway()
    graph, _ = graph_builder([call(), call(), call()], gateway=gateway)
    ask(graph, "我要退货")
    ask(graph, f"确认退货 {ORDER}")
    store = AfterSalesRepository(tmp_path / "state.sqlite3")
    operation = store.get_return_operation(USER.user_id, ORDER)
    store.update_return_operation(operation.model_copy(update={"status": status}))
    state = ask(graph, "我要再退一次")
    assert state["return_stage"] == "rejected"
    assert state["return_result"]["code"] == "RETURN_ALREADY_COMPLETED"
    assert "url" not in state["return_result"]
    assert len(gateway.calls) == 1
    with pytest.raises(ValueError):
        store.update_return_operation(operation)


def test_two_graphs_cannot_create_two_links_concurrently(graph_builder):
    started, release = Event(), Event()

    class BlockingGateway(CountingGateway):
        def create_link(self, request, now):
            started.set()
            assert release.wait(10), "test release timed out"
            return super().create_link(request, now)

    gateway = BlockingGateway()
    first, _ = graph_builder([call(), call()], gateway=gateway)
    second, _ = graph_builder([call(), call()], gateway=gateway)
    ask(first, "我要退货", thread="first")
    ask(second, "我要退货", thread="second")
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(ask, first, f"确认退货 {ORDER}", "first")
        try:
            assert started.wait(10)
            other = executor.submit(ask, second, f"确认退货 {ORDER}", "second").result(10)
            assert other["return_result"]["code"] == "RETURN_IN_PROGRESS"
            assert "url" not in other["return_result"]
        finally:
            release.set()
        assert pending.result(10)["return_stage"] == "link_ready"
    assert len(gateway.calls) == 1


def test_ambiguous_failure_remains_pending_across_restart(graph_builder, tmp_path):
    gateway = CountingGateway(fail=True)
    graph, _ = graph_builder([call(), call()], gateway=gateway)
    ask(graph, "我要退货")
    failed = ask(graph, f"确认退货 {ORDER}")
    assert failed["return_result"]["code"] == "RETURN_RESULT_UNCERTAIN"
    assert "private-platform-key" not in str(failed)
    store = AfterSalesRepository(tmp_path / "state.sqlite3")
    assert store.get_return_operation(USER.user_id, ORDER).status == "uncertain"
    fresh, _ = graph_builder([call()], gateway=gateway)
    state = ask(fresh, "我重试退货", thread="retry")
    assert state["return_result"]["code"] == "RETURN_IN_PROGRESS"
    assert len(gateway.calls) == 1


def test_expired_link_is_not_recreated(graph_builder):
    clock = [NOW]
    gateway = CountingGateway()
    graph, _ = graph_builder([call(), call(), call()], gateway=gateway, clock=lambda: clock[0])
    ask(graph, "我要退货")
    ask(graph, f"确认退货 {ORDER}")
    clock[0] += timedelta(minutes=31)
    state = ask(graph, "我还要退货入口")
    assert state["return_result"]["code"] == "RETURN_LINK_EXPIRED"
    assert "url" not in state["return_result"]
    assert len(gateway.calls) == 1


def test_claim_rechecks_terminal_state_changed_during_rule_lookup(graph_builder, tmp_path):
    class ChangingRule(TestReturnRules):
        def __init__(self):
            self.lookups = 0

        def find(self, query):
            self.lookups += 1
            if self.lookups == 2:
                AfterSalesRepository(tmp_path / "state.sqlite3").record(
                    USER.user_id, ORDER, "已退款", "platform-update-during-confirmation"
                )
            return super().find(query)

    gateway = CountingGateway()
    graph, _ = graph_builder([call(), call()], gateway=gateway, knowledge=ChangingRule())
    ask(graph, "我要退货")
    state = ask(graph, f"确认退货 {ORDER}")
    assert state["return_stage"] == "rejected"
    assert not gateway.calls


def test_atomic_claim_is_unique_and_isolated_by_user(tmp_path):
    store = AfterSalesRepository(tmp_path / "state.sqlite3")
    operation = ReturnOperation(
        user_id="A", order_id="O", request_id="stable-A-O", status="creating"
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: store.claim_return_operation(operation), range(8)))
    assert results.count(True) == 1
    assert store.get_return_operation("B", "O") is None
    assert store.claim_return_operation(operation.model_copy(update={"user_id": "B"}))
    with pytest.raises(ValueError):
        store.update_return_operation(
            operation.model_copy(
                update={
                    "status": "uncertain",
                    "request_id": "wrong-request",
                }
            )
        )


def test_preflight_rejects_foreign_order_before_accessing_records(order_rows):
    class ForbiddenStore:
        def get_return_operation(self, *args):
            raise AssertionError("不得读取他人退货记录")

    result = check_return_operation(
        StoredOrder.model_validate(order_rows[0]),
        ToolContext(user_id="OTHER"),
        ForbiddenStore(),
        NOW,
    )
    assert result[0] == "blocked"


def test_rule_query_uses_reported_symptom_without_simulated_evidence(graph_builder):
    class RecordingRules(TestReturnRules):
        def __init__(self):
            self.queries = []

        def find(self, query):
            self.queries.append(query)
            return super().find(query)

    rules = RecordingRules()
    graph, _ = graph_builder([call(reason="耳机左耳没有声音")], knowledge=rules)
    state = ask(graph, "我的耳机左耳没有声音，我要退货")
    assert state["return_stage"] == "awaiting_confirmation"
    assert len(rules.queries) == 1
    assert "用户报告：耳机左耳没有声音" in rules.queries[0].issue
    assert "已核验" not in rules.queries[0].issue
    assert "程序资格校验结果" not in rules.queries[0].issue
