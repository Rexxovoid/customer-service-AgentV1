from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from orderagent.agents.agent_factory import create_order_agent
from orderagent.integrations.ragflow import RagflowKnowledge, RuleSelection, TimeoutRAGFlow
from orderagent.repositories.after_sales_repository import AfterSalesRepository
from orderagent.repositories.order_repository import JsonOrderRepository
from orderagent.schemas.order_schema import OrderQuery, ToolContext
from orderagent.schemas.support_schema import SupportQuery
from orderagent.settings import Settings
from orderagent.tools.order_query_tool import query_order
from tests.fake_chat_model import ScriptedChatModel, tool_call
from tests.test_support_tools import RecordingGateway

ORDER = "DEMO-HP-O-0004"
CONTEXT = ToolContext(user_id="DEMO-U-004")


def invoke(graph, text, thread="persist"):
    return graph.invoke(
        {"messages": [("user", text)]},
        config={"configurable": {"thread_id": thread}},
        context=CONTEXT,
    )


def test_after_sales_starts_empty_and_follows_actual_processing(tmp_path):
    settings = Settings(after_sales_db_path=tmp_path / "state.sqlite3")
    store = AfterSalesRepository(settings.after_sales_db_path)
    orders = JsonOrderRepository(settings.orders_path, store)
    assert (
        query_order(orders, OrderQuery(order_id=ORDER), CONTEXT).data["after_sales_status"]
        == "未售后"
    )
    gateway = RecordingGateway()
    graph = create_order_agent(
        settings,
        model=ScriptedChatModel(
            responses=[
                tool_call(
                    "query_quality_support",
                    {
                        "category": "headphones",
                        "issue": "左耳没有声音",
                        "order_id": ORDER,
                    },
                )
            ]
        ),
        handoff_gateway=gateway,
    )
    invoke(graph, f"订单{ORDER}，我的耳机左耳没有声音")
    assert store.get_status(CONTEXT.user_id, ORDER) == "已提供一次排查"
    state = invoke(graph, "没有解决")
    assert state["support_phase"] == "transferred"
    assert store.get_status(CONTEXT.user_id, ORDER) == "已转人工（模拟）"
    assert (
        AfterSalesRepository(settings.after_sales_db_path).get_status(CONTEXT.user_id, ORDER)
        == "已转人工（模拟）"
    )
    # 删除官方会话检查点不会删除订单的实际处理记录。
    graph.checkpointer.delete_thread("persist")
    assert store.get_status(CONTEXT.user_id, ORDER) == "已转人工（模拟）"


@pytest.mark.parametrize("order_id", ["DEMO-HP-O-0002", "DEMO-HP-O-NOT-FOUND"])
def test_foreign_or_missing_order_cannot_create_after_sales_record(tmp_path, order_id):
    settings = Settings(after_sales_db_path=tmp_path / "state.sqlite3")
    graph = create_order_agent(
        settings,
        model=ScriptedChatModel(
            responses=[
                tool_call(
                    "query_quality_support",
                    {
                        "category": "headphones",
                        "issue": "左耳没有声音",
                        "order_id": order_id,
                    },
                ),
                AIMessage(content="请核对订单号。"),
            ]
        ),
    )
    invoke(graph, f"订单{order_id}，耳机左耳没有声音")
    assert (
        AfterSalesRepository(settings.after_sales_db_path).get_status(CONTEXT.user_id, order_id)
        == "未售后"
    )


def test_store_deduplicates_events_and_isolates_users(tmp_path):
    store = AfterSalesRepository(tmp_path / "state.sqlite3")
    store.record("A", "O", "问题已反馈", "event-1")
    store.record("A", "O", "已提供一次排查", "event-2")
    store.record("A", "O", "问题已反馈", "event-1")
    assert store.get_status("A", "O") == "已提供一次排查"
    assert store.get_status("B", "O") == "未售后"
    store.record("A", "O", "已转人工（模拟）", "event-3")
    store.record("A", "O", "已提供一次排查", "late-event")
    assert store.get_status("A", "O") == "已转人工（模拟）"


def test_concurrent_issue_claim_only_allows_one_first_attempt(tmp_path):
    store = AfterSalesRepository(tmp_path / "state.sqlite3")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda n: store.start_issue("A", "O", str(n)), [1, 2]))
    assert sorted(results) == [False, True]


@pytest.mark.parametrize(
    "order_id,category",
    [
        ("DEMO-HP-O-0001", "headphones"),
        (ORDER, "mice"),
    ],
)
def test_order_status_or_category_mismatch_never_provides_auto_advice(tmp_path, order_id, category):
    settings = Settings(after_sales_db_path=tmp_path / "state.sqlite3")
    context = ToolContext(user_id="DEMO-U-001" if order_id.endswith("0001") else CONTEXT.user_id)
    graph = create_order_agent(
        settings,
        model=ScriptedChatModel(
            responses=[
                tool_call(
                    "query_quality_support",
                    {
                        "category": category,
                        "issue": "左耳没有声音",
                        "order_id": order_id,
                    },
                ),
                AIMessage(content="请核对信息。"),
            ]
        ),
    )
    state = graph.invoke(
        {"messages": [("user", f"订单{order_id}商品坏了")]},
        config={"configurable": {"thread_id": "mismatch"}},
        context=context,
    )
    assert "support_advice" not in state


@pytest.mark.parametrize("raises", [False, True])
def test_rag_empty_or_unavailable_transfers_without_local_fallback(tmp_path, raises):
    class Unavailable:
        def find(self, query):
            if raises:
                raise RuntimeError("private-rag-api-key")
            return None

    gateway = RecordingGateway()
    settings = Settings(after_sales_db_path=tmp_path / "state.sqlite3")
    graph = create_order_agent(
        settings,
        model=ScriptedChatModel(
            responses=[
                tool_call(
                    "query_quality_support", {"category": "headphones", "issue": "左耳没有声音"}
                )
            ]
        ),
        handoff_gateway=gateway,
        support_repository=Unavailable(),
    )
    state = invoke(graph, "我的耳机左耳没有声音")
    assert state["support_phase"] == "transferred"
    assert "support_advice" not in state
    assert "private-rag-api-key" not in str(state)
    assert len(gateway.requests) == 1


@pytest.mark.parametrize(
    "text",
    [
        "我是在引用‘没有解决’，问题其实已经好了",
        "不用转人工，已经解决了",
        "如果我说退货，你会怎么处理？",
    ],
)
def test_quoted_feedback_does_not_trigger_keyword_handoff(tmp_path, text):
    gateway = RecordingGateway()
    graph = create_order_agent(
        Settings(after_sales_db_path=tmp_path / "state.sqlite3"),
        model=ScriptedChatModel(
            responses=[
                tool_call(
                    "query_quality_support", {"category": "headphones", "issue": "左耳没有声音"}
                ),
                AIMessage(content="好的。"),
            ]
        ),
        handoff_gateway=gateway,
    )
    invoke(graph, "我的耳机左耳没有声音")
    state = invoke(graph, text)
    assert state["support_phase"] == "advised"
    assert not gateway.requests


class Selector:
    def __init__(self, selection):
        self.selection = selection

    def invoke(self, messages):
        return self.selection


class FakeRagClient:
    def __init__(self, bound=True):
        self.bound = bound

    def list_datasets(self, **kwargs):
        return [SimpleNamespace(id="dataset", name="售后规则知识库")]

    def list_chats(self, **kwargs):
        return [SimpleNamespace(name="售后助手", dataset_ids=["dataset"] if self.bound else [])]

    def retrieve(self, **kwargs):
        assert kwargs["dataset_ids"] == ["dataset"]
        return [
            SimpleNamespace(
                id="chunk", content="如果支持声道平衡，请恢复居中。", document_name="故障知识.md"
            )
        ]


@pytest.mark.parametrize(
    "excerpt,expected",
    [
        ("如果支持声道平衡，请恢复居中。", True),
        ("拆开耳机焊接。", False),
        ("", False),
    ],
)
def test_rag_answer_must_be_exact_retrieved_evidence(excerpt, expected):
    repository = object.__new__(RagflowKnowledge)
    repository.client = FakeRagClient()
    repository.dataset_name = "售后规则知识库"
    repository.assistant_name = "售后助手"
    repository.selector = Selector(
        RuleSelection(
            applicable=True,
            chunk_id="chunk",
            excerpt=excerpt,
            handling="basic_troubleshooting",
        )
    )
    answer = repository.find(SupportQuery(category="headphones", issue="左耳没有声音"))
    assert (answer is not None) is expected
    repository.client = FakeRagClient(bound=False)
    with pytest.raises(ValueError):
        repository.find(SupportQuery(category="headphones", issue="左耳没有声音"))


def test_sdk_timeout_is_forwarded_without_retry(monkeypatch):
    calls = []

    def request(*args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("orderagent.integrations.ragflow.requests.get", request)
    monkeypatch.setattr("orderagent.integrations.ragflow.requests.post", request)
    client = TimeoutRAGFlow("test-key", "http://localhost", 60)
    client.get("/datasets")
    client.post("/retrieval", json={})
    assert len(calls) == 2
    assert all(call["timeout"] == 60 for call in calls)


def test_existing_order_record_cannot_get_second_advice_after_restart(tmp_path):
    settings = Settings(after_sales_db_path=tmp_path / "state.sqlite3")
    store = AfterSalesRepository(settings.after_sales_db_path)
    store.record(CONTEXT.user_id, ORDER, "已提供一次排查", "old-advice")
    gateway = RecordingGateway()
    graph = create_order_agent(
        settings,
        model=ScriptedChatModel(
            responses=[
                tool_call(
                    "query_quality_support",
                    {
                        "category": "headphones",
                        "issue": "左耳没有声音",
                        "order_id": ORDER,
                    },
                )
            ]
        ),
        handoff_gateway=gateway,
    )
    state = invoke(graph, f"订单{ORDER}耳机仍然没声音")
    assert state["support_phase"] == "transferred"
    assert "support_advice" not in state
    assert len(gateway.requests) == 1
