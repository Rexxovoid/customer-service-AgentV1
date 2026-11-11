from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from orderagent.agents.agent_factory import create_order_agent
from orderagent.integrations.ragflow import RagflowKnowledge, RuleSelection, TimeoutRAGFlow
from orderagent.repositories.after_sales_repository import AfterSalesRepository
from orderagent.schemas.order_schema import ToolContext
from orderagent.schemas.support_schema import SupportQuery
from orderagent.settings import Settings
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


def test_quality_queries_do_not_register_order_after_sales(tmp_path):
    settings = Settings(after_sales_db_path=tmp_path / "state.sqlite3")
    store = AfterSalesRepository(settings.after_sales_db_path)
    gateway = RecordingGateway()
    graph = create_order_agent(
        settings,
        model=ScriptedChatModel(
            responses=[
                tool_call(
                    "query_quality_support",
                    {"category": "headphones", "issue": "左耳没有声音", "order_id": ORDER},
                )
            ]
        ),
        handoff_gateway=gateway,
    )
    invoke(graph, f"订单{ORDER}，我的耳机左耳没有声音")
    assert store.get_status(CONTEXT.user_id, ORDER) == "未售后"
    state = invoke(graph, "没有解决")
    assert state["support_phase"] == "awaiting_information"
    invoke(graph, "转人工")
    assert store.get_status(CONTEXT.user_id, ORDER) == "未售后"
    graph.checkpointer.delete_thread("persist")
    assert store.get_status(CONTEXT.user_id, ORDER) == "未售后"


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
    assert store.get_status("A", "O") == "未售后"
    assert store.get_status("B", "O") == "未售后"
    store.record("A", "O", "已转人工（模拟）", "event-3")
    store.record("A", "O", "已提供一次排查", "late-event")
    assert store.get_status("A", "O") == "未售后"


def test_consultation_claim_api_is_removed(tmp_path):
    store = AfterSalesRepository(tmp_path / "state.sqlite3")
    assert not hasattr(store, "start_issue")
    store.record("A", "O", "已退款", "trusted-completion")
    store.record("A", "O", "退货申请待确认", "late-request")
    assert store.get_status("A", "O") == "已退款"


@pytest.mark.parametrize(
    "order_id,category",
    [
        ("DEMO-HP-O-0001", "headphones"),
        (ORDER, "mice"),
    ],
)
def test_order_snapshot_is_not_quality_evidence_but_category_still_checked(
    tmp_path, order_id, category
):
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
    assert ("support_advice" in state) is (category == "headphones")


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
    assert not state.get("support_advice")
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


def test_retrieval_only_measures_same_candidate_window_and_skips_selector():
    repository = object.__new__(RagflowKnowledge)
    repository.client = FakeRagClient()
    repository.dataset_name = "售后规则知识库"
    repository.assistant_name = "售后助手"
    repository.selector = None
    events = []
    repository._diagnostic = events.append
    rows = [
        SimpleNamespace(id="fault", document_name="MS_02_故障条件问答.md", content="故障"),
        SimpleNamespace(id="policy", document_name="MS_03_售后规则场景_01.md", content="政策"),
        SimpleNamespace(id="empty", document_name="empty", content=""),
    ]
    parameters = []
    repository.client.retrieve = lambda **kwargs: (parameters.append(kwargs), rows)[1]
    answer = repository.find(
        SupportQuery(category="mice", issue="退货规则与适用条件", task="return"),
        retrieval_only=True,
    )
    assert answer is None
    metrics = next(e for e in events if e["stage"] == "retrieval_metrics")
    assert metrics["completed"] is True and metrics["seconds"] >= 0
    assert metrics["raw_candidate_count"] == 3
    assert metrics["nonempty_candidate_count"] == 2
    assert metrics["filtered_candidate_count"] == 1
    assert parameters[0]["page_size"] == 5 and parameters[0]["top_k"] == 64
    assert parameters[0]["similarity_threshold"] == 0.3
    retrieved = next(e for e in events if e["stage"] == "retrieved")
    assert retrieved["retrieval_question"] == parameters[0]["question"]
    assert retrieved["similarity_threshold"] == parameters[0]["similarity_threshold"]
    assert retrieved["filtered_candidate_ids"] == ["policy"]
    assert not any(e["stage"].startswith("selection") for e in events)


@pytest.mark.parametrize("failure", ["binding", "request", "empty"])
def test_retrieval_metrics_distinguish_failed_request_empty_and_binding(failure):
    import requests

    repository = object.__new__(RagflowKnowledge)
    repository.client = FakeRagClient(bound=failure != "binding")
    repository.dataset_name = "售后规则知识库"
    repository.assistant_name = "售后助手"
    repository.selector = None
    events = []
    repository._diagnostic = events.append
    if failure == "request":

        def fail(**kwargs):
            raise requests.Timeout("timeout")

        repository.client.retrieve = fail
    elif failure == "empty":
        repository.client.retrieve = lambda **kwargs: []
    if failure == "empty":
        repository.find(SupportQuery(category="mice", issue="未知问题"), retrieval_only=True)
    else:
        with pytest.raises(ValueError, match="RAG_BINDING_MISMATCH|RAG_RETRIEVAL_TIMEOUT"):
            repository.find(SupportQuery(category="mice", issue="未知问题"), retrieval_only=True)
    metrics = [e for e in events if e["stage"] == "retrieval_metrics"]
    if failure == "binding":
        assert not metrics
        assert any(e.get("completed") is False for e in events if e["stage"] == "binding_metrics")
    elif failure == "request":
        assert metrics[0]["completed"] is False
        assert metrics[0]["raw_candidate_count"] is None
    else:
        assert metrics[0]["completed"] is True
        assert metrics[0]["raw_candidate_count"] == 0
        assert metrics[0]["filtered_candidate_count"] == 0


@pytest.mark.parametrize("reply", ["", " ", "知识依据：故障知识.md / chunk"])
def test_rag_quality_never_uses_raw_excerpt_when_customer_reply_invalid(reply):
    repository = object.__new__(RagflowKnowledge)
    repository.client = FakeRagClient()
    repository.dataset_name = "售后规则知识库"
    repository.assistant_name = "售后助手"
    repository.selector = Selector(
        RuleSelection(
            applicable=True,
            chunk_id="chunk",
            paragraph_start=1,
            paragraph_end=1,
            handling="basic_troubleshooting",
            customer_reply=reply,
        )
    )
    with pytest.raises(ValueError, match="RAG_REPLY_INVALID"):
        repository.find(SupportQuery(category="headphones", issue="左耳没有声音"))


def test_quality_customer_reply_is_separate_from_raw_evidence(tmp_path):
    from langchain_core.messages import ToolMessage

    evidence = (
        "适用条件：Windows 10/11的USB鼠标。\n基础答复：重新插入USB，换端口，集线器改直连。\n"
        "处理边界：按当前内部次数规则转人工。"
    )
    reply = (
        "在Windows电脑上，可以依次尝试：\n1. 重新插入鼠标USB连接。\n"
        "2. 换一个USB接口。\n3. 使用集线器时改为直连电脑。"
    )
    repository = object.__new__(RagflowKnowledge)
    repository.client = FakeRagClient()
    repository.client.retrieve = lambda **kwargs: [
        SimpleNamespace(
            id="source-id",
            content=evidence,
            document_name="MS_02_故障条件问答.md",
        )
    ]
    repository.dataset_name = "售后规则知识库"
    repository.assistant_name = "售后助手"
    repository.selector = Selector(
        RuleSelection(
            applicable=True,
            chunk_id="source-id",
            paragraph_start=1,
            paragraph_end=1,
            handling="basic_troubleshooting",
            customer_reply=reply,
        )
    )
    model = ScriptedChatModel(
        responses=[
            tool_call(
                "query_quality_support",
                {
                    "category": "mice",
                    "issue": "左键没反应",
                    "environment": "Windows 11 USB",
                },
            )
        ]
    )
    graph = create_order_agent(
        Settings(after_sales_db_path=tmp_path / "state.sqlite3"),
        model=model,
        support_repository=repository,
    )
    state = invoke(graph, "我的鼠标左键没反应，Windows 11，通过USB连接")
    assert state["messages"][-1].text == reply
    assert len(model._requests) == 1
    assert state["support_advice"]["answer"] == evidence
    tool = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert tool.artifact["data"]["source"]["reference_text"].endswith("source-id")
    assert tool.artifact["data"]["answer"] == evidence
    assert "source-id" not in tool.content
    assert "处理边界" not in tool.content


@pytest.mark.parametrize(
    "end,expected",
    [
        (1, True),
        (99, False),
        (99, False),
    ],
)
def test_rag_answer_uses_only_valid_candidate_range(end, expected):
    repository = object.__new__(RagflowKnowledge)
    repository.client = FakeRagClient()
    repository.dataset_name = "售后规则知识库"
    repository.assistant_name = "售后助手"
    repository.selector = Selector(
        RuleSelection(
            applicable=True,
            chunk_id="chunk",
            paragraph_start=1,
            paragraph_end=end,
            handling="basic_troubleshooting",
            customer_reply="如果支持声道平衡，请恢复居中。",
        )
    )
    if expected:
        assert (
            repository.find(SupportQuery(category="headphones", issue="左耳没有声音")) is not None
        )
    else:
        with pytest.raises(ValueError, match="RAG_EVIDENCE_INVALID"):
            repository.find(SupportQuery(category="headphones", issue="左耳没有声音"))
    repository.client = FakeRagClient(bound=False)
    with pytest.raises(ValueError):
        repository.find(SupportQuery(category="headphones", issue="左耳没有声音"))


def test_sdk_timeout_is_forwarded_without_retry(monkeypatch):
    calls = []

    def request(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr("orderagent.integrations.ragflow.requests.get", request)
    monkeypatch.setattr("orderagent.integrations.ragflow.requests.post", request)
    client = TimeoutRAGFlow("test-key", "http://localhost", 60)
    client.get("/datasets")
    client.post("/retrieval", json={})
    assert len(calls) == 2
    assert all(call["timeout"] == 60 for call in calls)


def test_legacy_consultation_does_not_block_quality_after_restart(tmp_path):
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
                    {"category": "headphones", "issue": "左耳没有声音", "order_id": ORDER},
                )
            ]
        ),
        handoff_gateway=gateway,
    )
    state = invoke(graph, f"订单{ORDER}耳机仍然没声音")
    assert state["support_phase"] == "advised"
    assert state["support_question_count"] == 1
    assert not gateway.requests
    assert store.get_status(CONTEXT.user_id, ORDER) == "未售后"
