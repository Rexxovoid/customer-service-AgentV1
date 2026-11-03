import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from orderagent.agents.agent_factory import create_order_agent
from orderagent.repositories.after_sales_repository import AfterSalesRepository
from orderagent.settings import Settings
from orderagent.web import COOKIE_NAME, create_web_app
from tests.fake_chat_model import ScriptedChatModel, tool_call


@pytest.fixture
def web(tmp_path):
    """官方图和独立SQLite，避免页面测试消耗远端模型或污染日常台账。"""
    settings = Settings(after_sales_db_path=tmp_path / "after_sales.sqlite3")
    model = ScriptedChatModel(responses=[AIMessage(content="测试回答")])
    graph = create_order_agent(settings, model=model)
    app = create_web_app(settings, "DEMO-U-004", graph=graph, log_dir=tmp_path / "logs")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/session")
        assert response.status_code == 200
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        yield client, app, graph, model, settings


def test_web_chat_and_refresh_preserve_multi_turn_history(web):
    client, _, _, model, _ = web
    assert client.post("/api/chat", json={"message": "你好"}).json()["answer"] == "测试回答"
    assert client.post("/api/chat", json={"message": "继续咨询"}).status_code == 200
    history = client.get("/api/session").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["text"] == "你好"
    assert any(m.text == "你好" for m in model._requests[-1])


def test_web_reuses_order_tool_and_server_identity(web):
    client, _, _, model, _ = web
    model.responses = [
        tool_call("query_order", {"order_id": "DEMO-HP-O-0004"}),
        AIMessage(content="订单 DEMO-HP-O-0004 已查询，以上是本地模拟快照。"),
    ]
    answer = client.post("/api/chat", json={"message": "查询订单 DEMO-HP-O-0004"})
    assert answer.status_code == 200
    assert "DEMO-HP-O-0004" in answer.json()["answer"]
    assert client.get("/api/session").json()["user_id"] == "DEMO-U-004"


@pytest.mark.parametrize(
    "body",
    [
        {"message": " "},
        {"message": "x" * 4001},
        {"message": "hi", "user_id": "OTHER"},
        {"message": "hi", "thread_id": "OTHER"},
    ],
)
def test_web_rejects_invalid_messages_and_identity_overrides(web, body):
    client, _, _, model, _ = web
    assert client.post("/api/chat", json=body).status_code == 422
    assert model._requests == []


def test_web_isolates_browser_threads(web):
    client, app, _, model, _ = web
    client.post("/api/chat", json={"message": "第一位访客的消息"})
    with TestClient(app, base_url="http://127.0.0.1") as second:
        data = second.get("/api/session").json()
        assert data["messages"] == []
        second.headers["X-CSRF-Token"] = data["csrf_token"]
        second.post("/api/chat", json={"message": "第二位访客的消息"})
        assert all(m.text != "第一位访客的消息" for m in model._requests[-1])
    assert len(client.get("/api/session").json()["messages"]) == 2


def test_web_reset_clears_chat_but_retains_business_records(web):
    client, _, _, _, settings = web
    repo = AfterSalesRepository(settings.after_sales_db_path)
    repo.record("DEMO-U-004", "DEMO-HP-O-0004", "refunded", "test")
    client.post("/api/chat", json={"message": "你好"})
    assert client.post("/api/reset", json={}).status_code == 200
    assert client.get("/api/session").json()["messages"] == []
    assert repo.get_status("DEMO-U-004", "DEMO-HP-O-0004") == "refunded"


def test_web_security_and_static_assets(web):
    client, _, _, _, _ = web
    assert (
        client.post(
            "/api/chat", json={"message": "你好"}, headers={"Origin": "https://other.example"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/chat", json={"message": "你好"}, headers={"X-CSRF-Token": "wrong"}
        ).status_code
        == 403
    )
    assert client.get("/api/session", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400
    page = client.get("/")
    assert page.status_code == 200
    assert "unsafe-inline" not in page.headers["content-security-policy"]
    for name in ("chat.css", "chat.js"):
        assert client.get(f"/static/{name}").status_code == 200
    assert client.get("/static/../settings.py").status_code == 404
    client.cookies.clear()
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401


def test_web_error_is_not_retried_or_exposed(web):
    client, _, _, model, _ = web
    model._fail_next = True
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 502
    assert "private-provider-detail" not in response.text
    assert model._fail_next is False


def test_web_rejects_concurrent_chat_and_reset(web, monkeypatch):
    client, _, graph, _, _ = web
    entered, release = Event(), Event()
    original = graph.invoke

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(graph, "invoke", blocked)
    with ThreadPoolExecutor() as pool:
        task = pool.submit(client.post, "/api/chat", json={"message": "first"})
        try:
            assert entered.wait(5)
            assert client.post("/api/chat", json={"message": "second"}).status_code == 409
            assert client.post("/api/reset", json={}).status_code == 409
        finally:
            release.set()
        assert task.result().status_code == 200


def test_web_unknown_cookie_does_not_select_existing_thread(web):
    client, _, _, _, _ = web
    client.post("/api/chat", json={"message": "private history"})
    client.cookies.clear()
    client.cookies.set(COOKIE_NAME, "untrusted-thread")
    assert client.get("/api/session").json()["messages"] == []


def test_web_logs_correlate_requests_and_agent_without_message_text(web):
    client, _, _, model, settings = web
    model.responses = [
        tool_call("query_order", {"order_id": "DEMO-HP-O-0004"}),
        AIMessage(content="PRIVATE-ANSWER"),
    ]
    response = client.post("/api/chat", json={"message": "PRIVATE-QUESTION"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]
    assert response.headers["X-Request-ID"] == request_id
    log_dir = settings.after_sales_db_path.parent / "logs"
    requests = [
        json.loads(line)
        for line in (log_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    agent = json.loads((log_dir / "agent.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    request = next(row for row in requests if row["request_id"] == request_id)
    assert request["http_status"] == 200 and request["seconds"] >= 0
    assert request["session_id"] == agent["session_id"]
    assert agent["request_id"] == request_id
    assert any(step["type"] == "llm" for step in agent["steps"])
    tool = next(step for step in agent["steps"] if step["type"] == "tool")
    assert tool["name"] == "query_order" and tool["ok"] is True
    assert tool["seconds"] >= 0
    text = json.dumps([requests, agent])
    for private in (
        "PRIVATE-QUESTION",
        "PRIVATE-ANSWER",
        "DEMO-HP-O-0004",
        "DEMO-U-004",
        client.cookies.get(COOKIE_NAME),
    ):
        assert private not in text


def test_web_logs_failure_cause_and_error_id(web, monkeypatch):
    client, _, graph, _, settings = web

    def fail(*args, **kwargs):
        try:
            raise TimeoutError("PRIVATE-CAUSE")
        except TimeoutError as exc:
            raise RuntimeError("PRIVATE-ERROR") from exc

    monkeypatch.setattr(graph, "invoke", fail)
    response = client.post("/api/chat", json={"message": "PRIVATE-INPUT"})
    assert response.status_code == 502
    request_id = response.json()["request_id"]
    assert request_id == response.headers["X-Request-ID"]
    log_dir = settings.after_sales_db_path.parent / "logs"
    request = json.loads((log_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    agent = json.loads((log_dir / "agent.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert request["request_id"] == agent["request_id"] == request_id
    assert request["error_type"] == "RuntimeError"
    assert request["cause_type"] == "TimeoutError"
    assert request["root_cause_type"] == "TimeoutError"
    assert request["cause_category"] == "timeout"
    assert "PRIVATE" not in json.dumps([request, agent, response.json()])


def test_web_logs_validation_rejection_with_no_agent_call(web):
    client, _, _, model, settings = web
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422
    row = json.loads(
        (settings.after_sales_db_path.parent / "logs/requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )
    assert row["request_id"] == response.json()["request_id"]
    assert row["error_type"] == "RequestValidationError"
    assert not model._requests


def test_web_logging_write_failure_does_not_change_response(web, monkeypatch):
    from logging.handlers import RotatingFileHandler

    client, _, _, _, _ = web

    def fail(*args, **kwargs):
        raise OSError("test write failure")

    monkeypatch.setattr(RotatingFileHandler, "emit", fail)
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 200
    assert response.json()["answer"] == "测试回答"


def test_web_logs_knowledge_id_without_showing_source_to_customer(web, monkeypatch):
    from orderagent.repositories.support_repository import JsonSupportRepository
    from orderagent.schemas.support_schema import SupportKnowledge

    client, _, _, model, settings = web
    monkeypatch.setattr(
        JsonSupportRepository,
        "find",
        lambda self, query: SupportKnowledge(
            id="knowledge-source-1",
            category="headphones",
            question="issue",
            answer="PRIVATE-EVIDENCE",
            customer_reply="请先检查设备连接。",
            applicability="PRIVATE-CONDITIONS",
            source={"reference_text": "PRIVATE-SOURCE"},
            data_origin="ragflow_retrieval",
        ),
    )
    model.responses = [
        tool_call(
            "query_quality_support",
            {
                "category": "headphones",
                "issue": "issue",
            },
        )
    ]
    response = client.post("/api/chat", json={"message": "issue"})
    assert response.json()["answer"] == "请先检查设备连接。"
    path = settings.after_sales_db_path.parent / "logs/agent.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    tool = next(step for step in row["steps"] if step["type"] == "tool")
    assert tool["knowledge"] == {"id": "knowledge-source-1", "data_origin": "ragflow_retrieval"}
    assert "PRIVATE" not in json.dumps(row)


def test_web_logs_tool_result_and_rag_diagnostics_without_evidence_text(web, monkeypatch):
    from langchain_core.callbacks.manager import dispatch_custom_event

    from orderagent.repositories.support_repository import JsonSupportRepository

    client, _, _, model, settings = web

    def no_rule(self, query):
        dispatch_custom_event(
            "rag_diagnostic",
            {
                "stage": "retrieved",
                "candidates": [
                    {
                        "id": "chunk-1",
                        "document": "rules.md",
                        "content": "PRIVATE-EVIDENCE",
                    }
                ],
                "query": "PRIVATE-QUERY",
            },
        )
        dispatch_custom_event(
            "rag_diagnostic",
            {
                "stage": "no_applicable_rule",
                "error_code": "RAG_NO_APPLICABLE_RULE",
            },
        )
        return None

    monkeypatch.setattr(JsonSupportRepository, "find", no_rule)
    model.responses = [
        tool_call(
            "query_quality_support",
            {
                "category": "headphones",
                "issue": "PRIVATE-SYMPTOM",
            },
        )
    ]
    response = client.post("/api/chat", json={"message": "PRIVATE-SYMPTOM"})
    assert response.status_code == 200
    path = settings.after_sales_db_path.parent / "logs/agent.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["state"]["support_phase"] == "transferred"
    tool = next(step for step in row["steps"] if step["type"] == "tool")
    assert tool["action"] == "handoff"
    assert row["rag_diagnostics"][0]["sources"] == [{"id": "chunk-1", "document": "rules.md"}]
    assert row["rag_diagnostics"][1]["error_code"] == "RAG_NO_APPLICABLE_RULE"
    assert "PRIVATE" not in json.dumps(row)
