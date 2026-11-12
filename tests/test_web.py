import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from orderagent.accounts import AccountStore, initialize
from orderagent.agents.agent_factory import create_order_agent
from orderagent.repositories.after_sales_repository import AfterSalesRepository
from orderagent.settings import Settings
from orderagent.web import COOKIE_NAME, create_web_app
from tests.fake_chat_model import ScriptedChatModel, tool_call


def read_web_logs(settings, name):
    directory = settings.after_sales_db_path.parent / "logs" / name
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
    ]


@pytest.fixture
def web(tmp_path):
    """官方图和独立SQLite，避免页面测试消耗远端模型或污染日常台账。"""
    settings = Settings(after_sales_db_path=tmp_path / "after_sales.sqlite3")
    model = ScriptedChatModel(responses=[AIMessage(content="测试回答")])
    graph = create_order_agent(settings, model=model)
    accounts = AccountStore(tmp_path / "accounts.sqlite3")
    initialize(
        accounts,
        {
            "DEMO-U-004": {"order_count": 1, "example_order_ids": []},
            "OTHER": {"order_count": 0, "example_order_ids": []},
        },
        tmp_path / "credentials.json",
        True,
    )
    accounts.reset("demo_demo-u-004", "test-password")
    accounts.reset("demo_other", "other-password")
    app = create_web_app(settings, graph=graph, log_dir=tmp_path / "logs", accounts=accounts)
    app.state.accounts = accounts
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/session")
        assert response.status_code == 200
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        assert (
            client.post(
                "/api/login", json={"username": "demo_demo-u-004", "password": "test-password"}
            ).status_code
            == 200
        )
        client.headers["X-CSRF-Token"] = client.get("/api/session").json()["csrf_token"]
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
        assert (
            second.post(
                "/api/login", json={"username": "demo_other", "password": "other-password"}
            ).status_code
            == 200
        )
        second.headers["X-CSRF-Token"] = second.get("/api/session").json()["csrf_token"]
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


def test_login_rotation_logout_and_password_reset_invalidate_session(web):
    client, app, _, _, _ = web
    old = client.cookies.get(COOKIE_NAME)
    assert client.post("/api/logout", json={}).status_code == 200
    assert client.post("/api/chat", json={"message": "hello"}).status_code == 401
    client.cookies.set(COOKIE_NAME, old)
    assert client.post("/api/chat", json={"message": "hello"}).status_code == 401
    client.cookies.clear()
    data = client.get("/api/session").json()
    assert data["authenticated"] is False and data["messages"] == []
    client.headers["X-CSRF-Token"] = data["csrf_token"]
    anonymous = client.cookies.get(COOKIE_NAME)
    assert (
        client.post(
            "/api/login", json={"username": "demo_other", "password": "other-password"}
        ).status_code
        == 200
    )
    assert client.cookies.get(COOKIE_NAME) != anonymous
    data = client.get("/api/session").json()
    assert data["user_id"] == "OTHER"
    client.headers["X-CSRF-Token"] = data["csrf_token"]
    app.state.accounts.reset("demo_other", "changed")
    assert client.post("/api/chat", json={"message": "hello"}).status_code == 401
    assert client.get("/api/session").json()["messages"] == []


def test_login_csrf_failure_throttling_and_secret_free_logs(web):
    client, _, _, _, settings = web
    payload = {"username": "unknown", "password": "LOGIN_SECRET_MARKER"}
    assert (
        client.post(
            "/api/login", json=payload, headers={"Origin": "https://evil.invalid"}
        ).status_code
        == 403
    )
    assert (
        client.post("/api/login", json=payload, headers={"X-CSRF-Token": "wrong"}).status_code
        == 403
    )
    for _ in range(10):
        assert client.post("/api/login", json=payload).status_code == 401
    assert client.post("/api/login", json=payload).status_code == 429
    for row in read_web_logs(settings, "requests"):
        assert "LOGIN_SECRET_MARKER" not in json.dumps(row)
        assert "unknown" not in json.dumps(row)


def test_account_disable_and_session_expiry(web, monkeypatch):
    client, app, _, _, _ = web
    with app.state.accounts.connect() as db:
        db.execute("UPDATE accounts SET enabled=0 WHERE user_id='DEMO-U-004'")
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
    with app.state.accounts.connect() as db:
        db.execute("UPDATE accounts SET enabled=1 WHERE user_id='DEMO-U-004'")
    import orderagent.web as module

    before = module.time.time()
    monkeypatch.setattr(module.time, "time", lambda: before + 3601)
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401


def test_logged_in_other_account_cannot_read_first_users_order(web):
    client, app, _, model, _ = web
    with TestClient(app, base_url="http://127.0.0.1") as other:
        other.headers["X-CSRF-Token"] = other.get("/api/session").json()["csrf_token"]
        other.post("/api/login", json={"username": "demo_other", "password": "other-password"})
        other.headers["X-CSRF-Token"] = other.get("/api/session").json()["csrf_token"]
        model.responses = [
            tool_call("query_order", {"order_id": "DEMO-HP-O-0004"}),
            AIMessage(content="无法查询该订单"),
        ]
        assert other.post("/api/chat", json={"message": "查询 DEMO-HP-O-0004"}).status_code == 200
        tools = [m for m in model._requests[-1] if getattr(m, "type", "") == "tool"]
        assert tools and "ORDER_NOT_FOUND" in tools[-1].content
    assert client.get("/api/session").json()["user_id"] == "DEMO-U-004"


def test_two_accounts_can_chat_concurrently_without_context_or_history_leak(web):
    from threading import Barrier

    first, app, graph, _, _ = web
    barrier = Barrier(2)
    contexts = []
    original = graph.invoke

    def invoke(*args, **kwargs):
        contexts.append(kwargs["context"].user_id)
        barrier.wait(timeout=5)
        return original(*args, **kwargs)

    graph.invoke = invoke
    with TestClient(app, base_url="http://127.0.0.1") as second:
        second.headers["X-CSRF-Token"] = second.get("/api/session").json()["csrf_token"]
        second.post("/api/login", json={"username": "demo_other", "password": "other-password"})
        second.headers["X-CSRF-Token"] = second.get("/api/session").json()["csrf_token"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(first.post, "/api/chat", json={"message": "ACCOUNT_A_ONLY"})
            b = pool.submit(second.post, "/api/chat", json={"message": "ACCOUNT_B_ONLY"})
            assert a.result().status_code == b.result().status_code == 200
        assert set(contexts) == {"DEMO-U-004", "OTHER"}
        assert "ACCOUNT_B_ONLY" not in json.dumps(first.get("/api/session").json()["messages"])
        assert "ACCOUNT_A_ONLY" not in json.dumps(second.get("/api/session").json()["messages"])


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
    requests = read_web_logs(settings, "requests")
    agent = read_web_logs(settings, "agent")[-1]
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
    request = read_web_logs(settings, "requests")[-1]
    agent = read_web_logs(settings, "agent")[-1]
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
    row = read_web_logs(settings, "requests")[-1]
    assert row["request_id"] == response.json()["request_id"]
    assert row["error_type"] == "RequestValidationError"
    assert not model._requests


def test_web_log_files_are_indented_json_and_legacy_logs_are_unchanged(web):
    client, _, _, _, settings = web
    directory = settings.after_sales_db_path.parent / "logs"
    legacy = directory / "requests.jsonl"
    legacy.write_text('{"legacy":true}\n', encoding="utf-8")
    response = client.post("/api/chat", json={"message": "hi"})
    request_id = response.json()["request_id"]
    for name in ("requests", "agent"):
        text = (directory / name / f"{request_id}.json").read_text(encoding="utf-8")
        assert text.startswith('{\n  "request_id":')
        assert json.loads(text)["request_id"] == request_id
        assert not list((directory / name).glob("*.tmp"))
    assert legacy.read_text(encoding="utf-8") == '{"legacy":true}\n'


def test_web_logging_write_failure_does_not_change_response(web, monkeypatch):
    from pathlib import Path

    client, _, _, _, _ = web

    def fail(*args, **kwargs):
        raise OSError("test write failure")

    monkeypatch.setattr(Path, "write_text", fail)
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
    row = read_web_logs(settings, "agent")[-1]
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
    row = read_web_logs(settings, "agent")[-1]
    assert row["state"]["support_phase"] == "transferred"
    tool = next(step for step in row["steps"] if step["type"] == "tool")
    assert tool["action"] == "handoff"
    assert row["rag_diagnostics"][0]["sources"] == [{"id": "chunk-1", "document": "rules.md"}]
    assert row["rag_diagnostics"][1]["error_code"] == "RAG_NO_APPLICABLE_RULE"
    assert "PRIVATE" not in json.dumps(row)
