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
    app = create_web_app(settings, "DEMO-U-004", graph=graph)
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
