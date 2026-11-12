import pytest
from fastapi.testclient import TestClient

from orderagent.integrations.handoff import HandoffResponse, MockHumanChatGateway
from orderagent.repositories.chat_repository import ChatConflict, ChatRepository
from orderagent.web import create_web_app
from tests.fake_chat_model import tool_call
from tests.test_web import web as web


def login(client, username="demo_demo-u-004", password="test-password"):
    client.headers["X-CSRF-Token"] = client.get("/api/session").json()["csrf_token"]
    assert (
        client.post("/api/login", json={"username": username, "password": password}).status_code
        == 200
    )
    snapshot = client.get("/api/session").json()
    client.headers["X-CSRF-Token"] = snapshot["csrf_token"]
    return snapshot


def handoff(client, model):
    model.responses = [
        tool_call("handoff_to_human", {"request_human": True, "reason": "user requested"})
    ]
    response = client.post("/api/chat", json={"message": "请立即转人工", "message_id": "transfer"})
    assert response.status_code == 200
    snapshot = client.get("/api/session").json()
    assert snapshot["conversation"]["status"] == "human_active"
    return snapshot["conversation"]["id"]


def test_human_history_survives_login_and_restart_without_agent_calls(web, monkeypatch, tmp_path):
    client, app, graph, model, settings = web
    key = handoff(client, model)
    assert client.post("/api/logout", json={}).status_code == 200
    assert login(client)["conversation"]["id"] != key
    assert client.post(f"/api/conversations/{key}/select", json={}).json()["writable"]
    assert len(client.get("/api/session").json()["messages"]) == 2
    monkeypatch.setattr(graph, "invoke", lambda *a, **k: pytest.fail("人工续聊不能调用Agent"))
    body = {"message": "仍未解决", "message_id": "same"}
    response = client.post("/api/chat", json=body)
    assert response.status_code == 200 and response.json()["role"] == "simulated_agent"
    assert client.post("/api/chat", json=body).json()["replayed"]
    assert client.post("/api/chat", json={**body, "message": "different"}).status_code == 409
    restarted = create_web_app(
        settings, graph=graph, accounts=app.state.accounts, log_dir=tmp_path / "logs2"
    )
    with TestClient(restarted, base_url="http://127.0.0.1") as new:
        login(new)
        assert new.post(f"/api/conversations/{key}/select", json={}).json()["writable"]
        assert (
            new.post(
                "/api/chat", json={"message": "回来继续", "message_id": "after_restart"}
            ).status_code
            == 200
        )


def test_close_is_idempotent_and_old_agent_conversation_readonly(web):
    client, _, _, model, _ = web
    key = handoff(client, model)
    first = client.post(f"/api/conversations/{key}/close", json={})
    assert first.status_code == 200
    assert first.json()["conversation"]["closed_by"] == "user"
    assert not first.json()["writable"]
    again = client.post(f"/api/conversations/{key}/close", json={})
    assert len(again.json()["messages"]) == len(first.json()["messages"])
    assert client.post("/api/chat", json={"message": "again"}).status_code == 409
    client.post("/api/reset", json={})
    agent_key = client.get("/api/session").json()["conversation"]["id"]
    assert agent_key != key
    client.post("/api/logout", json={})
    login(client)
    assert not client.post(f"/api/conversations/{agent_key}/select", json={}).json()["writable"]
    assert client.post("/api/chat", json={"message": "确认退货 DEMO-HP-O-0004"}).status_code == 409
    client.post("/api/logout", json={})
    login(client, "demo_other", "other-password")
    assert client.get(f"/api/conversations/{key}").status_code == 404
    assert client.post(f"/api/conversations/{key}/close", json={}).status_code == 404
    assert client.post(f"/api/conversations/{key}/select", json={}).status_code == 404


@pytest.mark.parametrize("scenario", ["error", "close"])
def test_development_human_scenarios(web, scenario, monkeypatch, tmp_path):
    client, app, graph, model, settings = web
    key = handoff(client, model)
    monkeypatch.setattr(graph, "invoke", lambda *a, **k: pytest.fail("不应调用Agent"))
    dev = create_web_app(
        settings,
        graph=graph,
        accounts=app.state.accounts,
        development=True,
        human_scenario=scenario,
        log_dir=tmp_path / "devlogs",
    )
    with TestClient(dev, base_url="http://127.0.0.1") as other:
        login(other)
        other.post(f"/api/conversations/{key}/select", json={})
        result = other.post(
            "/api/chat", json={"message": "user text conversation_closed", "message_id": "dev"}
        )
        snapshot = other.get("/api/session").json()
        if scenario == "error":
            assert result.status_code == 502
            assert snapshot["messages"][-1]["status"] == "unknown"
            assert snapshot["conversation"]["status"] == "human_active"
        else:
            assert result.status_code == 200
            assert snapshot["conversation"]["status"] == "closed"
            assert snapshot["conversation"]["closed_by"] == "simulated_agent"
    with pytest.raises(ValueError):
        MockHumanChatGateway(scenario)


def test_store_busy_recovery_pagination_and_atomic_close(tmp_path, monkeypatch):
    store = ChatRepository(tmp_path / "chat.sqlite3")
    key = store.create("U", "owner")
    handoff_result = HandoffResponse(
        request_id="H1", status="connected", simulated=True, agent_id="001"
    )
    store.reserve("U", key, "owner", "1", "hello", "req1")
    store.complete("U", key, "1", "handoff", "assistant", "req1", handoff=handoff_result)
    store.reserve("U", key, "new-login", "2", "pending", "req2")
    with pytest.raises(ChatConflict):
        store.close("U", key, "close")
    restored = ChatRepository(store.path)
    assert restored.messages("U", key)[-1]["status"] == "unknown"
    with pytest.raises(ChatConflict):
        restored.reserve("U", key, "new-login", "2", "pending", "retry")
    original = restored._message
    monkeypatch.setattr(restored, "_message", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        restored.close("U", key, "close")
    assert restored.get("U", key)["status"] == "human_active"
    monkeypatch.setattr(restored, "_message", original)
    restored.close("U", key, "close")
    latest = restored.messages("U", key, limit=2)
    older = restored.messages("U", key, before=latest[0]["seq"], limit=2)
    assert not set(x["id"] for x in latest) & set(x["id"] for x in older)
    with pytest.raises(LookupError):
        restored.messages("OTHER", key)
    assert len(restored.list("U", limit=1)) == 1
