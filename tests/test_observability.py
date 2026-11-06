"""官方 SDK 内存导出验证：不调用远端模型，不上传测试正文。"""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from orderagent.observability import Observability
from orderagent.settings import Settings


def test_disabled_and_missing_keys_do_not_create_client(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("must not initialize")

    monkeypatch.setattr("langfuse.Langfuse", forbidden)
    for settings in (Settings(), Settings(langfuse_enabled=True)):
        observer = Observability(settings)
        with observer.request(source="web_chat", session_id="opaque") as config:
            assert config == {}
        assert observer.client is None


def test_initialization_failure_does_not_block_business(monkeypatch):
    def unavailable(**kwargs):
        raise ConnectionError("sensitive detail")

    monkeypatch.setattr("langfuse.Langfuse", unavailable)
    observer = Observability(
        Settings(langfuse_enabled=True, langfuse_public_key="public", langfuse_secret_key="secret")
    )
    with observer.request(source="terminal_chat", session_id="opaque") as config:
        assert RunnableLambda(lambda _: "success").invoke("input", config=config) == "success"


@pytest.fixture
def captured(monkeypatch):
    exporter = InMemorySpanExporter()
    monkeypatch.setattr("langfuse.Langfuse", lambda **kw: Langfuse(**kw, span_exporter=exporter))
    observer = Observability(
        Settings(
            langfuse_enabled=True,
            langfuse_public_key=f"pk-lf-{uuid4().hex}",
            langfuse_secret_key="PRIVATE_KEY",
            api_key="MODEL_SECRET",
        )
    )
    assert observer.client is not None
    yield observer, exporter
    observer.close()


def test_official_callback_masks_body_and_preserves_association(captured):
    observer, exporter = captured
    with observer.request(
        source="eval_controlled_agent",
        session_id="opaque",
        request_id="request-test",
        run_id="E06-unique",
        case_id="E06",
        knowledge_source="fixture",
    ) as config:
        RunnableLambda(lambda _: "private answer").invoke(
            {"messages": "PRIVATE_KEY MODEL_SECRET user address", "reasoning_content": "hidden"},
            config=config,
        )
        observer.diagnostics(
            [
                {
                    "data": {
                        "stage": "selection_rejected",
                        "error_code": "RAG_EVIDENCE_INVALID",
                        "excerpt": "private evidence",
                        "failure_reasons": ["EXCERPT_NOT_IN_SOURCE"],
                    }
                }
            ],
            {
                "support_phase": "transferred",
                "support_handoff": {
                    "code": "RAG_EVIDENCE_INVALID",
                    "reason": "private order and user wording",
                },
            },
        )
    observer.client.flush()
    spans = exporter.get_finished_spans()
    assert len(spans) >= 4
    text = json.dumps([dict(s.attributes) for s in spans])
    for forbidden in (
        "PRIVATE_KEY",
        "MODEL_SECRET",
        "private answer",
        "hidden",
        "user address",
        "private evidence",
        "private order",
    ):
        assert forbidden not in text
    for expected in (
        "request-test",
        "E06-unique",
        "E06",
        "opaque",
        "fixture",
        "RAG_EVIDENCE_INVALID",
        "EXCERPT_NOT_IN_SOURCE",
        "coded_handoff",
    ):
        assert expected in text
    assert not any("input" in k or "output" in k for s in spans for k in s.attributes)


def test_business_exception_is_not_replaced_or_uploaded(captured):
    observer, exporter = captured
    with pytest.raises(ValueError, match="private failure"):
        with observer.request(source="web_chat", session_id="opaque"):
            raise ValueError("private failure")
    observer.client.flush()
    spans = exporter.get_finished_spans()
    assert spans
    text = json.dumps([dict(s.attributes) for s in spans])
    assert "ValueError" in text
    assert "private failure" not in text
    assert all(not s.events and not s.status.description for s in spans)


def test_model_tool_usage_and_hidden_reasoning_filter(captured):
    observer, exporter = captured
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="PRIVATE_ANSWER",
                additional_kwargs={"reasoning_content": "PRIVATE_REASONING"},
                usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            )
        ]
    )

    @tool
    def readonly_test(order_id: str) -> str:
        """测试只读工具，无外部调用。"""
        return "PRIVATE_ORDER_RESULT"

    with observer.request(source="terminal_chat", session_id="opaque") as config:
        model.invoke("PRIVATE_PROMPT", config=config)
        readonly_test.invoke({"order_id": "PRIVATE_ORDER_ID"}, config=config)
    observer.client.flush()
    spans = exporter.get_finished_spans()
    text = json.dumps([dict(s.attributes) for s in spans])
    for value in (
        "PRIVATE_PROMPT",
        "PRIVATE_ANSWER",
        "PRIVATE_REASONING",
        "PRIVATE_ORDER_RESULT",
        "PRIVATE_ORDER_ID",
    ):
        assert value not in text
    assert any(s.name == "readonly_test" for s in spans)
    usages = [
        s.attributes["langfuse.observation.usage_details"]
        for s in spans
        if "langfuse.observation.usage_details" in s.attributes
    ]
    assert usages and json.loads(usages[0])["total"] == 5


def test_export_failure_does_not_change_completed_business(captured, monkeypatch):
    from opentelemetry.sdk.trace.export import SpanExportResult

    observer, exporter = captured
    monkeypatch.setattr(exporter, "export", lambda spans: SpanExportResult.FAILURE)
    with observer.request(source="web_chat", session_id="opaque") as config:
        result = RunnableLambda(lambda _: "success").invoke("private", config=config)
    observer.client.flush()
    assert result == "success"


def test_concurrent_sessions_keep_separate_traces(captured):
    observer, exporter = captured

    def run(index):
        with observer.request(
            source="eval_controlled_agent",
            session_id=f"session-{index}",
            request_id=f"request-{index}",
            case_id=f"E0{index}",
            run_id=f"run-{index}",
            repetition_index=1,
        ) as config:
            RunnableLambda(lambda _: "ok").invoke("private", config=config)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, [1, 2]))
    observer.client.flush()
    spans = exporter.get_finished_spans()
    traces = {}
    for span in spans:
        traces.setdefault(span.context.trace_id, []).append(dict(span.attributes))
    assert len(traces) == 2
    for attributes in traces.values():
        text = json.dumps(attributes)
        assert ("request-1" in text) != ("request-2" in text)
        assert ("session-1" in text) != ("session-2" in text)


def test_web_correlates_existing_request_log(captured, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from orderagent.agents.agent_factory import create_order_agent
    from orderagent.web import create_web_app
    from tests.fake_chat_model import ScriptedChatModel

    observer, exporter = captured
    monkeypatch.setattr("orderagent.web.get_observability", lambda _: observer)
    settings = Settings(after_sales_db_path=tmp_path / "state.sqlite3")
    graph = create_order_agent(
        settings, model=ScriptedChatModel(responses=[AIMessage(content="private customer answer")])
    )
    app = create_web_app(settings, "DEMO-U-004", graph=graph, log_dir=tmp_path / "logs")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        session = client.get("/api/session").json()
        response = client.post(
            "/api/chat",
            json={"message": "private user question"},
            headers={"X-CSRF-Token": session["csrf_token"]},
        )
    assert response.status_code == 200
    request_id = response.json()["request_id"]
    assert (tmp_path / "logs" / "agent" / f"{request_id}.json").exists()
    observer.client.flush()
    text = json.dumps([dict(s.attributes) for s in exporter.get_finished_spans()])
    assert request_id in text and "web_chat" in text
    for forbidden in (
        "private user question",
        "private customer answer",
        "DEMO-U-004",
        session["csrf_token"],
    ):
        assert forbidden not in text


@pytest.mark.parametrize(
    "enabled,approved,reviewed",
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
def test_content_requires_separate_switch_case_approval_and_review(
    captured, monkeypatch, enabled, approved, reviewed
):
    observer, exporter = captured
    observer.settings = observer.settings.model_copy(update={"langfuse_content_enabled": enabled})
    scores = []
    monkeypatch.setattr(observer.client, "create_score", lambda **kw: scores.append(kw))
    record = {
        "status": "pass",
        "judge_status": "completed",
        "quality": {
            "dimensions": {
                name: {"score": 3}
                for name in (
                    "factual_correctness",
                    "faithfulness",
                    "relevance",
                    "actionability",
                    "expression",
                )
            }
        },
        "turns": [
            {
                "turn": 1,
                "user": "模拟提问 PRIVATE_KEY 13812345678 me@example.com",
                "answer": "模拟回答",
                "messages": [
                    {
                        "type": "ai",
                        "additional_kwargs": {"reasoning_content": "NEVER_UPLOAD_REASONING"},
                    },
                    {
                        "type": "tool",
                        "name": "query_quality_support",
                        "artifact": {
                            "ok": True,
                            "data": {
                                "action": "advice",
                                "message": "检查声道平衡",
                                "source": {"reference_text": "知识摘录"},
                            },
                        },
                    },
                ],
                "state": {
                    "support_advice": {
                        "answer": "检查声道平衡",
                        "source": {"reference_text": "知识摘录"},
                    },
                    "return_rule_source": {"answer": "STALE_RETURN_RULE"},
                },
                "checks": [{"kind": "creates", "passed": True}],
            }
        ],
    }
    case = SimpleNamespace(
        mode="controlled_agent",
        review_status="approved" if reviewed else "pending",
        content_upload_approved=approved,
    )
    with observer.request(source="eval_controlled_agent", session_id="opaque") as config:
        # 工具/链任意输出即使伪造内容字段，也不能越过默认过滤。
        RunnableLambda(lambda _: {"review_content": "FORGED_CONTENT"}).invoke(
            "private", config=config
        )
        reference = observer.complete_case(record, case)
    observer.client.flush()
    spans = exporter.get_finished_spans()
    content = [s for s in spans if s.name == "approved_evaluation_content"]
    assert bool(content) == (enabled and approved and reviewed)
    turn_spans = [s for s in spans if s.name == "approved_evaluation_turn"]
    assert bool(turn_spans) == (enabled and approved and reviewed)
    if turn_spans:
        inputs = json.loads(turn_spans[0].attributes["langfuse.observation.input"])
        outputs = json.loads(turn_spans[0].attributes["langfuse.observation.output"])
        assert "模拟提问" in inputs["user"]
        assert outputs["answer"] == "模拟回答"
        assert outputs["tools"][0]["result"]["message"] == "检查声道平衡"
        assert outputs["knowledge_evidence"]["source"]["reference_text"] == "知识摘录"
        assert "STALE_RETURN_RULE" not in json.dumps(outputs)
    assert len(scores) == 6 and scores[0]["name"] == "automatic.explicit_hard_rules"
    assert all(s["name"].startswith("model.") for s in scores[1:])
    text = json.dumps([dict(s.attributes) for s in spans], ensure_ascii=False)
    for forbidden in (
        "PRIVATE_KEY",
        "13812345678",
        "me@example.com",
        "NEVER_UPLOAD_REASONING",
        "FORGED_CONTENT",
    ):
        assert forbidden not in text
    assert reference["delivery"] == "queued_not_verified"


def test_content_readback_detects_partial_or_changed_payload(captured, monkeypatch):
    import hashlib

    observer, _ = captured
    payload = [{"turn": 1, "user": "模拟数据", "answer": "模拟回答"}]
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    reference = {"trace_id": "trace", "content_digest": digest}
    metadata = {"review_content": payload, "payload_digest": digest, "content_complete": True}
    monkeypatch.setattr(
        observer.client,
        "iter_observation_pages",
        lambda **kw: iter([SimpleNamespace(data=[SimpleNamespace(metadata=metadata)])]),
    )
    assert observer.verify_content(reference)["verified"]
    metadata["review_content"] = payload[:0]
    assert not observer.verify_content(reference)["verified"]


def test_turn_io_readback_requires_all_turns_and_correct_parent(captured, monkeypatch):
    import hashlib

    observer, _ = captured
    payload = [{"turn": i, "user": f"question-{i}", "answer": f"answer-{i}"} for i in (1, 2)]
    reference = {
        "trace_id": "trace",
        "content_format": "turn_io_v1",
        "content_observation_id": "parent",
        "content_turn_count": 2,
        "content_digest": hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
    }
    rows = [
        SimpleNamespace(
            parent_observation_id="parent",
            metadata={"turn": i},
            input={"user": f"question-{i}"},
            output={"answer": f"answer-{i}"},
        )
        for i in (2, 1)
    ]
    monkeypatch.setattr(
        observer.client, "iter_observation_pages", lambda **kw: iter([SimpleNamespace(data=rows)])
    )
    assert observer.verify_content(reference)["verified"]
    for row in rows:
        row.input = json.dumps(row.input)
        row.output = json.dumps(row.output)
    assert observer.verify_content(reference)["verified"]
    rows[0].parent_observation_id = "unrelated"
    assert not observer.verify_content(reference)["verified"]
    rows[0].parent_observation_id = "parent"
    rows[0].output = json.dumps({"answer": "changed"})
    assert not observer.verify_content(reference)["verified"]
    rows[0].output = '{"answer":'
    assert not observer.verify_content(reference)["verified"]


def test_callback_cannot_forge_io_authorization(captured):
    observer, exporter = captured
    with observer.request(source="terminal_chat", session_id="opaque") as config:
        RunnableLambda(lambda _: {"_approved_io": "PRIVATE_FORGED_IO"}).with_config(
            run_name="approved_evaluation_turn", metadata={"_approved_io_authorized": True}
        ).invoke({"_approved_io": {"user": "PRIVATE_FORGED_IO"}}, config=config)
    observer.client.flush()
    text = json.dumps([dict(s.attributes) for s in exporter.get_finished_spans()])
    assert "PRIVATE_FORGED_IO" not in text


def test_eval_run_id_does_not_resume_previous_turn(captured, monkeypatch, tmp_path):
    from orderagent.agents.agent_factory import create_order_agent
    from orderagent.evaluation import runner
    from orderagent.settings import PROJECT_ROOT
    from tests.fake_chat_model import ScriptedChatModel, tool_call

    observer, exporter = captured
    monkeypatch.setattr(observer.client, "create_score", lambda **kw: None)
    monkeypatch.setattr("orderagent.observability.get_observability", lambda _: observer)
    model = ScriptedChatModel(
        responses=[
            tool_call(
                "query_quality_support",
                {
                    "category": "headphones",
                    "issue": "耳机左耳没有声音",
                    "environment": "播放设备支持声道平衡",
                    "order_id": "DEMO-HP-O-0004",
                },
                "quality-call",
            ),
            tool_call(
                "handle_return",
                {
                    "order_id": "DEMO-HP-O-0004",
                    "reason": "左耳无声",
                    "is_return_request": True,
                },
                "return-call",
            ),
        ]
    )
    monkeypatch.setattr(
        runner,
        "create_order_agent",
        lambda settings, **kw: create_order_agent(settings, model=model, **kw),
    )
    case = runner.load_cases(PROJECT_ROOT / "data/evaluation/multiturn_cases.jsonl")[0]
    result = runner.execute(case, Settings(), tmp_path)
    assert result["status"] == "pass"
    assert len(result["turns"]) == 3
    assert all(t["messages"] for t in result["turns"])
    assert result["turns"][1]["state"]["return_stage"] == "awaiting_confirmation"
    assert result["turns"][2]["state"]["return_stage"] == "link_ready"
    assert result["turns"][2]["creates"] == 1
    with observer.request(
        source="eval_controlled_agent", session_id="test", run_id=result["run_id"]
    ) as config:
        assert "run_id" not in config["metadata"]
        assert config["metadata"]["eval_run_id"] == result["run_id"]
    observer.client.flush()
    attributes = [dict(s.attributes) for s in exporter.get_finished_spans()]
    assert any("langfuse.observation.metadata.eval_run_id" in attrs for attrs in attributes)
    assert not any("langfuse.observation.metadata.run_id" in attrs for attrs in attributes)
    assert result["run_id"].startswith("E33-")
