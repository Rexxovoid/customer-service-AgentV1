import json

import pytest
from langchain_core.messages import AIMessage
from langsmith import tracing_context
from langsmith.evaluation import evaluate
from pydantic import ValidationError

from orderagent.evaluation.evaluators import check_turn
from orderagent.evaluation.report import compare_reviews, summarize, write_report
from orderagent.evaluation.runner import (
    EvaluationReturns,
    execute,
    load_cases,
    sanitize,
)
from orderagent.evaluation.schemas import Check, QualityScore
from orderagent.settings import PROJECT_ROOT, Settings
from tests.fake_chat_model import ScriptedChatModel, tool_call


def test_dataset_load_and_fixtures_are_explicit():
    cases = load_cases(PROJECT_ROOT / "data/evaluation/cases.jsonl")
    assert {c.id for c in cases} == {f"E{number:02d}" for number in range(1, 21)}
    assert sum(c.mode == "live_rag" for c in cases) == 2
    assert all(t.checks for c in cases for t in c.turns)
    with pytest.raises(ValidationError):
        cases[5].model_validate({**cases[5].model_dump(), "fixture": "unknown"})


@pytest.mark.parametrize(
    "review,documents,status,hit",
    [
        ("pending", ["故障知识.md"], "unjudgeable", None),
        ("approved", ["故障知识.md"], "pass", True),
        ("approved", ["不同文档.md"], "fail", False),
        ("approved", [], "unjudgeable", None),
    ],
)
def test_retrieval_only_runs_without_model_and_preserves_metrics(
    tmp_path, monkeypatch, review, documents, status, hit
):
    from orderagent.evaluation.report import compact_record
    from orderagent.evaluation.schemas import Case
    from orderagent.integrations import ragflow
    from tests.test_ragflow_after_sales import FakeRagClient

    monkeypatch.setattr(ragflow, "TimeoutRAGFlow", lambda *args: FakeRagClient())
    monkeypatch.setattr(
        "orderagent.evaluation.runner.create_deepseek_model",
        lambda settings: pytest.fail("纯检索不应创建模型"),
    )
    case = Case(
        id="E67",
        category="retrieval",
        user_id="DEMO-U-004",
        mode="live_rag",
        retrieval_only=True,
        query={"category": "headphones", "issue": "左耳没声音"},
        reference="检索测试，不判断维修方案",
        review_status=review,
        relevant_documents=documents,
    )
    result = execute(
        case,
        Settings(ragflow_api_url="https://rag.example.invalid", ragflow_api_key="test-key"),
        tmp_path,
    )
    assert result["status"] == status
    assert result.get("hit_at_5") == hit
    assert result["model_calls"] == 0 and result["judge_status"] == "disabled"
    assert not result["turns"] and result["retrieval_only"] is True
    assert result["knowledge_source"] == "ragflow_retrieval_only"
    summary = summarize([result])
    assert summary == summarize([compact_record(result)])
    assert summary["retrieval"]["completed_requests"] == 1
    assert summary["retrieval"]["p50_seconds"] is not None
    assert summary["retrieval"]["labeled_runs"] == (0 if hit is None else 1)


def test_retrieval_only_is_for_rag_and_no_gold_does_not_mean_zero_accuracy():
    from orderagent.evaluation.schemas import Case

    with pytest.raises(ValidationError):
        Case(
            id="E67",
            category="x",
            user_id="demo",
            mode="live_agent",
            retrieval_only=True,
            reference="x",
            turns=[{"user": "x", "expect": [], "forbid": []}],
        )
    result = summarize([])["retrieval"]
    assert result["labeled_runs"] == 0 and result["hit_at_5_rate"] is None
    assert result["p50_seconds"] is None and result["raw_candidate_mean"] is None


@pytest.mark.parametrize("judge", [False, True])
def test_retrieval_cli_needs_no_deep_key_and_rejects_judge(tmp_path, monkeypatch, judge):
    import sys

    from orderagent.evaluation import cli
    from orderagent.integrations import ragflow
    from tests.test_ragflow_after_sales import FakeRagClient

    dataset = tmp_path / "retrieval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "E67",
                "category": "retrieval",
                "user_id": "demo",
                "mode": "live_rag",
                "retrieval_only": True,
                "query": {"category": "headphones", "issue": "左耳无声"},
                "reference": "仅检索",
                "review_status": "pending",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        Settings,
        "load",
        classmethod(
            lambda cls: Settings(
                ragflow_api_url="https://rag.example.invalid", ragflow_api_key="test-key"
            )
        ),
    )
    monkeypatch.setattr(ragflow, "TimeoutRAGFlow", lambda *args: FakeRagClient())
    monkeypatch.setattr(
        "orderagent.evaluation.runner.create_deepseek_model",
        lambda settings: pytest.fail("不应初始化模型"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["orderagent-eval", "run", "--cases", str(dataset)] + (["--judge"] if judge else []),
    )
    if judge:
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 2
    else:
        cli.main()
        root = next((tmp_path / "evals/results").iterdir())
        experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        assert experiment["agent_model"] is None and experiment["retrieval_only"]
        assert summary["overall"]["model_calls"] == 0
        assert summary["overall"]["retrieval"]["completed_requests"] == 1


def test_quality_score_requires_all_five_dimensions():
    with pytest.raises(ValidationError):
        QualityScore.model_validate({"expression": {"score": 4, "reason": "ok", "evidence": "x"}})


def test_judge_records_response_metadata_before_validation(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from orderagent.evaluation import evaluators

    client = MagicMock()
    response = SimpleNamespace(
        model="judge-model",
        id="response-1",
        _request_id="request-1",
        usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 123}),
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="{}"))],
    )
    stream = (
        client.__enter__.return_value.chat.completions.stream.return_value.__enter__.return_value
    )
    stream.__iter__.return_value = iter([])
    stream.get_final_completion.return_value = response
    monkeypatch.setattr(evaluators, "OpenAI", lambda **kwargs: client)
    diagnostics = {}
    with pytest.raises(ValidationError):
        evaluators.grade_quality(Settings(qiwen_api_key="test-key"), {}, diagnostics)
    assert diagnostics["finish_reason"] == "stop"
    assert diagnostics["usage"] == {"total_tokens": 123}
    assert diagnostics["request_id"] == "request-1"
    import hashlib

    sent = client.__enter__.return_value.chat.completions.stream.call_args.kwargs
    request = diagnostics["_request"]
    assert sent["messages"][1]["content"] == "{}"
    assert (
        request["user_content_sha256"]
        == hashlib.sha256(sent["messages"][1]["content"].encode()).hexdigest()
    )
    assert request["system_prompt"] == sent["messages"][0]["content"]
    assert sent["max_tokens"] == 2000 and sent["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize(
    "scenario", ["success", "no_usage", "before", "midway", "truncated", "empty", "invalid"]
)
def test_official_judge_stream_receives_and_keeps_partial_metrics(monkeypatch, scenario):
    import httpx2
    from openai import OpenAI

    from orderagent.evaluation import evaluators

    dimensions = {
        name: {"score": 3, "reason": "ok", "evidence": "x"}
        for name in (
            "factual_correctness",
            "faithfulness",
            "relevance",
            "actionability",
            "expression",
        )
    }
    body = json.dumps(dimensions)
    if scenario == "empty":
        body = ""
    if scenario == "invalid":
        body = "{}"

    def respond(request):
        # 官方SDK消费真实SSE格式；不使用自写模型累积器。
        class Bytes(httpx2.SyncByteStream):
            def __iter__(self):
                if scenario == "before":
                    raise httpx2.ReadTimeout("before", request=request)
                values = [
                    {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                    {"choices": [{"index": 0, "delta": {"content": body}}]},
                ]
                for value in values:
                    chunk = {
                        "id": "response-1",
                        "model": "judge",
                        "created": 0,
                        "object": "chat.completion.chunk",
                        **value,
                    }
                    yield ("data: " + json.dumps(chunk) + "\n\n").encode()
                if scenario == "midway":
                    raise httpx2.ReadTimeout("midway", request=request)
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": "response-1",
                            "model": "judge",
                            "created": 0,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "length"
                                    if scenario == "truncated"
                                    else "stop",
                                }
                            ],
                        }
                    )
                    + "\n\n"
                ).encode()
                if scenario != "no_usage":
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": "response-1",
                                "model": "judge",
                                "created": 0,
                                "object": "chat.completion.chunk",
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": 1,
                                    "completion_tokens": 2,
                                    "total_tokens": 3,
                                },
                            }
                        )
                        + "\n\n"
                    ).encode()
                yield b"data: [DONE]\n\n"

        assert json.loads(request.content)["stream"] is True
        return httpx2.Response(200, stream=Bytes())

    monkeypatch.setattr(
        evaluators,
        "OpenAI",
        lambda **kwargs: OpenAI(
            **kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(respond))
        ),
    )
    diagnostics = {}
    if scenario in {"before", "midway", "truncated", "empty", "invalid"}:
        with pytest.raises(Exception):
            evaluators.grade_quality(Settings(qiwen_api_key="test"), {}, diagnostics)
    else:
        result = evaluators.grade_quality(Settings(qiwen_api_key="test"), {}, diagnostics)
        assert result["dimensions"]["expression"]["score"] == 3
        assert (diagnostics["usage"] is None) == (scenario == "no_usage")
    if scenario == "before":
        assert diagnostics["first_event_seconds"] is None
        assert diagnostics["received_events"] is None
    else:
        assert diagnostics["received_events"] >= 2
        assert diagnostics["content_characters"] == len(body)
        assert diagnostics["max_event_gap_seconds"] is not None
        assert (diagnostics["first_content_seconds"] is None) == (not body)


@pytest.mark.parametrize("timeout", [False, True])
def test_judge_compact_input_and_failure_diagnostics(tmp_path, monkeypatch, timeout):
    from types import SimpleNamespace

    from orderagent.evaluation import runner

    case = load_cases(PROJECT_ROOT / "data/evaluation/cases.jsonl")[0]
    # 使用含工具知识证据的固定状态，独立于真实模型及远端知识库。
    from langchain_core.messages import HumanMessage, ToolMessage

    messages = [
        HumanMessage(content=case.turns[0].user, id="user-1"),
        ToolMessage(
            content="duplicate-tool-body",
            tool_call_id="call-1",
            id="tool-1",
            name="query_order",
            artifact={"data": {"source": "test-source", "value": 1}},
        ),
        AIMessage(content="final-answer", id="answer-1"),
    ]
    graph = SimpleNamespace(
        invoke=lambda *args, **kwargs: {
            "messages": messages,
            "support_private_state": "unneeded-state",
        }
    )
    monkeypatch.setattr(runner, "create_order_agent", lambda *args, **kwargs: graph)
    captured = {}

    def grade(settings, record, diagnostics):
        captured.update(record)
        if timeout:
            try:
                raise TimeoutError("read test-key")
            except TimeoutError as cause:
                raise RuntimeError("request test-key") from cause
        diagnostics.update(finish_reason="stop", usage={"total_tokens": 12})
        return {"dimensions": {}}

    monkeypatch.setattr(runner, "grade_quality", grade)
    result = execute(case, Settings(qiwen_api_key="test-key"), tmp_path, judge=True)
    turn = captured["turns"][0]
    assert set(turn) == {"turn", "user", "answer", "tool_results", "expect", "forbid"}
    assert turn["user"] == case.turns[0].user
    assert turn["tool_results"][0]["result"]["data"]["source"] == "test-source"
    assert "duplicate-tool-body" not in json.dumps(captured)
    from pathlib import Path

    from orderagent.evaluation.report import load_evidence

    evidence = load_evidence(Path(result["workdir"]), result)
    assert evidence["conversation"] == captured
    assert "duplicate-tool-body" not in json.dumps(evidence)
    assert "messages" not in result["turns"][0]
    assert "support_private_state" not in result["turns"][0]["state"]
    diagnostic = result["judge_diagnostics"]
    assert diagnostic["total_seconds"] == result["judge_seconds"]
    if timeout:
        assert result["judge_status"] == "error"
        assert diagnostic["cause_type"] == "TimeoutError"
        assert diagnostic["error_detail"] == "request [REDACTED]"
        assert diagnostic["cause_detail"] == "read [REDACTED]"
        assert diagnostic["usage"] is None
        assert diagnostic["attempt_count"] is None
    else:
        assert result["judge_status"] == "completed"
        assert diagnostic["finish_reason"] == "stop"
        assert diagnostic["usage"] == {"total_tokens": 12}


def test_sanitizer_removes_nested_secrets_and_thinking():
    data = {
        "api_key": "secret",
        "messages": [{"reasoning_content": "private", "content": "secret"}],
    }
    assert sanitize(data, ["secret"]) == {"messages": [{"content": "[REDACTED]"}]}
    assert sanitize({"type": "reasoning", "summary": "private"}) == {
        "type": "omitted_private_reasoning"
    }


def test_checks_inspect_actual_tool_argument_not_answer():
    messages = [tool_call("query_order", {"order_id": "wrong"})]
    checks = check_turn(
        [Check(kind="tool_arg", key="query_order.order_id", value="right")],
        {},
        messages,
        EvaluationReturns(),
    )
    assert not checks[0]["passed"]


def test_local_official_evaluate_needs_no_credentials(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    from uuid import uuid4

    from langsmith.schemas import Example

    with tracing_context(enabled=False):
        rows = list(
            evaluate(
                lambda inputs: {"answer": inputs["x"]},
                data=[Example(id=uuid4(), inputs={"x": 1})],
                upload_results=False,
                num_repetitions=2,
                max_concurrency=1,
            )
        )
    assert len(rows) == 2
    assert all(r["run"].outputs["answer"] == 1 for r in rows)


def test_runner_isolates_state_and_stops_midway(tmp_path, monkeypatch):
    from orderagent.agents.agent_factory import create_order_agent
    from orderagent.evaluation import runner

    case = load_cases(PROJECT_ROOT / "data/evaluation/cases.jsonl")[5]
    paths = []

    def build(settings, **kwargs):
        paths.append(settings.after_sales_db_path)
        # 故意不调用工具，必须中途失败且不继续追问。
        return create_order_agent(
            settings,
            model=ScriptedChatModel(responses=[AIMessage(content="故意缺失工具调用的测试回答")]),
            **kwargs,
        )

    monkeypatch.setattr(runner, "create_order_agent", build)
    settings = Settings(after_sales_db_path=tmp_path / "production.sqlite3")
    a = execute(case, settings, tmp_path)
    b = execute(case, settings, tmp_path)
    assert a["status"] == b["status"] == "fail"
    assert len(a["turns"]) == 1 and a["skipped_turns"] == len(case.turns) - 1
    assert paths[0] != paths[1]
    assert not settings.after_sales_db_path.exists()
    assert a["trace_summary"]
    assert "trace" not in a


def sample_record(status="pass"):
    return {
        "run_id": "sample",
        "case_id": "E01",
        "mode": "live_agent",
        "category": "order",
        "turns": [{"answer": "x"}],
        "status": status,
        "failure": None,
        "judge_status": "error",
        "target_seconds": 1.0,
    }


def test_report_denominators_and_human_comparison(tmp_path):
    records = [sample_record(), {**sample_record("error"), "run_id": "other"}]
    summary = summarize(records)
    assert summary["validity_rate"] == 0.5
    assert summary["hard_pass_rate_valid"] == 1
    assert summary["hard_pass_rate_planned"] == 0.5
    assert summary["judge_coverage"] == 0
    write_report(tmp_path, records)
    import csv

    path = tmp_path / "human_review.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0].update(reviewer="tester", verdict="fail", evidence="incorrect parameter")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    result = compare_reviews(tmp_path)
    assert result["comparable"] == 1
    assert result["agreement_rate"] == 0
    assert json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))["reviewed"] == 1


def test_fixture_is_not_live_rag_and_records_events(tmp_path, monkeypatch):
    from orderagent.agents.agent_factory import create_order_agent
    from orderagent.evaluation import runner

    case = load_cases(PROJECT_ROOT / "data/evaluation/cases.jsonl")[6]

    def build(settings, **kwargs):
        model = ScriptedChatModel(
            responses=[
                tool_call("query_quality_support", {"category": "headphones", "issue": "左耳无声"})
            ]
        )
        return create_order_agent(settings, model=model, **kwargs)

    monkeypatch.setattr(runner, "create_order_agent", build)
    result = execute(case, Settings(), tmp_path)
    assert result["status"] == "pass"
    from pathlib import Path

    from orderagent.evaluation.report import load_evidence

    evidence = load_evidence(Path(result["workdir"]), result)
    assert evidence["diagnostics"][0]["data"]["stage"] == "fixture"
