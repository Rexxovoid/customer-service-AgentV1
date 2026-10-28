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
    assert len(cases) == 22
    assert sum(c.mode == "live_rag" for c in cases) == 4
    assert all(t.checks for c in cases for t in c.turns)
    with pytest.raises(ValidationError):
        cases[5].model_validate({**cases[5].model_dump(), "fixture": "unknown"})


def test_quality_score_requires_all_five_dimensions():
    with pytest.raises(ValidationError):
        QualityScore.model_validate({"expression": {"score": 4, "reason": "ok", "evidence": "x"}})


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
    assert a["trace"]


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
    assert result["events"][0]["data"]["stage"] == "fixture"
