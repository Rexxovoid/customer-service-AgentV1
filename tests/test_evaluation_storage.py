"""索引去重不删除任何案例详情；人工对比兼容新旧结果。"""

import csv
import json

import pytest

from orderagent.evaluation.report import compare_reviews, summarize, write_report


def sample(run_id="sample", status="pass"):
    return {
        "run_id": run_id,
        "case_id": "E01",
        "mode": "controlled_agent",
        "category": "test",
        "review_status": "pending",
        "status": status,
        "failure": None,
        "judge_status": "completed",
        "quality": {
            "dimensions": {
                name: {"score": score, "reason": "PRIVATE_REASON"}
                for name, score in {
                    "factual_correctness": 3,
                    "faithfulness": None,
                    "relevance": 4,
                    "actionability": 3,
                    "expression": 4,
                }.items()
            }
        },
        "target_seconds": 1.2,
        "turns": [
            {
                "turn": 1,
                "user": "PRIVATE_USER",
                "answer": "PRIVATE_ANSWER",
                "messages": ["PRIVATE_MESSAGES"],
                "state": {"private": "PRIVATE_STATE"},
                "checks": [{"kind": "creates", "key": "", "value": 0, "actual": 0, "passed": True}],
            }
        ],
        "trace": ["PRIVATE_TRACE"],
        "diagnostics": [{"stage": "selection_metrics", "seconds": 0.5, "usage": {"total": 2}}],
        "langfuse": {"trace_id": "test-trace", "content_digest": "digest"},
    }


def test_index_deduplication_preserves_all_details_and_metrics(tmp_path):
    records = [sample(), sample("failed", "fail"), sample("error", "error")]
    write_report(tmp_path, records)
    text = (tmp_path / "results.jsonl").read_text()
    assert "PRIVATE_" not in text
    indices = [json.loads(line) for line in text.splitlines()]
    assert summarize(records) == summarize(indices)
    for row in indices:
        assert row["detail_retention"] == "manual_only"
        assert "PRIVATE_TRACE" in (tmp_path / row["detail_path"]).read_text()
    assert not (tmp_path / "retention.json").exists()


def test_retrieval_summary_keeps_failure_denominators_and_missing_values():
    from orderagent.evaluation.report import compact_record

    row = sample("retrieval-error", "error")
    row["retrieval_only"] = True
    row["diagnostics"] = [
        {"stage": "binding_metrics", "seconds": 0.1, "completed": True},
        {
            "stage": "retrieval_metrics",
            "seconds": 30,
            "completed": False,
            "raw_candidate_count": None,
            "filtered_candidate_count": None,
        },
    ]
    summary = summarize([row])
    assert summary == summarize([compact_record(row)])
    assert summary["planned"] == 1 and summary["target_error_rate"] == 1
    metrics = summary["retrieval"]
    assert metrics["request_samples"] == metrics["failed_requests"] == 1
    assert metrics["completed_requests"] == 0
    assert metrics["p50_seconds"] is None
    assert metrics["raw_candidate_mean"] is None
    assert metrics["labeled_runs"] == 0 and metrics["hit_at_5_rate"] is None


def test_manual_comparison_matches_legacy_and_survives_manual_detail_removal(tmp_path):
    records = [sample()]
    write_report(tmp_path, records)
    path = tmp_path / "human_review.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0].update(
        reviewer="tester", verdict="pass", evidence="reviewed local detail", factual_correctness="4"
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    compact_comparison = compare_reviews(tmp_path)
    assert (
        compact_comparison["items"][0]["score_deltas_human_minus_judge"]["factual_correctness"] == 1
    )
    (tmp_path / "sample/result.json").unlink()  # 模拟用户自己决定删除，不是产品清理功能。
    assert compare_reviews(tmp_path) == compact_comparison
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    assert compare_reviews(tmp_path) == compact_comparison


def test_writer_does_not_overwrite_existing_canonical_result(tmp_path):
    detail = tmp_path / "sample/result.json"
    detail.parent.mkdir()
    detail.write_text('{"preserve": true}')
    write_report(tmp_path, [sample()])
    assert json.loads(detail.read_text()) == {"preserve": True}


def test_index_path_rejects_escape(tmp_path):
    with pytest.raises(ValueError):
        write_report(tmp_path, [sample("../escape")])


def test_verify_command_is_read_only_for_business_results(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from orderagent.evaluation.cli import main

    write_report(tmp_path, [sample()])
    before = (tmp_path / "results.jsonl").read_bytes()
    observer = SimpleNamespace(
        client=None,
        verify_content=lambda _: {"verified": False, "reason": "content_missing_or_incomplete"},
    )
    monkeypatch.setattr("orderagent.observability.get_observability", lambda _: observer)
    monkeypatch.setattr("sys.argv", ["orderagent-eval", "verify", "--result", str(tmp_path)])
    assert main() == 0
    assert (tmp_path / "results.jsonl").read_bytes() == before
    report = json.loads((tmp_path / "langfuse_verification.json").read_text())
    assert not report[0]["cloud_content"]["verified"]
    assert (tmp_path / "sample/result.json").exists()


def archive_input(work, status="pass"):
    import hashlib

    from orderagent.evaluation.runner import build_judge_input

    raw = sample(status=status)
    raw.update(
        workdir=str(work),
        reference="测试依据",
        planned_turns=3,
        model_calls=1,
        tool_calls=1,
        skipped_turns=2,
    )
    raw["turns"][0].update(
        expect=["有依据答复"],
        forbid=["无依据承诺"],
        seconds=1.2,
        creates=0,
        messages=[
            {
                "type": "ai",
                "tool_calls": [
                    {"name": "query_order", "args": {"order_id": "DEMO-HP-O-0004"}, "id": "call-1"}
                ],
            },
            {
                "type": "tool",
                "name": "query_order",
                "content": "DUPLICATE_CONTENT",
                "artifact": {
                    "ok": True,
                    "data": {"order_status": "已签收", "source": {"reference_text": "工具原文"}},
                },
            },
        ],
        state={
            "return_stage": "awaiting_confirmation",
            "return_confirmation_payload": {
                "order_id": "DEMO-HP-O-0004",
                "reason": "左耳无声",
                "facts": {},
            },
            "support_private_state": "OMIT_ME",
        },
        operation={"status": "link_ready", "request_id": "mock-operation"},
    )
    if status == "fail":
        raw["turns"][0]["checks"][0]["passed"] = False
    raw["trace"] = [
        {
            "id": "root",
            "type": "chain",
            "inputs": {"history": "HUGE_HISTORY" * 1000},
            "outputs": {},
        },
        {
            "id": "llm",
            "parent_id": "root",
            "type": "llm",
            "name": "model",
            "seconds": 0.5,
            "outputs": {
                "generations": [
                    [
                        {
                            "message": {
                                "usage_metadata": {
                                    "input_tokens": 3,
                                    "output_tokens": 2,
                                    "total_tokens": 5,
                                }
                            }
                        }
                    ]
                ],
                "llm_output": {
                    "token_usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
                },
            },
        },
        {
            "id": "tool",
            "parent_id": "root",
            "type": "tool",
            "name": "query_order",
            "seconds": 0.1,
            "outputs": {},
        },
    ]
    raw["events"] = [
        {
            "name": "rag_diagnostic",
            "elapsed_seconds": 0.5,
            "data": {
                "stage": "selection_metrics",
                "seconds": 0.5,
                "usage": {"total": 2},
                "detail": "RAW_DIAGNOSTIC_ONLY",
            },
        }
    ]
    raw["diagnostics"] = [e["data"] for e in raw["events"]]
    raw["judge_input"] = build_judge_input(raw, raw["reference"])
    body = json.dumps(raw["judge_input"], ensure_ascii=False)
    raw["judge_diagnostics"] = {
        "_request": {
            "system_prompt": "裁判提示词",
            "system_prompt_sha256": hashlib.sha256("裁判提示词".encode()).hexdigest(),
            "user_content_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "input_version": "compact-v1",
        }
    }
    return raw


@pytest.mark.parametrize("status", ["pass", "fail", "error", "unjudgeable"])
def test_v3_archive_keeps_metrics_assertions_and_single_evidence(tmp_path, status):
    from orderagent.evaluation.report import archive_case, load_evidence

    raw = archive_input(tmp_path, status)
    archived = archive_case(raw, tmp_path)
    evidence = load_evidence(tmp_path, archived)
    assert archived["storage_version"] == 3
    assert summarize([raw]) == summarize([archived])
    assert archived["turns"][0]["checks"] == raw["turns"][0]["checks"]
    assert archived["judge_input_ref"]["matches_actual_request"] is True
    assert evidence["conversation"] == raw["judge_input"]
    assert archived["model_usage"]["total_tokens"] == 5  # 同一回调的两个用量来源不能双计。
    assert archived["turns"][0]["state"]["return_stage"] == "awaiting_confirmation"
    pointer = archived["turns"][0]["state_changes"]["return_confirmation_payload"]["after"][
        "evidence_pointer"
    ]
    assert evidence["state_values"][pointer.split("/")[-1]]["reason"] == "左耳无声"
    assert archived["turns"][0]["operation"]["evidence_pointer"].startswith("/state_values/")
    text = (tmp_path / "result.json").read_text(encoding="utf-8") + (
        tmp_path / "evidence.json"
    ).read_text(encoding="utf-8")
    assert "HUGE_HISTORY" not in text and "DUPLICATE_CONTENT" not in text and "OMIT_ME" not in text
    assert text.count("RAW_DIAGNOSTIC_ONLY") == 1
    assert "trace" not in archived and "messages" not in archived["turns"][0]


def test_v3_partial_usage_and_broken_evidence_are_not_invented(tmp_path):
    from orderagent.evaluation.report import archive_case, load_evidence

    raw = archive_input(tmp_path)
    raw["trace"].append({"id": "missing", "type": "llm", "name": "model", "outputs": {}})
    archived = archive_case(raw, tmp_path)
    assert archived["model_usage"]["calls_with_usage"] == 1
    assert archived["model_usage"]["total_tokens"] is None
    with (tmp_path / "evidence.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="哈希"):
        load_evidence(tmp_path, archived)


def test_http_error_context_drops_credentials_query_and_resource_ids():
    import requests

    from orderagent.integrations.ragflow import http_error_context

    response = requests.Response()
    response.status_code = 403
    response.request = requests.Request(
        "GET",
        "https://user:password@example.com/api/v1/datasets/private-id/documents/another-id?api_key=secret",
    ).prepare()
    error = requests.HTTPError("private response", response=response)
    context = http_error_context(error)
    assert context == {
        "status_code": 403,
        "endpoint_template": "/api/v1/datasets/{dataset_id}/documents/{document_id}",
    }
    response.request.url = "https://example.com/custom/private-id?secret=value"
    assert http_error_context(error)["endpoint_template"] is None


@pytest.mark.parametrize(
    "delivery", ["disabled", "queue_error", "queued_not_verified", "trace_unavailable"]
)
def test_v3_keeps_fallback_evidence_for_all_trace_delivery_states(tmp_path, delivery):
    from orderagent.evaluation.report import archive_case, load_evidence

    raw = archive_input(tmp_path)
    raw["langfuse"] = {"delivery": delivery, "content_complete": False}
    archived = archive_case(raw, tmp_path)
    evidence = load_evidence(tmp_path, archived)
    assert evidence["conversation"]["turns"][0]["user"] == raw["turns"][0]["user"]
    assert evidence["tool_calls"][0]["args"]["order_id"] == "DEMO-HP-O-0004"
    assert evidence["conversation"]["turns"][0]["tool_results"]
    assert archived["status"] == raw["status"]


def test_v3_execution_failure_keeps_partial_tools_without_tree(tmp_path):
    from orderagent.evaluation.report import archive_case, load_evidence

    raw = archive_input(tmp_path, "error")
    raw.update(
        failed_turn={"turn": 2, "user": "失败轮原文", "seconds": 0.8},
        skipped_turns=1,
        error_diagnostic={
            "stage": "agent_execution",
            "diagnostic_code": "EVAL_TARGET_EXCEPTION",
            "status_code": 503,
            "detail": "failed https://user:secret@example.com/private-id?key=secret",
        },
    )
    raw["_failed_runs"] = [
        {
            "type": "tool",
            "name": "handle_return",
            "inputs": {"order_id": "DEMO-HP-O-0004", "runtime": "PRIVATE_RUNTIME"},
            "outputs": {
                "output": {
                    "update": {
                        "messages": [
                            {
                                "artifact": {"ok": False, "error": {"code": "TEST_ERROR"}},
                                "content": "DUPLICATE_FAILED_CONTENT",
                            }
                        ]
                    }
                }
            },
        }
    ]
    archived = archive_case(raw, tmp_path)
    evidence = load_evidence(tmp_path, archived)
    assert archived["attempted_turns"] == [1, 2]
    assert archived["failed_turns"] == [2] and archived["skipped_turn_indices"] == [3]
    assert evidence["failed_execution"]["user"] == "失败轮原文"
    assert evidence["failed_execution"]["calls"][0]["args"] == {"order_id": "DEMO-HP-O-0004"}
    assert (
        evidence["failed_execution"]["tool_results"][0]["result"]["error"]["code"] == "TEST_ERROR"
    )
    assert evidence["execution_error"]["status_code"] == 503
    assert "private-id" not in json.dumps(
        evidence
    ) and "DUPLICATE_FAILED_CONTENT" not in json.dumps(evidence)


def test_v3_redaction_marks_request_difference_instead_of_faking_hash(tmp_path):
    import hashlib

    from orderagent.evaluation.report import archive_case, load_evidence

    raw = archive_input(tmp_path)
    raw["judge_input"]["turns"][0]["user"] = "联系我13812345678 me@example.com"
    raw["judge_diagnostics"]["_request"]["user_content_sha256"] = hashlib.sha256(
        json.dumps(raw["judge_input"], ensure_ascii=False).encode()
    ).hexdigest()
    archived = archive_case(raw, tmp_path)
    evidence = load_evidence(tmp_path, archived)
    assert "13812345678" not in json.dumps(evidence) and "me@example.com" not in json.dumps(
        evidence
    )
    assert archived["judge_input_ref"]["matches_actual_request"] is False


def test_v3_indexes_compare_identically_to_legacy(tmp_path):
    from orderagent.evaluation.report import archive_case

    raw_work = tmp_path / "sample"
    raw_work.mkdir()
    raw = archive_input(raw_work)
    archived = archive_case(raw, raw_work)
    write_report(tmp_path, [archived])
    path = tmp_path / "human_review.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0].update(reviewer="tester", verdict="pass", evidence="checked", factual_correctness="4")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    comparison = compare_reviews(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"] == summarize([raw])
    (tmp_path / "results.jsonl").write_text(json.dumps(raw), encoding="utf-8")
    assert compare_reviews(tmp_path) == comparison
