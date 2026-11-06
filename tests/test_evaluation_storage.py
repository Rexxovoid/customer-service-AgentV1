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
