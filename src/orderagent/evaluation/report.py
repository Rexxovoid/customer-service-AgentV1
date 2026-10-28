import csv
import json
import math
from collections import Counter


def percentile(values, quantile):
    """使用nearest-rank；空样本返回null，小样本不宣称容量结论。"""
    return sorted(values)[max(0, math.ceil(len(values) * quantile) - 1)] if values else None


def summarize(records):
    """业务有效率、总体成功比例、裁判覆盖率分别使用明确分母。"""
    total = len(records)
    valid = sum(r["status"] in {"pass", "fail"} for r in records)
    passed = sum(r["status"] == "pass" for r in records)
    judged = sum(r["judge_status"] == "completed" for r in records)
    eligible = sum(bool(r["turns"]) for r in records)
    latencies = [r["target_seconds"] for r in records]
    selection_metrics = [
        d
        for r in records
        for d in r.get("diagnostics", [])
        if d.get("stage") == "selection_metrics"
    ]
    selection_latencies = [d["seconds"] for d in selection_metrics]
    quality = {}
    for dim in ("factual_correctness", "faithfulness", "relevance", "actionability", "expression"):
        scores = [
            r["quality"]["dimensions"][dim]["score"]
            for r in records
            if r.get("quality") and r["quality"]["dimensions"][dim]["score"] is not None
        ]
        quality[dim] = {"count": len(scores), "mean": sum(scores) / len(scores) if scores else None}
    return {
        "planned": total,
        "valid": valid,
        "passed": passed,
        "validity_rate": valid / total if total else None,
        "hard_pass_rate_valid": passed / valid if valid else None,
        "hard_pass_rate_planned": passed / total if total else None,
        "judge_coverage": judged / eligible if eligible else None,
        "judge_completed": judged,
        "judge_eligible": eligible,
        "target_error_rate": sum(r["status"] == "error" for r in records) / total
        if total
        else None,
        "p50_seconds": percentile(latencies, 0.5),
        "p95_seconds": percentile(latencies, 0.95),
        "selection_p50_seconds": percentile(selection_latencies, 0.5),
        "selection_p95_seconds": percentile(selection_latencies, 0.95),
        "selection_usage_available": sum(d.get("usage") is not None for d in selection_metrics),
        "selection_truncations": sum(
            d.get("error_code") == "RAG_SELECTION_TRUNCATED"
            for r in records
            for d in r.get("dependency_errors", [])
        ),
        "note_safety_rates": "错误放行/误转人工需人工金标及明确断言，不能从通用通过率推算。",
        "failures": dict(Counter(r["failure"] for r in records if r.get("failure"))),
        "rag_outcomes": dict(
            Counter(r["rag_outcome_code"] for r in records if r.get("rag_outcome_code"))
        ),
        "judge_errors": sum(r["judge_status"] == "error" for r in records),
        "quality": quality,
        "model_calls": sum(r.get("model_calls", 0) for r in records),
        "tool_calls": sum(r.get("tool_calls", 0) for r in records),
    }


def write_report(root, records):
    """结果不可覆盖人工表；按模式分类汇总并创建待填写的独立审核表。"""
    (root / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    summary = {"overall": summarize(records), "groups": {}, "stability": {}}
    for case_id in sorted({r["case_id"] for r in records}):
        subset = [r for r in records if r["case_id"] == case_id]
        summary["stability"][case_id] = {
            "runs": len(subset),
            "statuses": dict(Counter(r["status"] for r in subset)),
            "note": "少于2次无法评价重复稳定性" if len(subset) < 2 else "同配置重复运行观察",
        }
    for mode, category in sorted({(r["mode"], r["category"]) for r in records}):
        summary["groups"][f"{mode}/{category}"] = summarize(
            [r for r in records if r["mode"] == mode and r["category"] == category]
        )
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "run_id",
        "case_id",
        "turn",
        "reviewer",
        "verdict",
        "evidence",
        "reason",
        "factual_correctness",
        "faithfulness",
        "relevance",
        "actionability",
        "expression",
    ]
    with (root / "human_review.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            # 裁判按整场打分；turn=all与人工整场结论比较，逐轮可另外添加。
            writer.writerow({"run_id": r["run_id"], "case_id": r["case_id"], "turn": "all"})
    lines = [
        "# 自动测评报告",
        "",
        "硬规则分数仅覆盖显式断言，不代表全部自然语言要求通过。",
        "RAG未标注题为unjudgeable；异常仍留在计划分母。人工结论保存在独立CSV，不自动覆盖。",
        "",
        "## 汇总",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 案例",
        "",
        "|案例|运行|模式|硬规则结果|归因|裁判|",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"|{r['case_id']}|{r['run_id']}|{r['mode']}|{r['status']}|"
            f"{r.get('failure') or ''}|{r['judge_status']}|"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_reviews(root):
    """保留人工原文，比较整场结论及逐维评分；拒绝重复或未知运行编号。"""
    records = {
        r["run_id"]: r
        for r in map(json.loads, (root / "results.jsonl").read_text(encoding="utf-8").splitlines())
    }
    with (root / "human_review.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    compared = []
    seen = set()
    for row in rows:
        if not row["verdict"]:
            continue
        key = (row["run_id"], row["turn"], row["reviewer"])
        if key in seen or row["run_id"] not in records:
            raise ValueError("人工表存在重复审核或未知run_id")
        seen.add(key)
        if row["verdict"] not in {"pass", "fail", "unjudgeable"}:
            raise ValueError("verdict仅允许pass/fail/unjudgeable")
        if not row["reviewer"] or not row["evidence"]:
            raise ValueError("已审核行需要reviewer及evidence")
        record = records[row["run_id"]]
        if row["case_id"] != record["case_id"]:
            raise ValueError("case_id与run_id不匹配")
        deltas = {}
        dimensions = (
            "factual_correctness",
            "faithfulness",
            "relevance",
            "actionability",
            "expression",
        )
        for dim in dimensions:
            if row.get(dim) and int(row[dim]) not in range(5):
                raise ValueError("人工维度评分必须为0至4")
        for dim, value in record.get("quality", {}).get("dimensions", {}).items():
            if row.get(dim):
                score = int(row[dim])
                if score not in range(5):
                    raise ValueError("人工维度评分必须为0至4")
                deltas[dim] = score - value["score"] if value["score"] is not None else None
        compared.append(
            {
                "human": row,
                "automatic_hard": record["status"],
                "comparable": row["turn"] == "all"
                and row["verdict"] in {"pass", "fail"}
                and record["status"] in {"pass", "fail"},
                "agreement": row["verdict"] == record["status"],
                "score_deltas_human_minus_judge": deltas,
            }
        )
    comparable = [r for r in compared if r["comparable"]]
    result = {
        "reviewed": len(compared),
        "comparable": len(comparable),
        "agreement_rate": (
            sum(r["agreement"] for r in comparable) / len(comparable) if comparable else None
        ),
        "items": compared,
        "note": "人工整场结论范围可能比机器断言广；不一致需复核，不能直接认定人工或机器错误",
    }
    (root / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
