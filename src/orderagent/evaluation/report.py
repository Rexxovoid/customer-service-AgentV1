import csv
import hashlib
import json
import math
import re
from collections import Counter

from orderagent.evaluation.schemas import RESULT_STORAGE_VERSION


def archive_case(record, work):
    """执行/评分结束后归档：结论与唯一证据分离，绝不依据排队状态省略本地证据。"""
    from orderagent.evaluation.runner import build_judge_input, sanitize

    evidence = {
        "schema_version": 1,
        "conversation": record.get("judge_input")
        or build_judge_input(record, record.get("reference", "")),
        "tool_calls": [],
        "state_values": {},
        "diagnostics": [],
    }
    result = {
        k: v
        for k, v in record.items()
        if k
        not in {
            "trace",
            "events",
            "diagnostics",
            "evidence_failures",
            "selection_records",
            "dependency_errors",
            "knowledge_misses",
            "judge_input",
            "turns",
            "failed_turn",
            "_failed_runs",
            "rag_answer",
            "error_diagnostic",
        }
    }
    result.update(
        storage_version=RESULT_STORAGE_VERSION,
        experiment_id=work.parent.name,
        executed_turns=len(record.get("turns", [])),
        turns=[],
        trace_summary=[],
    )
    state_keys = {
        "active_support_topic",
        "consultation_id",
        "question_processed_messages",
        "support_phase",
        "support_order_id",
        "support_issue",
        "support_question_count",
        "support_question_scope",
        "support_problem_key",
        "support_error_code",
        "support_handoff",
        "support_handoff_topic",
        "support_question_history",
        "support_advice_history",
        "return_stage",
        "return_order_id",
        "return_reason",
        "return_target_product_id",
        "return_request_count",
        "return_request_history",
        "return_problem_key",
        "return_question_scope",
        "return_has_request",
        "return_error_code",
        "return_confirmation_payload",
        "return_reported_facts",
        "return_fact_diagnostics",
        "return_rule_source",
        "return_information_history",
        "return_application_revision",
        "return_reason_update_pending",
        "return_information_pending",
    }
    previous = {}

    text_keys = {"support_issue", "return_reason"}

    def state_value(value, force_ref=False):
        if (
            isinstance(value, (dict, list))
            or isinstance(value, str)
            and (len(value) > 160 or force_ref)
        ):
            key = hashlib.sha256(
                json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            evidence["state_values"].setdefault(key, value)
            return {"evidence_pointer": f"/state_values/{key}"}
        return value

    for index, turn in enumerate(record.get("turns", [])):
        checks = turn.get("checks", [])
        keys = state_keys | {c["key"] for c in checks if c["kind"] == "state"}
        state = {k: v for k, v in turn.get("state", {}).items() if k in keys}
        changes = {
            k: {
                "before": state_value(previous.get(k), k in text_keys),
                "after": state_value(state.get(k), k in text_keys),
                "removed": k not in state,
            }
            for k in set(state) | set(previous)
            if previous.get(k) != state.get(k) or (k in previous) != (k in state)
        }
        # 保留常用小状态快照，完整绑定/历史用差异及证据引用，而非整份状态副本。
        snapshot = {
            k: v
            for k, v in state.items()
            if k not in text_keys
            and not isinstance(v, (dict, list))
            and not (isinstance(v, str) and len(v) > 160)
        }
        row = {k: turn[k] for k in ("turn", "seconds", "creates", "checks") if k in turn}
        row.update(
            state=snapshot, state_changes=changes, evidence_pointer=f"/conversation/turns/{index}"
        )
        if turn.get("operation") is not None:
            row["operation"] = state_value(turn["operation"])
        calls = [
            call for message in turn.get("messages", []) for call in message.get("tool_calls", [])
        ]
        row["tool_calls"] = []
        for call in calls:
            pointer = f"/tool_calls/{len(evidence['tool_calls'])}"
            evidence["tool_calls"].append({"turn": turn["turn"], **call})
            row["tool_calls"].append({"name": call.get("name"), "evidence_pointer": pointer})
        result["turns"].append(row)
        previous = state

    events = record.get("events")
    if events is None:
        events = [{"name": "rag_diagnostic", "data": d} for d in record.get("diagnostics", [])]
    for index, event in enumerate(events):
        evidence["diagnostics"].append({"diagnostic_id": f"D{index + 1:04}", **event})
    result["diagnostics"] = []
    for name in ("evidence_failures", "selection_records", "dependency_errors", "knowledge_misses"):
        result[name] = []
    for index, event in enumerate(events):
        data = event.get("data", {})
        summary = {
            k: data[k]
            for k in (
                "stage",
                "error_type",
                "error_code",
                "seconds",
                "failure_reasons",
                "status_code",
                "endpoint_template",
            )
            if k in data
        }
        summary.update(diagnostic_id=f"D{index + 1:04}", evidence_pointer=f"/diagnostics/{index}")
        summary.update({k: event[k] for k in ("turn", "request_id") if k in event})
        if data.get("stage") == "selection_metrics":
            summary["usage"] = data.get("usage")
        result["diagnostics"].append(summary)
        misses = {"RAG_NO_CANDIDATES", "RAG_NO_APPLICABLE_RULE"}
        for name, matches in {
            "evidence_failures": data.get("stage") == "selection_rejected",
            "selection_records": "debug_status" in data,
            "dependency_errors": data.get("error_type") and data.get("error_code") not in misses,
            "knowledge_misses": data.get("error_code") in misses,
        }.items():
            if matches:
                result[name].append(
                    {
                        k: summary[k]
                        for k in ("diagnostic_id", "error_code", "evidence_pointer")
                        if k in summary
                    }
                )
    if record.get("rag_answer") is not None:
        evidence["rag_answer"] = record["rag_answer"]
        result["rag_answer_ref"] = "/rag_answer"
    failed = record.get("failed_turn")
    if failed:
        evidence["failed_execution"] = {"user": failed.get("user"), "calls": [], "tool_results": []}
        for run in record.get("_failed_runs", []):
            if run.get("type") == "tool":
                arguments = {
                    k: v
                    for k, v in (run.get("inputs") or {}).items()
                    if k not in {"runtime", "context", "config", "callbacks"}
                }
                if arguments:
                    evidence["failed_execution"]["calls"].append(
                        {"name": run["name"], "args": arguments, "source": "tool_callback"}
                    )
                output = (run.get("outputs") or {}).get("output")
                messages = (
                    [output]
                    if isinstance(output, dict) and "update" not in output
                    else (output.get("update") or {}).get("messages", [])
                    if isinstance(output, dict)
                    else [{"content": output}]
                    if output is not None
                    else []
                )
                for message in messages:
                    value = message.get("artifact")
                    if value is None:
                        value = message.get("content")
                    if value is not None:
                        evidence["failed_execution"]["tool_results"].append(
                            {"name": run["name"], "result": value}
                        )
            if run.get("type") == "llm":
                for group in (run.get("outputs") or {}).get("generations", []):
                    for generation in group:
                        evidence["failed_execution"]["calls"].extend(
                            (generation.get("message") or {}).get("tool_calls", [])
                        )
        result["failed_turn"] = {k: failed[k] for k in ("turn", "seconds") if k in failed}
        result["failed_turn"]["evidence_pointer"] = "/failed_execution"
    else:
        failed_checks = [
            t["turn"] for t in result["turns"] if any(not c["passed"] for c in t.get("checks", []))
        ]
        result["failed_turn"] = {"turn": failed_checks[0]} if failed_checks else None
    if record.get("error_diagnostic"):
        evidence["execution_error"] = record["error_diagnostic"]
        result["error_diagnostic_ref"] = "/execution_error"
        result["error_stage"] = record["error_diagnostic"].get("stage")
        result["execution_error_code"] = record["error_diagnostic"].get("diagnostic_code")
    attempted = [turn["turn"] for turn in result["turns"]]
    if failed and failed.get("turn") not in attempted:
        attempted.append(failed["turn"])
    result["attempted_turns"] = attempted
    result["failed_turns"] = [
        turn["turn"]
        for turn in result["turns"]
        if any(not c["passed"] for c in turn.get("checks", []))
    ]
    if failed and failed.get("turn") not in result["failed_turns"]:
        result["failed_turns"].append(failed["turn"])
    result["skipped_turn_indices"] = [
        i for i in range(1, record.get("planned_turns", len(attempted)) + 1) if i not in attempted
    ]
    result.setdefault("skipped_turns", len(result["skipped_turn_indices"]))

    # 用量只从实际llm回调提取，每次调用选一个来源，避免generations/llm_output双计。
    def find_usage(value):
        if not isinstance(value, dict):
            return None
        for key in ("usage_metadata", "token_usage", "usage"):
            usage = value.get(key)
            if isinstance(usage, dict) and any(
                k in usage
                for k in (
                    "input_tokens",
                    "prompt_tokens",
                    "output_tokens",
                    "completion_tokens",
                    "total_tokens",
                )
            ):
                return usage
        for child in value.values():
            for item in child if isinstance(child, list) else [child]:
                for nested in item if isinstance(item, list) else [item]:
                    usage = find_usage(nested)
                    if usage is not None:
                        return usage
        return None

    usages = []
    for run in record.get("trace", []):
        if run.get("type") not in {"llm", "tool"}:
            continue
        step = {k: run.get(k) for k in ("id", "parent_id", "name", "type", "seconds")}
        if run.get("error"):
            step["error_detail"] = re.sub(r"https?://[^\s]+", "[ENDPOINT]", run["error"])
        if run["type"] == "llm":
            usage = find_usage(run.get("outputs")) or {}
            normalized = {
                name: usage.get(name, usage.get(alias))
                for name, alias in (
                    ("input_tokens", "prompt_tokens"),
                    ("output_tokens", "completion_tokens"),
                    ("total_tokens", "total_tokens"),
                )
            }
            step["usage"] = normalized
            usages.append(normalized)
        result["trace_summary"].append(step)
    result["model_usage"] = {
        "measured_calls": len(usages),
        "calls_with_usage": sum(any(v is not None for v in u.values()) for u in usages),
        **{
            k: sum(u[k] for u in usages)
            if usages and all(u[k] is not None for u in usages)
            else None
            for k in ("input_tokens", "output_tokens", "total_tokens")
        },
        "scope": "framework_llm_calls; SDK HTTP attempts unknown",
    }
    judge_diagnostics = dict(result.get("judge_diagnostics") or {})
    request = judge_diagnostics.pop("_request", None)
    if request:
        evidence["judge_request"] = request
        result["judge_input_ref"] = {
            "pointer": "/conversation",
            "request_pointer": "/judge_request",
            "input_version": "compact-v1",
            "actual_user_sha256": request["user_content_sha256"],
        }
    elif record.get("judge_status") != "disabled":
        result["judge_input_ref"] = {
            "pointer": "/conversation",
            "input_version": "compact-v1",
            "actual_user_sha256": None,
            "request_capture": "unavailable",
        }
    if judge_diagnostics:
        result["judge_diagnostics"] = judge_diagnostics
    result["local_evidence_policy"] = "minimal_per_turn_fallback; never_removed_on_queue_status"

    # 原始诊断中的URL仅保留模板；不保存URL中的查询串、凭据或资源ID。
    def redact_urls(value):
        if isinstance(value, str):
            return re.sub(r"https?://[^\s\"<>]+", "[ENDPOINT]", value)
        if isinstance(value, dict):
            return {k: redact_urls(v) for k, v in value.items()}
        if isinstance(value, list):
            return [redact_urls(v) for v in value]
        return value

    evidence = sanitize(evidence, limit=None)

    def redact_contacts(value):
        if isinstance(value, str):
            if "@" in value:
                value = re.sub(
                    r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", value
                )
            value = re.sub(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)", "[PHONE]", value)
            return re.sub(
                r"(?im)^.*(?:地址|收件人|联系人|身份证|银行卡|cookie|authorization)[:：].*$",
                "[SENSITIVE_LINE]",
                value,
            )
        if isinstance(value, list):
            return [redact_contacts(v) for v in value]
        if isinstance(value, dict):
            return {
                k: redact_contacts(v)
                for k, v in value.items()
                if k.lower() not in {"cookie", "password", "address", "phone", "email"}
            }
        return value

    evidence = redact_contacts(evidence)
    if evidence.get("judge_request"):
        request = evidence["judge_request"]
        request["stored_system_prompt_sha256"] = hashlib.sha256(
            request["system_prompt"].encode()
        ).hexdigest()
        request["system_prompt_matches_actual"] = (
            request["stored_system_prompt_sha256"] == request["system_prompt_sha256"]
        )
    for key in ("judge_diagnostics", "quality", "trace_summary"):
        if key in result:
            result[key] = redact_contacts(result[key])
    evidence["diagnostics"] = redact_urls(evidence["diagnostics"])
    if "execution_error" in evidence:
        evidence["execution_error"] = redact_urls(evidence["execution_error"])
    body = json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    digest = hashlib.sha256(body.encode()).hexdigest()
    conversation_sha = hashlib.sha256(
        json.dumps(evidence["conversation"], ensure_ascii=False).encode()
    ).hexdigest()
    if result.get("judge_input_ref"):
        result["judge_input_ref"].update(
            stored_user_sha256=conversation_sha,
            matches_actual_request=(
                conversation_sha == result["judge_input_ref"]["actual_user_sha256"]
            )
            if result["judge_input_ref"].get("actual_user_sha256")
            else None,
        )
    result["evidence"] = {"path": "evidence.json", "sha256": digest, "schema_version": 1}
    # 哈希对应实际UTF-8字节，避免Windows文本写入转换CRLF造成引用失效。
    (work / "evidence.json").write_bytes(body.encode("utf-8"))
    (work / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def load_evidence(work, result):
    """仅读取固定证据文件并校验哈希，不通过引用跳转任意文件。"""
    reference = result.get("evidence") or {}
    if reference.get("path") != "evidence.json":
        raise ValueError("证据引用缺失或路径不合法")
    body = (work / "evidence.json").read_bytes()
    if hashlib.sha256(body).hexdigest() != reference.get("sha256"):
        raise ValueError("证据哈希不匹配")
    return json.loads(body)


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


def compact_record(record):
    """索引保留汇总/对比所需值，正文、执行树和状态仅保留在案例详情。"""
    fields = (
        "case_id",
        "run_id",
        "mode",
        "category",
        "review_status",
        "repetition_index",
        "status",
        "failure",
        "judge_status",
        "target_seconds",
        "judge_seconds",
        "model_calls",
        "tool_calls",
        "hard_check_count",
        "rag_outcome_code",
        "hit_at_5",
        "error_type",
        "error_code",
        "judge_error_type",
        "knowledge_source",
        "langfuse",
        "experiment_id",
        "executed_turns",
        "planned_turns",
        "skipped_turns",
        "failed_turn",
        "model_usage",
        "evidence",
        "judge_input_ref",
        "local_evidence_policy",
        "error_stage",
        "execution_error_code",
        "attempted_turns",
        "failed_turns",
        "skipped_turn_indices",
    )
    result = {k: record[k] for k in fields if k in record}
    result["storage_version"] = record.get("storage_version", 2)
    result["turns"] = [
        {
            "turn": t.get("turn"),
            "seconds": t.get("seconds"),
            "checks": [
                {k: c.get(k) for k in ("kind", "key", "passed")} for c in t.get("checks", [])
            ],
        }
        for t in record.get("turns", [])
    ]
    result["diagnostics"] = [
        {k: d.get(k) for k in ("stage", "seconds", "usage")}
        for d in record.get("diagnostics", [])
        if d.get("stage") == "selection_metrics"
    ]
    result["dependency_errors"] = [
        {"error_code": d.get("error_code")} for d in record.get("dependency_errors", [])
    ]
    if record.get("quality"):
        result["quality"] = {
            "dimensions": {
                k: {"score": v["score"]} for k, v in record["quality"]["dimensions"].items()
            }
        }
    result["detail_path"] = f"{record['run_id']}/result.json"
    return result


def write_report(root, records):
    """每案例一份详情，根JSONL仅存索引；不采样、不删除、不设保留天数。"""
    indices = []
    for record in records:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", record["run_id"]):
            raise ValueError("Invalid evaluation run_id")
        detail = root / record["run_id"] / "result.json"
        detail.parent.mkdir(exist_ok=True)
        if not detail.exists():
            detail.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        index = compact_record(record)
        index["detail_retention"] = "manual_only"
        index["detail_status"] = "local_available"
        indices.append(index)
    (root / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in indices) + "\n", encoding="utf-8"
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
