import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from threading import Lock
from uuid import uuid4

from dotenv import dotenv_values
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example

from orderagent.evaluation.report import archive_case, compare_reviews, load_evidence, write_report
from orderagent.evaluation.runner import execute, fingerprint, load_cases
from orderagent.settings import PROJECT_ROOT, Settings


def main():
    """官方evaluate组织重复与并发；本地案例是单个实验样本，多轮由目标薄适配。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="售后Agent独立测评")
    parser.add_argument("action", choices=["validate", "run", "compare", "verify"])
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "data/evaluation/cases.jsonl"))
    parser.add_argument("--ids", help="逗号分隔案例编号")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--upload", action="store_true", help="显式上传LangSmith；详情输入输出隐藏")
    parser.add_argument("--result", help="compare/verify时指定已有实验目录")
    args = parser.parse_args()
    from pathlib import Path

    if not 1 <= args.repeat <= 10 or not 1 <= args.concurrency <= 4:
        parser.error("repeat范围1—10，concurrency范围1—4")
    if args.action == "verify":
        if not args.result:
            parser.error("verify需要--result")
        from orderagent.observability import get_observability

        root = Path(args.result).resolve()
        observer = get_observability(Settings.load())
        rows = [
            json.loads(line)
            for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        verification = []
        for row in rows:
            local_evidence = {"verified": None, "reason": "legacy_format_not_checked"}
            if row.get("storage_version") == 3:
                try:
                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", row["run_id"]):
                        raise ValueError("非法运行编号")
                    load_evidence(root / row["run_id"], row)
                    local_evidence = {"verified": True, "reason": "file_hash_matched"}
                except (OSError, ValueError) as exc:
                    local_evidence = {"verified": False, "reason": type(exc).__name__}
            proof = observer.verify_content(row.get("langfuse") or {})
            trace_id = (row.get("langfuse") or {}).get("trace_id")
            trace_url = None
            if trace_id and observer.client is not None:
                try:
                    trace_url = observer.client.get_trace_url(trace_id=trace_id)
                except Exception:
                    pass
            verification.append(
                {
                    "run_id": row["run_id"],
                    "case_id": row["case_id"],
                    "cloud_content": proof,
                    "verification_scope": (
                        "specified_turn_content_only; "
                        "all_spans_scores_usage_not_checked"
                    ),
                    "local_evidence": local_evidence,
                    "trace_id": trace_id,
                    "trace_url": trace_url,
                }
            )
        (root / "langfuse_verification.json").write_text(
            json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        return 0
    if args.action == "compare":
        if not args.result:
            parser.error("compare需要--result")
        print(json.dumps(compare_reviews(Path(args.result)), ensure_ascii=False, indent=2))
        return 0
    dataset = Path(args.cases).resolve()
    cases = load_cases(dataset)
    if args.ids:
        ids = set(args.ids.split(","))
        if ids - {c.id for c in cases}:
            parser.error("存在未知案例编号")
        cases = [c for c in cases if c.id in ids]
    if args.action == "validate":
        approved = sum(c.review_status == "approved" for c in cases)
        print(f"格式有效：{len(cases)}条；已人工确认{approved}条")
        return 0
    settings = Settings.load()
    retrieval_only = all(case.retrieval_only for case in cases)
    if any(case.retrieval_only for case in cases) and args.judge:
        parser.error("纯检索没有最终答复，不能使用--judge；完整链路请使用原验收集")
    if any(case.retrieval_only for case in cases) and (
        not settings.ragflow_api_url or not settings.ragflow_api_key.get_secret_value()
    ):
        parser.error("纯检索需要RAGFLOW_API_URL及RAGFLOW_API_KEY，不回退本地JSON")
    if not retrieval_only:
        settings.require_api_key()
    if args.judge and not settings.qiwen_api_key.get_secret_value():
        parser.error("--judge需要QIWEN_API_KEY")
    client = None
    if args.upload:
        env = dotenv_values(PROJECT_ROOT / ".env")
        key = os.environ.get("LANGSMITH_API_KEY") or env.get("LANGSMITH_API_KEY")
        if not key:
            parser.error("--upload需要LANGSMITH_API_KEY")
        # 不上传详细模型/工具输入输出或元数据；本地保存脱敏证据。
        client = Client(api_key=key, hide_inputs=True, hide_outputs=True, hide_metadata=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    root = PROJECT_ROOT / "evals/results" / f"{stamp}-{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "cases.snapshot.jsonl").write_bytes(dataset.read_bytes())
    meta = fingerprint(dataset, settings, include_model=not retrieval_only)
    meta.update(
        repetitions=args.repeat,
        concurrency=args.concurrency,
        upload=args.upload,
        selected_ids=[c.id for c in cases],
        judge_enabled=args.judge,
        retrieval_only=retrieval_only,
        dataset_hash=meta["files"][str(dataset)],
        storage_policy="manual",
    )
    (root / "experiment.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    repetition_counts = {}
    repetition_lock = Lock()

    def target(inputs):
        """单案例目标函数；官方SDK负责实验样本调度。"""
        from orderagent.evaluation.schemas import Case

        case_id = inputs["case"]["id"]
        with repetition_lock:
            repetition_counts[case_id] = repetition_counts.get(case_id, 0) + 1
            repetition_index = repetition_counts[case_id]
        return execute(
            Case.model_validate(inputs["case"]),
            settings,
            root,
            judge=args.judge,
            upload=args.upload,
            client=client,
            repetition_index=repetition_index,
        )

    def hard_rule_score(outputs):
        status = outputs.get("status")
        hard = {
            "key": "explicit_hard_rules",
            "score": (1 if status == "pass" else 0 if status == "fail" else None),
        }
        return {
            "results": [
                hard,
                *[
                    {"key": f"quality_{name}", "score": dim["score"]}
                    for name, dim in outputs.get("quality", {}).get("dimensions", {}).items()
                ],
            ]
        }

    from langsmith import tracing_context

    examples = [Example(id=uuid4(), inputs={"case": c.model_dump()}) for c in cases]
    if args.upload:
        cloud_dataset = client.create_dataset(dataset_name=f"orderagent-{root.name}")
        examples = [
            client.create_example(inputs=e.inputs, dataset_id=cloud_dataset.id) for e in examples
        ]
    with tracing_context(enabled=args.upload, client=client):
        results = evaluate(
            target,
            data=examples,
            evaluators=[hard_rule_score],
            num_repetitions=args.repeat,
            max_concurrency=args.concurrency,
            upload_results=args.upload,
            client=client,
            experiment_prefix=root.name,
            metadata={"dataset_hash": meta["files"][str(dataset)]},
        )
        records = []
        for item in results:
            outputs = item["run"].outputs
            if not outputs or "case_id" not in outputs:
                # 调度器/目标初始化异常也必须保留在计划分母。
                case = item["example"].inputs["case"]
                outputs = {
                    "case_id": case["id"],
                    "run_id": uuid4().hex,
                    "mode": case["mode"],
                    "category": case["category"],
                    "turns": [],
                    "status": "error",
                    "failure": "evaluation_runner",
                    "judge_status": "disabled",
                    "target_seconds": 0,
                    "planned_turns": len(case.get("turns", [])),
                    "reference": case.get("reference", ""),
                    "knowledge_source": case.get("fixture") or "configured_rag_or_json",
                    "skipped_turns": len(case.get("turns", [])),
                    "error_diagnostic": {
                        "stage": "evaluation_runner",
                        "diagnostic_code": "EVAL_RUNNER_FAILURE",
                        "detail": "执行初始化或调度失败",
                    },
                }
                work = root / outputs["run_id"]
                work.mkdir(exist_ok=False)
                outputs = archive_case(outputs, work)
            records.append(outputs)
    from orderagent.observability import get_observability

    observer = get_observability(settings)
    with observer.request(
        source="evaluation_experiment",
        session_id=root.name,
        experiment_id=root.name,
        case_trace_ids=[r.get("langfuse", {}).get("trace_id") for r in records],
    ):
        pass
    write_report(root, records)
    if client:
        client.flush()
    print(f"测评完成：{root}\n人工审核：human_review.csv；自动汇总：report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
