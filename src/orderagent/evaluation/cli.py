import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from dotenv import dotenv_values
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example

from orderagent.evaluation.report import compare_reviews, write_report
from orderagent.evaluation.runner import execute, fingerprint, load_cases
from orderagent.settings import PROJECT_ROOT, Settings


def main():
    """官方evaluate组织重复与并发；本地案例是单个实验样本，多轮由目标薄适配。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="售后Agent独立测评")
    parser.add_argument("action", choices=["validate", "run", "compare"])
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "data/evaluation/cases.jsonl"))
    parser.add_argument("--ids", help="逗号分隔案例编号")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--upload", action="store_true", help="显式上传LangSmith；详情输入输出隐藏")
    parser.add_argument("--result", help="compare时指定已有实验目录")
    args = parser.parse_args()
    from pathlib import Path

    if not 1 <= args.repeat <= 10 or not 1 <= args.concurrency <= 4:
        parser.error("repeat范围1—10，concurrency范围1—4")
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
    meta = fingerprint(dataset, settings)
    meta.update(
        repetitions=args.repeat,
        concurrency=args.concurrency,
        upload=args.upload,
        selected_ids=[c.id for c in cases],
        judge_enabled=args.judge,
    )
    (root / "experiment.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def target(inputs):
        """单案例目标函数；官方SDK负责实验样本调度。"""
        from orderagent.evaluation.schemas import Case

        return execute(
            Case.model_validate(inputs["case"]),
            settings,
            root,
            judge=args.judge,
            upload=args.upload,
            client=client,
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
                }
            records.append(outputs)
    write_report(root, records)
    if client:
        client.flush()
    print(f"测评完成：{root}\n人工审核：human_review.csv；自动汇总：report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
