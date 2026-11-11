import hashlib
import json
import re
import subprocess
import time
from importlib.metadata import version
from uuid import uuid4

from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langsmith import tracing_context
from pydantic import ValidationError

from orderagent.agents.agent_factory import create_order_agent
from orderagent.evaluation.evaluators import check_turn, grade_quality
from orderagent.evaluation.schemas import Case
from orderagent.integrations.ragflow import RagflowKnowledge, http_error_context
from orderagent.integrations.returns import MockReturnGateway
from orderagent.models import create_deepseek_model
from orderagent.repositories.after_sales_repository import AfterSalesRepository
from orderagent.schemas.order_schema import ToolContext
from orderagent.schemas.support_schema import SupportKnowledge, SupportQuery
from orderagent.settings import PROJECT_ROOT


def sanitize(value, secrets=(), *, limit=16000):
    """持久化前移除凭据及推理字段，限制长字段；不保存原始异常堆栈。"""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        if value.get("type") in {"reasoning", "thinking", "redacted_thinking"}:
            return {"type": "omitted_private_reasoning"}
        return {
            str(k): sanitize(v, secrets, limit=None if k == "_request" else limit)
            for k, v in value.items()
            if not any(
                x in str(k).lower()
                for x in (
                    "api_key",
                    "authorization",
                    "secret",
                    "reasoning",
                    "thinking",
                    "signature",
                    "encrypted",
                )
            )
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(v, secrets, limit=limit) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value[:limit] if limit is not None else value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:500]


class Collector(RunCollectorCallbackHandler):
    """复用官方RunCollector，仅增加项目RAG诊断事件的本地保存。"""

    def __init__(self):
        super().__init__()
        self.events = []
        self.started = time.perf_counter()

    def on_custom_event(self, name, data, *, run_id, **kwargs):
        self.events.append(
            {
                "name": name,
                "data": data,
                "run_id": str(run_id),
                "elapsed_seconds": time.perf_counter() - self.started,
            }
        )


class EvaluationReturns(MockReturnGateway):
    """计数官方图调用的模拟创建接口；不替代业务幂等逻辑。"""

    def __init__(self):
        super().__init__()
        self.requests = []
        self.last_operation = None

    def create_link(self, request, now):
        self.requests.append(request.model_dump())
        return super().create_link(request, now)


class FixtureKnowledge:
    """显式受控规则与故障，仅供测评；不能成为生产RAG降级路径。"""

    def __init__(self, name):
        self.name = name

    def find(self, query):
        dispatch_custom_event("rag_diagnostic", {"stage": "fixture", "fixture": self.name})
        if self.name == "rag_timeout":
            raise TimeoutError("测评注入超时")
        if self.name in {"no_applicable_rule", "handoff_error_and_no_rule"}:
            return None
        is_return = query.issue.startswith("退货规则与适用条件")
        if self.name == "basic_headphone_rule" and is_return:
            return None
        return SupportKnowledge(
            id="EVAL-FIXTURE-RULE",
            category=query.category,
            question=query.issue,
            answer=(
                "模拟规则：该用户描述支持提供模拟退货申请入口，须明确确认。"
                if is_return
                else "播放设备支持声道平衡时，请检查左右声道平衡是否居中，再测试左右耳。"
            ),
            applicability="仅供受控测评；非真实平台政策",
            source={
                "handling": "return_eligible" if is_return else "basic_troubleshooting",
                "reference_text": "独立测评固定夹具",
            },
            data_origin="synthetic_test_fixture",
        )


def flatten_runs(runs):
    """导出官方执行树中的步骤、耗时、错误类别与消息证据。"""
    result = []
    for run in runs:
        result.append(
            {
                "id": str(run.id),
                "parent_id": str(run.parent_run_id),
                "name": run.name,
                "type": run.run_type,
                "seconds": (
                    (run.end_time - run.start_time).total_seconds() if run.end_time else None
                ),
                "error": run.error.splitlines()[-1][:1000] if run.error else None,
                "inputs": run.inputs,
                "outputs": run.outputs,
            }
        )
        result.extend(flatten_runs(run.child_runs))
    return result


def load_cases(path):
    """严格解析并拒绝重复编号；完整数据集不要求金标已确认。"""
    cases = [
        Case.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("案例编号重复")
    return cases


def build_judge_input(record, reference):
    """唯一compact-v1输入构建器：裁判和归档复用同一内容，不增加上下文。"""
    return {
        "reference": reference,
        "knowledge_source": record.get("knowledge_source"),
        "turns": [
            {
                "turn": turn["turn"],
                "user": turn["user"],
                "answer": turn["answer"],
                "tool_results": [
                    {
                        "name": message.get("name"),
                        "result": message.get("artifact")
                        if message.get("artifact") is not None
                        else message.get("content"),
                    }
                    for message in turn["messages"]
                    if message.get("type") == "tool"
                ],
                "expect": turn["expect"],
                "forbid": turn["forbid"],
            }
            for turn in record.get("turns", [])
        ],
    }


def fingerprint(dataset, settings, *, include_model=True):
    """只记录配置白名单与文件哈希；不导出Settings或.env。"""
    paths = [
        *PROJECT_ROOT.joinpath("src").rglob("*.py"),
        *PROJECT_ROOT.joinpath("src").rglob("*.yaml"),
        dataset,
        settings.orders_path,
        settings.products_path,
        settings.support_path,
        PROJECT_ROOT / "uv.lock",
        PROJECT_ROOT / "pyproject.toml",
    ]

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    model = create_deepseek_model(settings) if include_model else None
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "versions": {
            p: version(p)
            for p in (
                "deepagents",
                "langchain",
                "langchain-core",
                "langgraph",
                "langsmith",
                "openai",
                "langfuse",
            )
        },
        "agent_model": settings.model_name if include_model else None,
        "judge_model": settings.qianwen,
        "observability": {
            "langfuse_enabled": settings.langfuse_enabled,
            "payload_policy": "metadata-with-reviewed-content-v2"
            if settings.langfuse_content_enabled
            else "metadata-only-v2",
            "content_enabled": settings.langfuse_content_enabled,
            "plan": settings.langfuse_plan,
        },
        "agent_parameters": {
            "max_tokens": model.max_tokens,
            "max_retries": model.max_retries,
            "timeout_seconds": model.request_timeout,
            "temperature": model.temperature,
        }
        if model is not None
        else None,
        "judge_parameters": {
            "stream": True,
            "max_tokens": 2000,
            "temperature": 0,
            "timeout_seconds": 30,
            "max_retries": 1,
        },
        "rag": {
            "dataset": settings.ragflow_dataset_name,
            "assistant": settings.ragflow_assistant_name,
            "timeout": settings.ragflow_timeout_seconds,
            "selection_max_tokens": settings.ragflow_selection_max_tokens,
            "selection_reasoning_effort": settings.ragflow_selection_reasoning_effort,
            "selection_thinking": "enabled",
            "top_k": 64,
            "page_size": 5,
            "threshold": 0.3,
            "external_snapshot": "not_available",
        },
        "reproducibility": "外部知识库和动态模型可能变化，不能保证完全复现",
    }


def run_case(case, settings, root, *, judge=False, repetition_index=None):
    """可选云追踪包围现有案例执行，保留本地记录和独立状态。"""
    from orderagent.observability import get_observability

    observability = get_observability(settings)
    run_id = f"{case.id}-{uuid4().hex}"
    experiment = {}
    if (root / "experiment.json").exists():
        experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    with observability.request(
        source=f"eval_{case.mode}",
        session_id=run_id,
        run_id=run_id,
        case_id=case.id,
        repetition_index=repetition_index,
        experiment_id=root.name,
        code_version=(experiment["commit"] + ("+dirty" if experiment.get("dirty") else ""))
        if experiment.get("commit")
        else None,
        dataset_hash=experiment.get("dataset_hash"),
        model_config_summary={
            "agent": settings.model_name if not case.retrieval_only else None,
            "selection_effort": settings.ragflow_selection_reasoning_effort
            if not case.retrieval_only
            else None,
        },
        knowledge_source=case.fixture
        if case.mode == "controlled_agent"
        else (
            "ragflow_retrieval_only"
            if case.retrieval_only
            else "configured_rag"
            if settings.ragflow_api_url
            else "local_json"
        ),
    ) as tracking:
        record = _run_case(
            case,
            settings,
            root,
            judge=judge,
            tracking=tracking,
            run_id=run_id,
            repetition_index=repetition_index,
            vector_weight=experiment.get("vector_similarity_weight", 0.5),
        )
        if case.mode == "live_rag":
            observability.diagnostics(record.get("events", []))
        if record.get("error_type"):
            observability.diagnostics(
                [
                    {
                        "data": {
                            "stage": "evaluation_error",
                            "error_type": record["error_type"],
                            "error_code": record.get("error_code"),
                        }
                    }
                ]
            )
        record["langfuse"] = observability.complete_case(record, case)
    from orderagent.evaluation.report import archive_case

    return archive_case(record, root / run_id)


def _run_case(
    case,
    settings,
    root,
    *,
    judge=False,
    tracking=None,
    run_id=None,
    repetition_index=None,
    vector_weight=0.5,
):
    """每案例新图、新会话、新SQLite；同案例保留多轮状态，失败分支不继续确认。"""
    started = time.perf_counter()
    run_id = run_id or f"{case.id}-{uuid4().hex}"
    work = root / run_id
    work.mkdir(parents=True, exist_ok=False)
    isolated = settings.model_copy(
        update={
            "after_sales_db_path": work / "state.sqlite3",
            "ragflow_vector_weight": 0.5 if vector_weight is None else vector_weight,
            "mock_handoff_scenario": (
                "error" if case.fixture == "handoff_error_and_no_rule" else "success"
            ),
        }
    )
    collector = Collector()
    record = {
        "case_id": case.id,
        "user_id": case.user_id,
        "run_id": run_id,
        "experiment_id": root.name,
        "planned_turns": len(case.turns),
        "repetition_index": repetition_index,
        "mode": case.mode,
        "retrieval_only": case.retrieval_only,
        "vector_similarity_weight": isolated.ragflow_vector_weight
        if case.mode != "controlled_agent" and isolated.ragflow_api_url
        else None,
        "category": case.category,
        "review_status": case.review_status,
        "turns": [],
        "status": "unjudgeable",
        "failure": None,
        "judge_status": "disabled",
        "workdir": str(work),
        "reference": case.reference,
        "knowledge_source": case.fixture
        if case.mode == "controlled_agent"
        else "ragflow_retrieval_only"
        if case.retrieval_only
        else "configured_rag_or_json",
        "initial_state": {
            "session": "new",
            "database": "isolated",
            "after_sales": "refunded" if case.fixture == "trusted_refunded_order" else "未售后",
        },
    }
    gateway = EvaluationReturns()
    config = {
        **(tracking or {}),
        "configurable": {"thread_id": run_id},
        "callbacks": [collector, *(tracking or {}).get("callbacks", [])],
        "recursion_limit": 40,
    }
    try:
        if case.mode == "live_rag":
            from langchain_core.runnables import RunnableLambda

            repo = RagflowKnowledge(
                isolated, None if case.retrieval_only else create_deepseek_model(isolated)
            )
            try:
                lookup = (
                    (
                        lambda query: repo.find(
                            query, retrieval_only=True, vector_weight=vector_weight
                        )
                    )
                    if case.retrieval_only
                    else repo.find
                )
                answer = RunnableLambda(lookup).invoke(
                    SupportQuery.model_validate(case.query), config=config
                )
            except ValueError as exc:
                if str(exc) not in {"RAG_NO_CANDIDATES", "RAG_NO_APPLICABLE_RULE"}:
                    raise
                answer = None
                record["rag_outcome_code"] = str(exc)
            record["rag_answer"] = answer.model_dump() if answer else None
            # 文档级Hit@5来自检索诊断，不用最终选择反推检索是否命中。
            documents = [
                d["document"] for e in collector.events for d in e["data"].get("candidates", [])
            ]
            record["retrieved_documents"] = documents
            record["filtered_documents"] = [
                d["document"]
                for event in collector.events
                for d in event["data"].get("candidates", [])
                if d["id"] in event["data"].get("filtered_candidate_ids", [])
            ]
            if case.relevant_documents and case.review_status == "approved":
                record["hit_at_5"] = bool(set(documents) & set(case.relevant_documents))
                # 历史诊断没有过滤后ID时不伪造过滤后命中。
                has_filtered = any("filtered_candidate_ids" in e["data"] for e in collector.events)
                record["filtered_hit_at_5"] = (
                    bool(set(record["filtered_documents"]) & set(case.relevant_documents))
                    if has_filtered
                    else None
                )
                record["status"] = "pass" if record["hit_at_5"] else "fail"
                record["failure"] = None if record["hit_at_5"] else "retrieval"
        else:
            if case.fixture == "trusted_refunded_order":
                AfterSalesRepository(isolated.after_sales_db_path).record(
                    case.user_id, "DEMO-HP-O-0004", "refunded", "eval-initial-state"
                )
            graph = create_order_agent(
                isolated,
                return_gateway=gateway,
                support_repository=(
                    FixtureKnowledge(case.fixture) if case.mode == "controlled_agent" else None
                ),
            )
            previous_ids = set()
            all_checks = []
            for index, turn in enumerate(case.turns, 1):
                turn_start = time.perf_counter()
                config["metadata"] = {
                    **config.get("metadata", {}),
                    "turn": index,
                    "request_id": uuid4().hex,
                }
                from orderagent.observability import get_observability

                event_start = len(collector.events)
                run_start = len(collector.traced_runs)
                observer = get_observability(settings)
                with observer.request(
                    source="evaluation_turn",
                    session_id=run_id,
                    run_id=run_id,
                    case_id=case.id,
                    turn=index,
                    experiment_id=root.name,
                    request_id=config["metadata"]["request_id"],
                    repetition_index=repetition_index,
                ) as turn_tracking:
                    try:
                        state = graph.invoke(
                            {"messages": [{"role": "user", "content": turn.user}]},
                            config={
                                **config,
                                **turn_tracking,
                                "callbacks": [collector, *turn_tracking.get("callbacks", [])],
                            },
                            context=ToolContext(user_id=case.user_id),
                        )
                        observer.diagnostics(state=state)
                    finally:
                        for event in collector.events[event_start:]:
                            event["turn"] = index
                            event["request_id"] = config["metadata"]["request_id"]
                        observer.diagnostics(
                            collector.events[event_start:], runs=collector.traced_runs[run_start:]
                        )
                messages = [m for m in state["messages"] if m.id not in previous_ids]
                previous_ids = {m.id for m in state["messages"]}
                store = AfterSalesRepository(isolated.after_sales_db_path)
                order_id = state.get("return_order_id")
                op = store.get_return_operation(case.user_id, order_id) if order_id else None
                gateway.last_operation = op.model_dump(mode="json") if op else None
                checks = check_turn(turn.checks, state, messages, gateway)
                all_checks.extend(checks)
                record["turns"].append(
                    {
                        "turn": index,
                        "user": turn.user,
                        "answer": state["messages"][-1].text,
                        "messages": [m.model_dump(mode="json") for m in messages],
                        "state": {
                            k: v
                            for k, v in state.items()
                            if k.startswith(("support_", "return_"))
                            or k
                            in {
                                "active_support_topic",
                                "consultation_id",
                                "question_processed_messages",
                            }
                        },
                        "operation": gateway.last_operation,
                        "creates": len(gateway.requests),
                        "checks": checks,
                        "expect": turn.expect,
                        "forbid": turn.forbid,
                        "seconds": time.perf_counter() - turn_start,
                    }
                )
                passed = all(c["passed"] for c in checks)
                if (not passed and turn.on_fail == "stop") or (passed and turn.on_pass == "stop"):
                    break
            record["skipped_turns"] = len(case.turns) - len(record["turns"])
            record["hard_check_count"] = len(all_checks)
            if all_checks:
                record["status"] = "pass" if all(c["passed"] for c in all_checks) else "fail"
                record["failure"] = None if record["status"] == "pass" else "business_assertion"
            record["scope"] = "仅显式硬规则；自然语言expect/forbid仍需人工核对"
            failed_checks = [c for c in all_checks if not c["passed"]]
            if failed_checks:
                record["failure"] = (
                    "tool_routing_or_arguments"
                    if any(c["kind"] in {"tool", "tool_arg"} for c in failed_checks)
                    else "business_state_or_boundary"
                )
        record["trace"] = flatten_runs(collector.traced_runs)
        record["events"] = collector.events
    except Exception as exc:
        original = exc.__cause__ or exc
        record.update(
            status="error", failure="target_exception", error_type=type(original).__name__
        )
        record["error_diagnostic"] = {
            "stage": "rag_execution" if case.mode == "live_rag" else "agent_execution",
            "diagnostic_code": "EVAL_TARGET_EXCEPTION",
            "error_type": type(exc).__name__,
            "cause_type": type(original).__name__,
            "detail": re.sub(r"https?://[^\s]+", "[ENDPOINT]", str(original))[:1000],
            **http_error_context(original),
        }
        record["_failed_runs"] = (
            flatten_runs(collector.traced_runs[run_start:]) if "run_start" in locals() else []
        )
        if isinstance(exc, ValueError) and str(exc).startswith("RAG_"):
            record["error_code"] = str(exc)
        if case.mode != "live_rag" and "turn" in locals():
            record["failed_turn"] = {
                "turn": index,
                "user": turn.user,
                "seconds": time.perf_counter() - turn_start,
            }
            record["skipped_turns"] = len(case.turns) - index
        if "LengthFinishReason" in type(original).__name__:
            record["failure"] = "model_output_truncation"
        elif case.mode == "live_rag":
            record["failure"] = (
                "rule_selection_exception"
                if any(e["data"].get("stage") == "selection_started" for e in collector.events)
                else "rag_connection_or_retrieval"
            )
        record["trace"] = flatten_runs(collector.traced_runs)
        record["events"] = collector.events
    record["target_seconds"] = time.perf_counter() - started
    record["diagnostics"] = [e["data"] for e in collector.events]
    record["evidence_failures"] = [
        d for d in record["diagnostics"] if d.get("stage") == "selection_rejected"
    ]
    record["selection_records"] = [d for d in record["diagnostics"] if "debug_status" in d]
    misses = {"RAG_NO_CANDIDATES", "RAG_NO_APPLICABLE_RULE"}
    record["dependency_errors"] = [
        d
        for d in record["diagnostics"]
        if d.get("error_type") and d.get("error_code") not in misses
    ]
    record["knowledge_misses"] = [d for d in record["diagnostics"] if d.get("error_code") in misses]
    record["model_calls"] = sum(r["type"] == "llm" for r in record["trace"])
    record["tool_calls"] = sum(r["type"] == "tool" for r in record["trace"])
    secrets = [
        settings.api_key.get_secret_value(),
        settings.ragflow_api_key.get_secret_value(),
        settings.qiwen_api_key.get_secret_value(),
        settings.langfuse_secret_key.get_secret_value(),
        settings.langfuse_public_key.get_secret_value(),
    ]
    record = sanitize(record, secrets)
    record["judge_input"] = build_judge_input(record, case.reference)
    if judge and record["turns"]:
        # 原始执行记录仅供本轮内存处理；评分后按v3保存必要证据，不重复归档整棵树。
        judge_input = record["judge_input"]
        record["judge_diagnostics"] = {
            "stream": True,
            "input_version": "compact-v1",
            "input_characters": len(json.dumps(judge_input, ensure_ascii=False)),
            "timeout_seconds": 30,
            "max_retries": 1,
            "error_type": None,
            "error_detail": None,
            "cause_type": None,
            "cause_detail": None,
            "status_code": None,
            "model": None,
            "response_id": None,
            "request_id": None,
            "finish_reason": None,
            "usage": None,
            "attempt_count": None,
        }
        judge_start = time.perf_counter()
        try:
            record["quality"] = grade_quality(
                settings,
                judge_input,
                diagnostics=record["judge_diagnostics"],
            )
            record["judge_status"] = "completed"
        except Exception as exc:
            record["judge_status"] = "error"
            record["judge_error_type"] = type(exc).__name__
            cause = exc.__cause__ or exc.__context__
            # 不写请求头、异常body或模型正文；仅保留脱敏、限长的错误摘要。
            detail = (
                json.dumps(
                    [
                        {"type": error["type"], "loc": error["loc"]}
                        for error in exc.errors(include_input=False, include_context=False)
                    ]
                )
                if isinstance(exc, ValidationError)
                else re.sub(r"https?://[^\s]+", "[ENDPOINT]", str(exc))
            )
            cause_detail = (
                re.sub(r"https?://[^\s]+", "[ENDPOINT]", str(cause)) if cause is not None else None
            )
            for secret in secrets:
                if secret:
                    detail = detail.replace(secret, "[REDACTED]")
                    if cause_detail is not None:
                        cause_detail = cause_detail.replace(secret, "[REDACTED]")
            record["judge_diagnostics"].update(
                stage="judge",
                diagnostic_code="EVAL_JUDGE_EXCEPTION",
                error_type=type(exc).__name__,
                error_detail=detail[:1000],
                cause_type=type(cause).__name__ if cause is not None else None,
                cause_detail=cause_detail[:1000] if cause_detail is not None else None,
                status_code=getattr(exc, "status_code", None),
                endpoint_template=http_error_context(exc)["endpoint_template"],
                request_id=getattr(exc, "request_id", None)
                or record["judge_diagnostics"]["request_id"],
            )
        record["judge_seconds"] = time.perf_counter() - judge_start
        record["judge_diagnostics"]["total_seconds"] = record["judge_seconds"]
    record = sanitize(record, secrets)
    return record


def execute(case, settings, root, judge=False, upload=False, client=None, repetition_index=None):
    """LangSmith仅由upload控制；Langfuse使用Settings中的独立显式开关。"""
    # langchain-core 1.6.12将字符串local视为真值并创建远程Tracer；
    # 本地显式False，使用独立RunCollector保留执行树，避免无意联网。
    with tracing_context(enabled=upload, client=client):
        return run_case(case, settings, root, judge=judge, repetition_index=repetition_index)
