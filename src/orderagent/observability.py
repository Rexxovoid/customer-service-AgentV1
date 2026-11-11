"""可选 Langfuse 官方回调接入；上传白名单摘要，不参与业务决定。"""

import atexit
import hashlib
import json
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from threading import Lock, Thread
from uuid import uuid4

from orderagent.settings import Settings

# 仅项目生成的关联字段；自动回调附带的配置和身份不属于白名单。
META = {
    "request_id",
    "session_id",
    "eval_run_id",
    "case_id",
    "source",
    "knowledge_source",
    "turn",
    "error_type",
    "cause_type",
    "status_code",
    "endpoint_template",
    "stage",
    "error_code",
    "failure_reasons",
    "handling",
    "applicable",
    "decision",
    "excerpt_valid",
    "paragraph_start",
    "paragraph_end",
    "support_phase",
    "return_stage",
    "return_error_code",
    "handoff_code",
    "handoff_reason",
    "seconds",
    "tool_action",
    "tool_ok",
    "repetition_index",
    "support_error_code",
    "experiment_id",
    "code_version",
    "dataset_hash",
    "model_config_summary",
    "judge_status",
    "judge_error_type",
    "hard_status",
    "hard_checked",
    "hard_failed",
    "payload_digest",
    "content_complete",
    "turn_count",
    "case_trace_ids",
    "original_stage_seconds",
    "completed",
    "raw_candidate_count",
    "nonempty_candidate_count",
    "filtered_candidate_count",
    "candidate_limit",
    "top_k",
    "similarity_threshold",
    "summary_generated_after_execution",
}
_content_scope = ContextVar("approved_evaluation_content", default=False)
ATTRS = {
    "langfuse.observation.type",
    "langfuse.observation.level",
    "langfuse.observation.model.name",
    "langfuse.observation.usage_details",
    "langfuse.trace.name",
    "session.id",
}
_client_lock = Lock()


def mask_spans(*, params):
    """官方导出钩子：删除正文、模型参数、身份及所有非白名单属性。"""
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for identifier, span in params.spans.items():
        keep = set(ATTRS)
        keep.update(f"langfuse.observation.metadata.{k}" for k in META)
        keep.update(f"langfuse.trace.metadata.{k}" for k in META)
        if span.name == "approved_evaluation_content":
            keep.add("langfuse.observation.metadata.review_content")
        if (
            span.name == "approved_evaluation_turn"
            and span.attributes.get("langfuse.observation.metadata.content_export_authorized")
            == "true"
        ):
            keep.update({"langfuse.observation.input", "langfuse.observation.output"})
        patches[identifier] = OtelSpanPatch(
            delete_attributes=tuple(k for k in span.attributes if k not in keep)
        )
    return MaskOtelSpansResult(span_patches=patches)


def exportable(span):
    """属性钩子无法清理事件/状态正文，含这些内容的 span 不上传。"""
    return not span.events and not span.links and not span.status.description


class Observability:
    """共享官方客户端；每轮独立回调，异常仅降级追踪，不重试业务。"""

    def __init__(self, settings):
        self.client = None
        self.initialization_status = "disabled"
        self.initialization_error_type = None
        self.settings = settings
        self.secrets = [
            settings.api_key.get_secret_value(),
            settings.ragflow_api_key.get_secret_value(),
            settings.qiwen_api_key.get_secret_value(),
            settings.langfuse_secret_key.get_secret_value(),
            settings.langfuse_public_key.get_secret_value(),
        ]
        if not settings.langfuse_enabled:
            return
        if not all(self.secrets[-2:]):
            self.initialization_status = "missing_configuration"
            print("Langfuse 未配置完整，继续使用本地记录。")
            return
        try:
            from langfuse import Langfuse
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider

            # SDK 在 mask 前处理媒体，必须同时关闭官方媒体上传旁路。
            os.environ["LANGFUSE_MEDIA_UPLOAD_ENABLED"] = "false"
            self.client = Langfuse(
                public_key=self.secrets[-1],
                secret_key=self.secrets[-2],
                base_url=settings.langfuse_base_url,
                timeout=2,
                tracing_enabled=True,
                mask=self.mask_data,
                mask_otel_spans=mask_spans,
                should_export_span=exportable,
                tracer_provider=TracerProvider(
                    resource=Resource({"service.name": "orderagent"}), shutdown_on_exit=False
                ),
            )
            atexit.register(self.close)
            self.initialization_status = "ready"
        except Exception as exc:
            self.initialization_status = "initialization_error"
            self.initialization_error_type = type(exc).__name__
            print("Langfuse 初始化失败，继续使用本地记录。")

    def clean(self, data):
        """仅接受受控摘要；错误文本及任意正文不经过此接口。"""
        result = {k: v for k, v in data.items() if k in META}
        text = json.dumps(result, ensure_ascii=False, default=str)
        for secret in self.secrets:
            if secret:
                text = text.replace(json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]")
        return json.loads(text)

    def mask_data(self, *, data, **kwargs):
        """自动回调始终删正文；只有本方法受控作用域内允许审查过的内容包。"""
        if not isinstance(data, dict):
            return None
        safe = self.clean(data)
        if _content_scope.get() and "_approved_io" in data:
            return data["_approved_io"]
        if _content_scope.get() and data.get("_approved_io_authorized") is True:
            safe["content_export_authorized"] = True
        if _content_scope.get() and "review_content" in data:
            safe["review_content"] = data["review_content"]
        return safe

    def redact_content(self, text):
        """审核模拟数据的二次过滤；不声称能自动识别任意自然语言个人信息。"""
        for secret in self.secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        if "@" in text:
            text = re.sub(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
        text = re.sub(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)", "[PHONE]", text)
        text = re.sub(r"(?<!\d)\d{15,19}[Xx]?(?!\d)", "[IDENTIFIER]", text)
        text = re.sub(
            r"(?im)^.*(?:地址|收件人|联系人|身份证|银行卡|authorization|cookie|api[_-]?key)[:：].*$",
            "[SENSITIVE_LINE]",
            text,
        )
        text = re.sub(r"(?:DEMO|LOAD)-[A-Z0-9-]+", "[BUSINESS_ID]", text)
        return text

    def complete_case(self, record, case):
        """提交已有断言及裁判分数；仅审核过的受控模拟案例可另存精简正文。"""
        result = {
            "trace_id": None,
            "observation_id": None,
            "trace_url": None,
            "content_digest": None,
            "content_complete": False,
            "delivery": "disabled",
            "plan": self.settings.langfuse_plan,
            "content_status": "disabled",
            "initialization_status": self.initialization_status,
            "initialization_error_type": self.initialization_error_type,
            "verification_scope": "specified_turn_content_only",
            "verification_status": "not_verified",
            "content_enabled": self.settings.langfuse_content_enabled,
            "content_policy": "approved_controlled_only",
        }
        if self.client is None:
            return result
        if self.settings.langfuse_content_enabled:
            result["content_status"] = (
                "source_not_allowed"
                if case.mode != "controlled_agent"
                else "case_not_reviewed"
                if case.review_status != "approved"
                else "content_not_approved"
                if not case.content_upload_approved
                else "eligible"
            )
        try:
            result.update(
                trace_id=self.client.get_current_trace_id(),
                observation_id=self.client.get_current_observation_id(),
                delivery="queued_not_verified",
            )
            if result["trace_id"] is None:
                result["delivery"] = "trace_unavailable"
                return result
            # 官方get_trace_url首次会查询项目ID；不在案例执行路径额外联网。
            # trace_id可直接用于UI筛选，URL待显式回读时再获取。
            checks = [c for t in record.get("turns", []) for c in t.get("checks", [])]
            metadata = self.clean(
                {
                    "hard_status": record["status"],
                    "hard_checked": len(checks),
                    "hard_failed": sum(not c["passed"] for c in checks),
                    "judge_status": record["judge_status"],
                    "judge_error_type": record.get("judge_error_type"),
                    "summary_generated_after_execution": True,
                }
            )
            with self.client.start_as_current_observation(
                name="evaluation_result", metadata=metadata
            ):
                pass
            if checks and record["status"] in {"pass", "fail"}:
                self.client.create_score(
                    name="automatic.explicit_hard_rules",
                    value=int(record["status"] == "pass"),
                    trace_id=result["trace_id"],
                    observation_id=result["observation_id"],
                    data_type="BOOLEAN",
                )
            for name, dimension in (record.get("quality") or {}).get("dimensions", {}).items():
                if record["judge_status"] == "completed" and dimension["score"] is not None:
                    self.client.create_score(
                        name=f"model.{name}",
                        value=dimension["score"],
                        trace_id=result["trace_id"],
                        observation_id=result["observation_id"],
                        data_type="NUMERIC",
                    )
            if (
                self.settings.langfuse_content_enabled
                and case.mode == "controlled_agent"
                and case.review_status == "approved"
                and case.content_upload_approved
            ):
                turns = []
                for turn in record.get("turns", []):
                    tools = []
                    for message in turn.get("messages", []):
                        if message.get("type") != "tool":
                            continue
                        artifact = message.get("artifact") or {}
                        data = artifact.get("data") or {}
                        tools.append(
                            {
                                "name": message.get("name"),
                                "ok": artifact.get("ok"),
                                "action": data.get("action"),
                                "code": (artifact.get("error") or {}).get("code")
                                or data.get("code"),
                                "evidence": (data.get("source") or {}).get("reference_text"),
                                "result": {
                                    k: data[k]
                                    for k in (
                                        "action",
                                        "message",
                                        "missing_information",
                                        "order_status",
                                        "shipping_status",
                                        "logistics_status",
                                        "status",
                                        "order_status_label",
                                        "after_sales_status",
                                        "after_sales_progress",
                                        "snapshot_at",
                                        "synthetic",
                                        "match_type",
                                        "needs_clarification",
                                        "total_matches",
                                        "returned_count",
                                        "notice",
                                        "parameter_scope",
                                    )
                                    if k in data
                                },
                            }
                        )
                        if message.get("name") == "query_products":
                            tools[-1]["result"]["products"] = [
                                {
                                    k: p[k]
                                    for k in (
                                        "id",
                                        "name",
                                        "brand",
                                        "model",
                                        "category",
                                        "description",
                                        "evidence_groups",
                                        "source_last_updated",
                                    )
                                    if k in p
                                }
                                for p in data.get("products", [])
                            ]
                        if message.get("name") == "query_order":
                            tools[-1]["result"]["items"] = [
                                {
                                    k: p[k]
                                    for k in (
                                        "product_id",
                                        "product_name",
                                        "name",
                                        "model",
                                        "quantity",
                                    )
                                    if k in p
                                }
                                for p in data.get("items", [])
                            ]
                    state = turn.get("state") or {}
                    names = {
                        m.get("name") for m in turn.get("messages", []) if m.get("type") == "tool"
                    }
                    knowledge = (
                        state.get("return_rule_source")
                        if "handle_return" in names
                        else state.get("support_advice")
                        if "query_quality_support" in names
                        else {}
                    ) or {}
                    evidence = {
                        k: knowledge[k]
                        for k in (
                            "id",
                            "answer",
                            "applicability",
                            "data_origin",
                            "missing_information",
                        )
                        if k in knowledge
                    }
                    source = knowledge.get("source") or {}
                    evidence["source"] = {
                        k: source[k]
                        for k in (
                            "reference_text",
                            "document",
                            "chunk_id",
                            "handling",
                        )
                        if k in source
                    }
                    turns.append(
                        {
                            "turn": turn["turn"],
                            "user": self.redact_content(turn["user"]),
                            "answer": self.redact_content(turn["answer"]),
                            "tools": tools,
                            "checks": turn.get("checks", []),
                            "knowledge_evidence": evidence,
                        }
                    )

                def redact(value):
                    if isinstance(value, str):
                        return self.redact_content(value)
                    if isinstance(value, list):
                        return [redact(v) for v in value]
                    if isinstance(value, dict):
                        return {k: redact(v) for k, v in value.items()}
                    return value

                payload = redact(turns)
                digest = hashlib.sha256(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest()
                # 不静默截断内容。超出限制则保留本地，不标为云端完整。
                if len(json.dumps(payload).encode()) <= 128 * 1024:
                    token = _content_scope.set(True)
                    try:
                        with self.client.start_as_current_observation(
                            name="approved_evaluation_content",
                            metadata={
                                "payload_digest": digest,
                                "content_complete": True,
                                "turn_count": len(turns),
                            },
                        ) as content_observation:
                            result["content_observation_id"] = content_observation.id
                            for turn in payload:
                                with self.client.start_as_current_observation(
                                    name="approved_evaluation_turn",
                                    input={"_approved_io": {"user": turn["user"]}},
                                    output={
                                        "_approved_io": {
                                            k: v
                                            for k, v in turn.items()
                                            if k not in {"user", "turn"}
                                        }
                                    },
                                    metadata={
                                        "turn": turn["turn"],
                                        "_approved_io_authorized": True,
                                        "summary_generated_after_execution": True,
                                    },
                                ):
                                    pass
                    finally:
                        _content_scope.reset(token)
                    result.update(content_digest=digest, content_complete=True)
                    result.update(content_format="turn_io_v1", content_turn_count=len(payload))
                    result["content_status"] = "queued"
                else:
                    result["content_status"] = "over_budget_local_preserved"
        except Exception as exc:
            result["delivery"] = "queue_error"
            result["upload_error_type"] = type(exc).__name__
        return result

    def verify_content(self, reference):
        """官方分页回读且重新计算摘要；flush/ID本身不是完整性证明。"""
        outcome = {"verified": False, "reason": "unavailable"}
        if self.client is None or not reference.get("content_digest"):
            return outcome
        try:
            if reference.get("content_format") == "turn_io_v1":
                turns = []
                for page in self.client.iter_observation_pages(
                    trace_id=reference["trace_id"],
                    name="approved_evaluation_turn",
                    fields="core,metadata,io",
                    expand_metadata="*",
                    limit=50,
                ):
                    for item in page.data:
                        if item.parent_observation_id != reference.get("content_observation_id"):
                            continue
                        metadata = item.metadata or {}
                        # 官方API现返回原始JSON字符串；使用标准解码，不修补无效正文。
                        inputs = (
                            json.loads(item.input) if isinstance(item.input, str) else item.input
                        )
                        outputs = (
                            json.loads(item.output) if isinstance(item.output, str) else item.output
                        )
                        if isinstance(inputs, dict) and isinstance(outputs, dict):
                            turns.append(
                                {
                                    "turn": metadata.get("turn"),
                                    "user": inputs.get("user"),
                                    **outputs,
                                }
                            )
                turns.sort(key=lambda t: t["turn"])
                expected = reference.get("content_turn_count")
                contiguous = [t["turn"] for t in turns] == list(range(1, (expected or 0) + 1))
                digest = hashlib.sha256(
                    json.dumps(turns, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest()
                verified = bool(expected) and contiguous and digest == reference["content_digest"]
                return {
                    "verified": bool(verified),
                    "reason": "turn_inputs_outputs_matched"
                    if verified
                    else "turn_content_missing_or_incomplete",
                }
            for page in self.client.iter_observation_pages(
                trace_id=reference["trace_id"],
                name="approved_evaluation_content",
                fields="core,metadata",
                expand_metadata="*",
                limit=50,
            ):
                for item in page.data:
                    data = item.metadata or {}
                    payload = data.get("review_content")
                    if payload is not None:
                        digest = hashlib.sha256(
                            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
                        ).hexdigest()
                        if (
                            digest == reference["content_digest"]
                            and data.get("payload_digest") == digest
                            and data.get("content_complete") is True
                        ):
                            return {
                                "verified": True,
                                "reason": "content_digest_matched",
                            }
            outcome["reason"] = "content_missing_or_incomplete"
        except Exception as exc:
            outcome["reason"] = type(exc).__name__
        return outcome

    @contextmanager
    def request(self, *, source, session_id, request_id=None, **metadata):
        """建立父 observation，业务异常原样抛出，关闭追踪失败不覆盖业务异常。"""
        config = {}
        scope = observation = None
        # metadata.run_id是LangGraph的恢复标识；案例ID仅作为自有追踪字段传入。
        if "run_id" in metadata:
            metadata["eval_run_id"] = metadata.pop("run_id")
        if self.client is not None:
            try:
                from langfuse.langchain import CallbackHandler

                safe = self.clean(
                    dict(
                        metadata,
                        source=source,
                        session_id=session_id,
                        request_id=request_id or uuid4().hex,
                    )
                )
                scope = self.client.start_as_current_observation(name=source, metadata=safe)
                observation = scope.__enter__()
                config = {
                    "callbacks": [CallbackHandler(public_key=self.secrets[-1])],
                    "metadata": {**safe, "langfuse_session_id": session_id},
                }
            except Exception:
                print("Langfuse 追踪不可用，本轮继续执行。")
        try:
            yield config
        except BaseException as exc:
            if observation is not None:
                try:
                    observation.update(
                        level="ERROR",
                        metadata=self.clean(
                            {
                                "error_type": type(exc).__name__,
                                "cause_type": type(exc.__cause__).__name__
                                if exc.__cause__
                                else None,
                            }
                        ),
                    )
                except Exception:
                    pass
            raise
        finally:
            if scope is not None:
                try:
                    # 不把异常传给 OTel，避免自动写入异常正文和堆栈。
                    scope.__exit__(None, None, None)
                except Exception:
                    pass

    def diagnostics(self, events=(), state=None, runs=()):
        """官方回调不接收 custom event；以安全子 observation 补充结果摘要。"""
        if self.client is None:
            return
        try:
            from orderagent.evaluation.runner import flatten_runs

            for run in flatten_runs(runs):
                if run.get("error"):
                    match = re.search(r"\b(\w+(?:Error|Exception))\b", run["error"])
                    with self.client.start_as_current_observation(
                        name="execution_error",
                        level="ERROR",
                        metadata={"error_type": match.group(1) if match else None},
                    ):
                        pass
                output = (run.get("outputs") or {}).get("output")
                artifact = getattr(output, "artifact", None)
                if artifact is None and isinstance(getattr(output, "update", None), dict):
                    artifact = next(
                        (
                            getattr(m, "artifact", None)
                            for m in output.update.get("messages", [])
                            if isinstance(getattr(m, "artifact", None), dict)
                        ),
                        None,
                    )
                if isinstance(artifact, dict):
                    data = artifact.get("data") or {}
                    with self.client.start_as_current_observation(
                        name="tool_result",
                        metadata=self.clean(
                            {
                                "tool_ok": artifact.get("ok"),
                                "tool_action": data.get("action"),
                                "error_code": (artifact.get("error") or {}).get("code")
                                or data.get("code"),
                            }
                        ),
                    ):
                        pass
            for event in events:
                data = event.get("data", {})
                safe = self.clean(data)
                safe.update(self.clean(data.get("selection", {})))
                if safe:
                    safe["summary_generated_after_execution"] = True
                    if "seconds" in safe:
                        safe["original_stage_seconds"] = safe["seconds"]
                    with self.client.start_as_current_observation(
                        name="rag_diagnostic", metadata=safe
                    ):
                        pass
            if state:
                safe = self.clean(state)
                handoff = state.get("support_handoff") or {}
                # 原始 reason 含订单和用户原文；仅上传固定分类，不上传该文本。
                safe["handoff_code"] = handoff.get("code")
                safe["handoff_reason"] = (
                    ("coded_handoff" if handoff.get("code") else "handoff_without_code")
                    if handoff
                    else None
                )
                reason = str(handoff.get("reason") or "").split("；", 1)[0]
                # 仅映射代码中已定义的固定原因，不上传任意模型/用户文本。
                known_reasons = {
                    "用户明确要求人工": "explicit_human_request",
                    "同一退货问题第6次有效求解仍未解决": "unresolved_threshold",
                    "同一质量问题第6次有效求解仍未解决，自动转人工": "unresolved_threshold",
                    "未查询到适用的故障知识": "no_applicable_quality_rule",
                    "未找到适用退货规则": "no_applicable_return_rule",
                    "规则冲突或知识不支持自动排查": "rule_conflict_or_unsupported",
                    "未命中适用退货方案或规则不支持自动处理": "no_automatic_return_rule",
                    "当前问题已转人工，复用交接结果": "reuse_handoff",
                }
                if reason in known_reasons:
                    safe["handoff_reason"] = known_reasons[reason]
                with self.client.start_as_current_observation(
                    name="business_result", metadata=safe
                ):
                    pass
        except Exception:
            pass

    def close(self):
        """SDK shutdown 含队列等待；后台调用官方清理，主线程最多等两秒。"""
        if self.client is None:
            return
        client, self.client = self.client, None

        def stop():
            try:
                client.shutdown()
            except Exception:
                pass

        worker = Thread(target=stop, daemon=True)
        worker.start()
        worker.join(timeout=2)


@lru_cache(maxsize=8)
def _cached_observability(settings: Settings):
    """同配置复用 SDK 客户端，不为每轮请求创建后台发送线程。"""
    return Observability(settings)


def get_observability(settings: Settings):
    """并发测评启动时也只初始化一次同配置客户端。"""
    with _client_lock:
        return _cached_observability(settings)
