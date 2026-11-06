"""可选 Langfuse 官方回调接入；上传白名单摘要，不参与业务决定。"""

import atexit
import json
import os
import re
from contextlib import contextmanager
from functools import lru_cache
from threading import Lock, Thread
from uuid import uuid4

from orderagent.settings import Settings

# 仅项目生成的关联字段；自动回调附带的配置和身份不属于白名单。
META = {
    "request_id",
    "session_id",
    "run_id",
    "case_id",
    "source",
    "knowledge_source",
    "turn",
    "error_type",
    "cause_type",
    "stage",
    "error_code",
    "failure_reasons",
    "handling",
    "applicable",
    "decision",
    "excerpt_valid",
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
}
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
                mask=lambda *, data, **_: self.clean(data) if isinstance(data, dict) else None,
                mask_otel_spans=mask_spans,
                should_export_span=exportable,
                tracer_provider=TracerProvider(
                    resource=Resource({"service.name": "orderagent"}), shutdown_on_exit=False
                ),
            )
            atexit.register(self.close)
        except Exception:
            print("Langfuse 初始化失败，继续使用本地记录。")

    def clean(self, data):
        """仅接受受控摘要；错误文本及任意正文不经过此接口。"""
        result = {k: v for k, v in data.items() if k in META}
        text = json.dumps(result, ensure_ascii=False, default=str)
        for secret in self.secrets:
            if secret:
                text = text.replace(json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]")
        return json.loads(text)

    @contextmanager
    def request(self, *, source, session_id, request_id=None, **metadata):
        """建立父 observation，业务异常原样抛出，关闭追踪失败不覆盖业务异常。"""
        config = {}
        scope = observation = None
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
