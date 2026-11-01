import json
import time
from difflib import unified_diff
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

import requests
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.exceptions import OutputParserException
from langgraph.config import get_stream_writer
from openai import (
    APITimeoutError,
    AuthenticationError,
    LengthFinishReasonError,
    PermissionDeniedError,
)
from pydantic import BaseModel, Field, ValidationError
from ragflow_sdk import RAGFlow

from orderagent.prompts.order_prompt import load_prompt
from orderagent.schemas.support_schema import SupportKnowledge
from orderagent.settings import PROJECT_ROOT, ConfigurationError

# 稳定且不含供应商敏感正文的业务诊断码；不新增异常/执行框架。
RAG_ERROR_MESSAGES = {
    "RAG_CONNECTION_ERROR": "RAGFlow连接失败，无法读取知识",
    "RAG_RETRIEVAL_TIMEOUT": "RAGFlow请求超时，未能完成知识检索",
    "RAG_AUTH_ERROR": "RAGFlow接口鉴权或权限校验失败",
    "RAG_HTTP_ERROR": "RAGFlow接口返回HTTP错误",
    "RAG_BINDING_MISMATCH": "知识库或助手名称不唯一、未找到或绑定不匹配",
    "RAG_RETRIEVAL_ERROR": "RAGFlow检索或业务响应异常",
    "RAG_NO_CANDIDATES": "当前检索条件下没有候选知识，不代表知识库必然没有对应规则",
    "RAG_NO_APPLICABLE_RULE": "已有候选，但未选出适用于当前问题的规则",
    "RAG_SELECTION_TRUNCATED": "候选已检索，规则选择模型输出截断",
    "RAG_SELECTION_TIMEOUT": "候选已检索，规则选择模型超时",
    "RAG_SELECTION_AUTH_ERROR": "规则选择模型鉴权或权限校验失败",
    "RAG_SELECTION_ERROR": "候选已检索，规则选择模型或输出结构异常",
    "RAG_EVIDENCE_INVALID": "规则摘录、来源或澄清信息未通过校验",
}


class TimeoutRAGFlow(RAGFlow):
    """仅补齐0.25.4官方SDK未暴露的请求超时，不重写检索或会话协议。"""

    def __init__(self, api_key: str, base_url: str, timeout: float):
        super().__init__(api_key=api_key, base_url=base_url)
        self.timeout = timeout

    def get(self, path, params=None, json=None):
        """保留官方GET协议，增加连接/读取超时，不配置重试。"""
        response = requests.get(
            self.api_url + path,
            params=params,
            json=json,
            headers=self.authorization_header,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response

    def post(self, path, json=None, stream=False, files=None):
        """保留官方POST协议，增加连接/读取超时，不配置重试。"""
        response = requests.post(
            self.api_url + path,
            json=json,
            stream=stream,
            files=files,
            headers=self.authorization_header,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response


class RuleSelection(BaseModel):
    """模型只能选择适用原文；不允许生成脱离来源的维修步骤或资格证据。"""

    applicable: bool = Field(description="是否有直接适用且足够完整的安全处理知识")
    chunk_id: str = Field(description="所选分块ID；不适用时为空")
    excerpt: str = Field(description="逐字摘录该分块中完整且带条件的答案；不改写")
    handling: Literal["basic_troubleshooting", "return_eligible", "clarify", "human"] = Field(
        description="按用户描述匹配的处理方式；return_eligible只表示支持模拟入口，不是已核验资格"
    )
    missing_information: list[str] = Field(default_factory=list)
    decision: (
        Literal["applicable", "missing_information", "not_applicable", "conflict", "unreliable"]
        | None
    ) = None


class RagflowKnowledge:
    """通过官方检索获取证据，再用官方结构化输出选择原文，不创建第二个执行Agent。"""

    def __init__(self, settings, model):
        if urlsplit(settings.api_base_url).hostname != "api.deepseek.com":
            raise ConfigurationError(
                "规则选择低推理配置仅验证了DeepSeek官方服务；其他服务需先验证参数支持，不静默忽略。"
            )
        self.client = TimeoutRAGFlow(
            settings.ragflow_api_key.get_secret_value(),
            settings.ragflow_api_url,
            settings.ragflow_timeout_seconds,
        )
        self.dataset_name = settings.ragflow_dataset_name
        self.assistant_name = settings.ragflow_assistant_name
        self.selection_budget = settings.ragflow_selection_max_tokens
        self.selection_effort = settings.ragflow_selection_reasoning_effort
        self.evidence_debug = settings.ragflow_evidence_debug
        self.debug_secrets = [
            settings.ragflow_api_key.get_secret_value(),
            settings.api_key.get_secret_value(),
        ]
        # 官方Pydantic模型复制只改选择器预算，主Agent模型实例及其1200上限不变。
        selector_model = model.model_copy(
            update={
                "max_tokens": settings.ragflow_selection_max_tokens,
                "reasoning_effort": settings.ragflow_selection_reasoning_effort,
                "extra_body": {**(model.extra_body or {}), "thinking": {"type": "enabled"}},
            }
        )
        self.selector = selector_model.with_structured_output(
            RuleSelection, method="json_mode", include_raw=True
        )

    def find(self, query):
        """仅检索指定库并核对助手绑定；异常由业务工具转人工，禁止回退常识答案。"""
        started = time.perf_counter()
        try:
            return self._find(query)
        except Exception as exc:
            selecting = "rag_stage=selection" in getattr(exc, "__notes__", [])
            code = (
                str(exc) if isinstance(exc, ValueError) and str(exc) in RAG_ERROR_MESSAGES else ""
            )
            if not code:
                if isinstance(exc, LengthFinishReasonError):
                    code = "RAG_SELECTION_TRUNCATED"
                elif isinstance(exc, APITimeoutError):
                    code = "RAG_SELECTION_TIMEOUT"
                elif isinstance(exc, (AuthenticationError, PermissionDeniedError)):
                    code = "RAG_SELECTION_AUTH_ERROR"
                elif isinstance(exc, requests.Timeout):
                    code = "RAG_RETRIEVAL_TIMEOUT"
                elif isinstance(exc, requests.ConnectionError):
                    code = "RAG_CONNECTION_ERROR"
                elif isinstance(exc, requests.HTTPError):
                    code = (
                        "RAG_AUTH_ERROR"
                        if exc.response is not None and exc.response.status_code in {401, 403}
                        else "RAG_HTTP_ERROR"
                    )
                else:
                    code = "RAG_SELECTION_ERROR" if selecting else "RAG_RETRIEVAL_ERROR"
            self._diagnostic(
                {"stage": "rag_exception", "error_type": type(exc).__name__, "error_code": code}
            )
            raise ValueError(code) from exc
        finally:
            self._diagnostic({"stage": "rag_finished", "seconds": time.perf_counter() - started})

    def _find(self, query):
        """官方检索与结构化选择，失败以稳定诊断码交给业务处理。"""
        self._diagnostic({"stage": "retrieval_started"})
        try:
            datasets = self.client.list_datasets(name=self.dataset_name)
            datasets = [item for item in datasets if item.name == self.dataset_name]
            chats = self.client.list_chats(name=self.assistant_name)
            chats = [item for item in chats if item.name == self.assistant_name]
            if len(datasets) != 1 or len(chats) != 1 or datasets[0].id not in chats[0].dataset_ids:
                raise ValueError("RAG_BINDING_MISMATCH")
            chunks = self.client.retrieve(
                dataset_ids=[datasets[0].id],
                question=(
                    f"{dict(headphones='耳机', mice='鼠标', keyboards='键盘')[query.category]} "
                    f"{query.issue} {query.environment} {query.search_description}"
                ),
                page_size=5,
                top_k=64,
                similarity_threshold=0.3,
            )
        except Exception as exc:
            exc.add_note("rag_stage=retrieval")
            raise
        evidence = {
            str(item.id): {"content": item.content[:6000], "document": item.document_name}
            for item in chunks[:5]
            if item.content
        }
        # 仅在明确命名的项目文档上过滤；未知来源保留，防止丢失跨规则例外。
        if query.task == "return" or query.issue.startswith("退货规则与适用条件"):
            evidence = {
                key: value
                for key, value in evidence.items()
                if "_02_故障条件问答" not in value["document"]
            }
        self._diagnostic(
            {
                "stage": "retrieved",
                "candidates": [
                    {
                        "id": str(item.id),
                        "document": item.document_name,
                        "similarity": getattr(item, "similarity", None),
                        "content": item.content[:6000],
                    }
                    for item in chunks[:5]
                    if item.content
                ],
            }
        )
        if not evidence:
            self._diagnostic({"stage": "no_candidates"})
            raise ValueError("RAG_NO_CANDIDATES")
        messages = [
            ("system", load_prompt("rag_selection_prompt")),
            (
                "human",
                json.dumps(
                    {
                        "query": query.model_dump(exclude={"search_description"}),
                        "evidence": evidence,
                    },
                    ensure_ascii=False,
                ),
            ),
        ]
        capture = {
            "record_id": f"evidence-{uuid4().hex}.json",
            "created_at_unix": time.time(),
            "phase": "before_call",
            "messages": messages,
            "candidates": evidence,
            "retrieval_search_description": query.search_description,
            "effective_max_tokens": getattr(self, "selection_budget", None),
            "reasoning_effort": getattr(self, "selection_effort", None),
            "thinking": "enabled",
            "structured_result": None,
            "raw_content": None,
            "raw_content_truncated": False,
            "parsing_error": None,
            "exception_type": None,
            "partial_response": None,
            "finish_reasons": None,
            "token_usage": None,
            "response_availability": "not_returned",
        }
        self._diagnostic({"stage": "selection_started", "_local_capture": capture})
        selection_started = time.perf_counter()
        try:
            selection = self.selector.invoke(messages)
            if isinstance(selection, dict) and "raw" in selection and "parsed" in selection:
                raw = selection["raw"]
                capture.update(
                    response_availability="raw_and_structured"
                    if selection["parsed"] is not None
                    else "raw_returned_parse_failed",
                    token_usage=raw.usage_metadata,
                    finish_reasons=[raw.response_metadata.get("finish_reason")],
                    response_model=raw.response_metadata.get("model_name"),
                    raw_content=raw.content[:16000]
                    if isinstance(raw.content, str)
                    else [
                        {"type": "text", "text": str(block.get("text", ""))[:16000]}
                        for block in raw.content[:16]
                        if isinstance(block, dict) and block.get("type") == "text"
                    ],
                    raw_content_truncated=isinstance(raw.content, str) and len(raw.content) > 16000,
                    structured_result=selection["parsed"].model_dump()
                    if isinstance(selection["parsed"], BaseModel)
                    else selection["parsed"],
                    phase="model_returned",
                )
                # 先落盘已返回的可见正文；不复制additional_kwargs中的隐藏推理。
                self._diagnostic(
                    {"stage": "selection_response_received", "_local_capture": capture}
                )
                if selection.get("parsing_error") is not None:
                    raise selection["parsing_error"]
                selection = selection["parsed"]
            selection = RuleSelection.model_validate(selection)
        except Exception as exc:
            if isinstance(exc, (OutputParserException, ValidationError)):
                causes = []
                cause = exc
                categories = []
                validation_errors = None
                for _ in range(6):
                    if cause is None:
                        break
                    causes.append(type(cause).__name__)
                    if isinstance(cause, json.JSONDecodeError):
                        categories.append("json_format")
                    if isinstance(cause, ValidationError):
                        validation_errors = cause.errors(
                            include_url=False, include_context=False, include_input=False
                        )
                        for error in validation_errors:
                            kind = error["type"]
                            categories.append(
                                "enum_value"
                                if kind in {"literal_error", "enum"}
                                else "missing_field"
                                if kind == "missing"
                                else "field_type"
                                if kind.endswith("_type")
                                else "field_validation"
                            )
                    cause = cause.__cause__
                parser_input = getattr(exc, "llm_output", None)
                capture["parsing_error"] = {
                    "type": type(exc).__name__,
                    "detail": str(exc)[:16000],
                    "cause_types": causes,
                    "categories": list(dict.fromkeys(categories)) or ["unclassified_parser_error"],
                    "validation_errors": validation_errors,
                    "parser_input": parser_input[:16000] if isinstance(parser_input, str) else None,
                }
            capture.update(
                phase="exception",
                exception_type=type(exc).__name__,
                error_code="RAG_SELECTION_TRUNCATED"
                if isinstance(exc, LengthFinishReasonError)
                else "RAG_SELECTION_TIMEOUT"
                if isinstance(exc, APITimeoutError)
                else "RAG_SELECTION_AUTH_ERROR"
                if isinstance(exc, (AuthenticationError, PermissionDeniedError))
                else "RAG_SELECTION_ERROR",
            )
            capture["selection_seconds"] = time.perf_counter() - selection_started
            # 官方异常保留completion；仅提取可见响应，不记录reasoning字段。
            completion = getattr(exc, "completion", None)
            if completion is not None:
                capture["response_availability"] = "available_from_exception"
                capture["partial_response"] = [
                    {
                        "index": choice.index,
                        "content": choice.message.content[:16000]
                        if choice.message.content
                        else None,
                    }
                    for choice in completion.choices
                ]
                capture["finish_reasons"] = [choice.finish_reason for choice in completion.choices]
                capture["token_usage"] = completion.usage.model_dump() if completion.usage else None
                capture["partial_response_truncated"] = any(
                    len(choice.message.content or "") > 16000 for choice in completion.choices
                )
            else:
                if capture["response_availability"] not in {
                    "raw_and_structured",
                    "raw_returned_parse_failed",
                }:
                    capture["response_availability"] = "unavailable_from_exception"
            self._diagnostic({"stage": "selection_exception", "_local_capture": capture})
            self._diagnostic(
                {
                    "stage": "selection_metrics",
                    "seconds": capture["selection_seconds"],
                    "usage": capture["token_usage"],
                    "reasoning_effort": getattr(self, "selection_effort", None),
                    "error_code": capture["error_code"],
                }
            )
            exc.add_note("rag_stage=selection")
            raise
        capture.update(
            phase="returned",
            structured_result=selection.model_dump(),
            response_availability="raw_and_structured"
            if capture["response_availability"] == "raw_and_structured"
            else "structured_only_raw_unavailable",
        )
        capture["selection_seconds"] = time.perf_counter() - selection_started
        self._diagnostic({"stage": "selection_returned", "_local_capture": capture})
        self._diagnostic(
            {
                "stage": "selection_metrics",
                "seconds": capture["selection_seconds"],
                "usage": capture["token_usage"],
                "reasoning_effort": getattr(self, "selection_effort", None),
            }
        )
        chosen = evidence.get(selection.chunk_id)
        self._diagnostic({"stage": "selected", "selection": selection.model_dump()})
        if (
            not selection.applicable
            or selection.handling == "human"
            or selection.decision in {"not_applicable", "conflict", "unreliable"}
        ):
            self._diagnostic({"stage": "no_applicable_rule"})
            raise ValueError("RAG_NO_APPLICABLE_RULE")
        failure_reasons = []
        normalized_excerpt = selection.excerpt.replace("\r\n", "\n").replace("\r", "\n")
        normalized_source = (
            chosen["content"].replace("\r\n", "\n").replace("\r", "\n") if chosen else ""
        )
        raw_match = bool(
            chosen and selection.excerpt.strip() and selection.excerpt in chosen["content"]
        )
        normalized_match = bool(
            chosen and selection.excerpt.strip() and normalized_excerpt in normalized_source
        )
        capture.update(
            raw_excerpt_match=raw_match,
            normalized_excerpt_match=normalized_match,
            evidence_normalization="CRLF/CR_to_LF_only",
        )
        self._diagnostic(
            {
                "stage": "evidence_checked",
                "raw_excerpt_match": raw_match,
                "normalized_excerpt_match": normalized_match,
                "normalization": "CRLF/CR_to_LF_only",
                "_local_capture": capture,
            }
        )
        if chosen is None:
            failure_reasons.append("UNKNOWN_CHUNK")
        if not selection.excerpt.strip():
            failure_reasons.append("EMPTY_EXCERPT")
        elif chosen is not None and not normalized_match:
            failure_reasons.append("EXCERPT_NOT_IN_SOURCE")
        if len(selection.excerpt) > 2500:
            failure_reasons.append("EXCERPT_TOO_LONG")
        if selection.handling == "clarify" and not selection.missing_information:
            failure_reasons.append("MISSING_CLARIFICATION_FIELDS")
        if selection.decision == "missing_information" and selection.handling != "clarify":
            failure_reasons.append("INCONSISTENT_DECISION")
        if selection.missing_information and selection.handling == "return_eligible":
            failure_reasons.append("INCONSISTENT_DECISION")
        if failure_reasons:
            capture.update(
                phase="evidence_rejected",
                error_code="RAG_EVIDENCE_INVALID",
                failure_reasons=failure_reasons,
                candidate_text=str(chosen["content"]) if chosen else "",
                model_excerpt=selection.excerpt[:16000],
                text_truncated=len(selection.excerpt) > 16000,
                diff="\n".join(
                    unified_diff(
                        (str(chosen["content"]) if chosen else "").splitlines(),
                        selection.excerpt[:16000].splitlines(),
                        fromfile="candidate",
                        tofile="model_excerpt",
                        lineterm="",
                    )
                )[:64000],
            )
            self._diagnostic(
                {
                    "stage": "selection_rejected",
                    "error_code": "RAG_EVIDENCE_INVALID",
                    "failure_reasons": failure_reasons,
                    "_local_capture": capture,
                    "chunk_id": selection.chunk_id[:200],
                    "document": str(chosen["document"])[:200] if chosen else None,
                    "excerpt_length": len(selection.excerpt),
                    "excerpt_limit": 2500,
                    "applicable": selection.applicable,
                    "known_chunk": chosen is not None,
                    "handling": selection.handling,
                    "excerpt_valid": bool(
                        chosen
                        and selection.excerpt.strip()
                        and normalized_match
                        and len(selection.excerpt) <= 2500
                    ),
                }
            )
            raise ValueError("RAG_EVIDENCE_INVALID")
        return SupportKnowledge(
            id=selection.chunk_id,
            category=query.category,
            question=query.issue,
            answer=selection.excerpt,
            applicability="仅适用所引原文明确列出的条件；未知条件需核实",
            source={
                "reference_text": f"RAGFlow / {chosen['document']} / {selection.chunk_id}",
                "handling": selection.handling,
            },
            data_origin="ragflow_retrieval",
            missing_information=selection.missing_information,
        )

    def _diagnostic(self, data):
        """发出官方自定义事件；独立非Runnable调用无回调上下文时保持原行为。"""
        capture = data.pop("_local_capture", None)
        if capture is not None:
            data.update(debug_status="disabled", debug_record=None)
            if getattr(self, "evidence_debug", False):
                debug_dir = PROJECT_ROOT / "evals" / "debug" / "rag_evidence"
                try:
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    target = debug_dir / capture["record_id"]
                    if not target.exists() and len(list(debug_dir.glob("evidence-*.json"))) >= 20:
                        data["debug_status"] = "limit_reached"
                    else:
                        body = json.dumps(capture, ensure_ascii=False, indent=2)
                        for secret in getattr(self, "debug_secrets", []):
                            if secret:
                                body = body.replace(
                                    json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]"
                                )
                        with target.open("w", encoding="utf-8") as file:
                            file.write(body)
                        data.update(debug_status="saved", debug_record=capture["record_id"])
                except (OSError, ValueError):
                    data["debug_status"] = "write_failed"
        try:
            dispatch_custom_event("rag_diagnostic", data)
        except RuntimeError:
            pass
        # 官方custom流用于终端调试；只发送拒绝摘要，不包含摘录、用户输入或推理。
        if data.get("stage") in {
            "selection_started",
            "selection_returned",
            "selection_exception",
            "selection_rejected",
        }:
            try:
                get_stream_writer()({"rag_diagnostic": data})
            except (RuntimeError, KeyError):
                pass
