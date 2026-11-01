import json
import time

from langchain_core.messages import AIMessage, ToolMessage
from openai import OpenAI

from orderagent.evaluation.schemas import QualityScore
from orderagent.prompts.order_prompt import load_prompt


def check_turn(checks, state, messages, gateway):
    """只检查显式机器断言，不将自然语言规格自动视为已覆盖。"""
    calls = [c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    results = []
    for check in checks:
        actual = None
        if check.kind == "state":
            actual = state.get(check.key)
        elif check.kind == "creates":
            actual = len(gateway.requests)
        elif check.kind == "operation":
            actual = gateway.last_operation.get(check.key) if gateway.last_operation else None
        elif check.kind == "tool":
            actual = any(c["name"] == check.key for c in calls)
        elif check.kind == "tool_arg":
            name, field = check.key.split(".", 1)
            actual = next(
                (c["args"].get(field) for c in reversed(calls) if c["name"] == name), None
            )
        elif check.kind == "tool_error":
            actual = any(
                isinstance(m, ToolMessage) and check.key in str(m.content) for m in messages
            )
        elif check.kind == "answer_contains":
            actual = check.key in state["messages"][-1].text
        results.append({**check.model_dump(), "actual": actual, "passed": actual == check.value})
    return results


def grade_quality(settings, record, diagnostics=None):
    """仅调用官方兼容SDK；结构校验失败不重写答案、不伪造分数。"""
    if diagnostics is None:
        diagnostics = {}
    diagnostics.update(
        model=None,
        response_id=None,
        request_id=None,
        finish_reason=None,
        usage=None,
        stream=True,
        first_event_seconds=None,
        first_content_seconds=None,
        received_events=None,
        content_characters=None,
        max_event_gap_seconds=None,
    )
    if not settings.qiwen_api_key.get_secret_value():
        raise ValueError("缺少QIWEN_API_KEY")
    with OpenAI(
        api_key=settings.qiwen_api_key.get_secret_value(),
        base_url=settings.qiwen_base_url,
        timeout=30,
        max_retries=1,
    ) as client:
        started = time.perf_counter()
        previous_event = None
        with client.chat.completions.stream(
            model=settings.qianwen,
            messages=[
                {"role": "system", "content": load_prompt("judge_prompt", evaluation=True)},
                {"role": "user", "content": json.dumps(record, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            max_tokens=2000,
            temperature=0,
            stream_options={"include_usage": True},
        ) as stream:
            try:
                for event in stream:
                    if event.type != "chunk":
                        continue
                    elapsed = time.perf_counter() - started
                    if diagnostics["first_event_seconds"] is None:
                        diagnostics["first_event_seconds"] = elapsed
                        diagnostics["received_events"] = 0
                        diagnostics["content_characters"] = 0
                    if previous_event is not None:
                        gap = elapsed - previous_event
                        diagnostics["max_event_gap_seconds"] = max(
                            diagnostics["max_event_gap_seconds"] or 0, gap
                        )
                    previous_event = elapsed
                    diagnostics["received_events"] += 1
                    chunk = event.chunk
                    diagnostics["model"] = chunk.model or diagnostics["model"]
                    diagnostics["response_id"] = chunk.id or diagnostics["response_id"]
                    if chunk.usage is not None:
                        diagnostics["usage"] = chunk.usage.model_dump()
                    for choice in chunk.choices:
                        if choice.index != 0:
                            continue
                        if choice.finish_reason is not None:
                            diagnostics["finish_reason"] = choice.finish_reason
                        if choice.delta.content:
                            if diagnostics["first_content_seconds"] is None:
                                diagnostics["first_content_seconds"] = elapsed
                            diagnostics["content_characters"] += len(choice.delta.content)
            finally:
                if previous_event is not None:
                    snapshot = stream.current_completion_snapshot
                    diagnostics["model"] = snapshot.model
                    diagnostics["response_id"] = snapshot.id
                    if snapshot.usage is not None:
                        diagnostics["usage"] = snapshot.usage.model_dump()
                    if snapshot.choices:
                        diagnostics["finish_reason"] = snapshot.choices[0].finish_reason
            # 由官方SDK累积正文，不自行拼接、解析流式JSON。
            response = stream.get_final_completion()
    diagnostics.update(
        model=response.model,
        response_id=response.id,
        request_id=getattr(response, "_request_id", None),
        finish_reason=response.choices[0].finish_reason if response.choices else None,
        usage=response.usage.model_dump() if response.usage else None,
    )
    if not response.choices or response.choices[0].finish_reason != "stop":
        raise ValueError("裁判输出未完整结束")
    content = response.choices[0].message.content
    if not content:
        raise ValueError("裁判未返回可见正文")
    score = QualityScore.model_validate_json(content)
    return {
        "dimensions": score.model_dump(),
        "model": response.model,
        "response_id": response.id,
        "usage": response.usage.model_dump() if response.usage else None,
    }
