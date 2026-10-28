import json

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


def grade_quality(settings, record):
    """仅调用官方兼容SDK；结构校验失败不重写答案、不伪造分数。"""
    if not settings.qiwen_api_key.get_secret_value():
        raise ValueError("缺少QIWEN_API_KEY")
    with OpenAI(
        api_key=settings.qiwen_api_key.get_secret_value(),
        base_url=settings.qiwen_base_url,
        timeout=30,
        max_retries=1,
    ) as client:
        response = client.chat.completions.create(
            model=settings.qianwen,
            messages=[
                {"role": "system", "content": load_prompt("judge_prompt", evaluation=True)},
                {"role": "user", "content": json.dumps(record, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            max_tokens=2000,
            temperature=0,
        )
    if response.choices[0].finish_reason != "stop":
        raise ValueError("裁判输出未完整结束")
    score = QualityScore.model_validate_json(response.choices[0].message.content or "")
    return {
        "dimensions": score.model_dump(),
        "model": response.model,
        "response_id": response.id,
        "usage": response.usage.model_dump() if response.usage else None,
    }
