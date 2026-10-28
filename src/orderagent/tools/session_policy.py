"""问题级会话策略；只生成官方Agent状态增量，不执行模型或平台请求。"""

from uuid import uuid4

from langchain_core.messages import HumanMessage

SUGGEST_HANDOFF_FROM = 4
AUTO_HANDOFF_FROM = 6
HANDOFF_SUGGESTION = "\n这个问题已多次尝试仍未解决，建议转人工；如需转接请明确告诉我。"


def advance_question(state, topic, scope, *, new_issue=False, countable=True):
    """按主题/订单问题计数，单条用户消息仅累计一次；补信息不增加次数。"""
    prefix = "support" if topic == "quality" else "return"
    history_key = "support_question_history" if topic == "quality" else "return_request_history"
    count_key = "support_question_count" if topic == "quality" else "return_request_count"
    latest = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    message_id = (latest.id or f"human-{len(state['messages'])}") if latest else ""
    history = {k: dict(v) for k, v in state.get(history_key, {}).items()}
    previous_scope = state.get(f"{prefix}_question_scope", "")
    # 先匿名咨询、后补订单时迁移当前问题；不是新诉求，不清零也不重复计数。
    anonymous = "__unbound__" if topic == "return" else scope.split(":", 1)[0] + ":__unbound__"
    if scope != anonymous and previous_scope == anonymous and anonymous in history:
        if scope not in history:
            history[scope] = history.pop(anonymous)
    entry = dict(history.get(scope, {"count": 0, "message_ids": [], "generation": 0}))
    seen = dict(state.get("question_processed_messages", {}))
    first_processing = message_id and message_id not in seen
    if new_issue and first_processing:
        entry = {"count": 0, "message_ids": [], "generation": entry.get("generation", 0) + 1}
    # 完整确认始终不计次，即使模型误填countable=True。
    confirmation_order = scope if topic == "return" else state.get("return_order_id", "")
    confirmation_text = latest.text.strip() if latest else ""
    if confirmation_text.endswith(("。", ".", "！", "!")):
        confirmation_text = confirmation_text[:-1]
    confirmation = bool(latest and confirmation_text == f"确认退货 {confirmation_order}")
    if countable and first_processing and not confirmation:
        entry = {
            **entry,
            "count": entry["count"] + 1,
            "message_ids": [*entry["message_ids"], message_id],
        }
    if message_id:
        seen[message_id] = topic
    history[scope] = entry
    entry.setdefault("problem_key", f"{scope}#{entry.get('generation', 0)}")
    return {
        "consultation_id": state.get("consultation_id") or uuid4().hex,
        history_key: history,
        count_key: entry["count"],
        f"{prefix}_question_scope": scope,
        f"{prefix}_problem_key": entry["problem_key"],
        "question_processed_messages": seen,
        "active_support_topic": topic,
    }


def reset_consultation():
    """结束会话级业务上下文，绝不删除SQLite记录或撤销平台申请。"""
    return {
        "consultation_id": uuid4().hex,
        "support_question_history": {},
        "support_question_count": 0,
        "support_question_scope": "",
        "support_problem_key": "",
        "return_request_history": {},
        "return_request_count": 0,
        "return_question_scope": "",
        "return_problem_key": "",
        "question_processed_messages": {},
        "active_support_topic": "",
        "support_phase": None,
        "support_handoff": {},
        "support_handoff_history": {},
        "support_issue": "",
        "support_advice": {},
        "support_advice_history": {},
        "support_order_id": "",
        "support_result": {},
        "support_error_code": "",
        "support_handoff_topic": "",
        "return_stage": None,
        "return_order_id": "",
        "return_reason": "",
        "return_result": {},
        "return_error_code": "",
        "return_rule_source": {},
        "return_target_product_id": "",
        "return_reported_facts": {},
        "return_original_statements": [],
        "return_confirmation_payload": {},
        "return_has_request": False,
    }
