import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Literal
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from orderagent.integrations.handoff import HandoffGateway
from orderagent.integrations.ragflow import RAG_ERROR_MESSAGES
from orderagent.integrations.returns import ReturnGateway
from orderagent.repositories.after_sales_repository import AfterSalesRepository
from orderagent.repositories.order_repository import OrderRepository
from orderagent.schemas.order_schema import StoredOrder, ToolContext, ToolResult
from orderagent.schemas.return_schema import (
    ReturnLinkRequest,
    ReturnLinkResponse,
    ReturnOperation,
    ReturnQuery,
)
from orderagent.schemas.support_schema import ReportedFact, SupportQuery
from orderagent.tools.session_policy import (
    AUTO_HANDOFF_FROM,
    HANDOFF_SUGGESTION,
    SUGGEST_HANDOFF_FROM,
    advance_question,
)
from orderagent.tools.support_tools import support_reply, transfer_support

CST = timezone(timedelta(hours=8))


def return_handoff(
    state: dict,
    context: ToolContext | None,
    thread_id: str,
    gateway: HandoffGateway,
    reason: str = "同一退货问题第6次有效求解仍未解决或用户明确要求人工",
) -> dict:
    """将达到次数阈值或明确要求人工的退货诉求交接，保留订单和原因摘要。"""
    issue = (
        f"退货订单={state.get('return_order_id') or '未提供'}；"
        f"原因={state.get('return_reason') or '未提供'}"
    )
    update = transfer_support(
        {**state, "support_issue": issue},
        context,
        thread_id,
        reason + "；" + issue,
        gateway,
        topic="return",
    )
    return {**update, "support_issue": issue, "return_stage": "handoff"}


def check_return_operation(
    order: StoredOrder, context: ToolContext, store: AfterSalesRepository, now: datetime
) -> tuple[str, str, ReturnOperation | None]:
    """核验当前订单幂等状态；只接受可信记录，用户文本和RAG不能写入退货终态。

    调用方须先通过find_owned定位订单；本函数再次核对用户，防止误传他人订单。
    返回new/reuse/pending/blocked/expired，不负责判断质量或生成入口。
    """
    if order.user_id != context.user_id:
        return "blocked", "订单不存在或不属于当前用户。", None
    operation = store.get_return_operation(context.user_id, order.id)
    if operation and (operation.user_id != context.user_id or operation.order_id != order.id):
        raise ValueError("退货记录身份不一致")
    statuses = {
        order.order_status.casefold(),
        order.after_sales_status.casefold(),
        store.get_status(context.user_id, order.id).casefold(),
    }
    if operation:
        statuses.add(operation.status)
    if statuses & {"returned", "refunded", "已退货", "已退款", "退货完成", "退款完成"}:
        return "blocked", "该订单已退货或已退款，拒绝重复办理退货。", operation
    if statuses & {
        "returning",
        "return_processing",
        "refund_processing",
        "退货中",
        "退款中",
        "退货处理中",
        "退款处理中",
        "退货申请处理中",
        "processing",
    }:
        return "pending", "该订单已有退货/退款正在处理中，不重复发起。", operation
    if operation:
        if operation.status == "failed":
            return "pending", "退货入口明确失败，请后台核查或联系人工，不自动重试。", operation
        if operation.status in {"creating", "uncertain"}:
            return "pending", "该订单的退货操作正在执行或结果待核查，不重复调用接口。", operation
        response = operation.response
        if response is None or response.expires_at is None:
            return "pending", "已有退货记录，但入口结果无法核实，请通过后台核查。", operation
        if response.expires_at.tzinfo is None or response.expires_at <= now:
            return "expired", "原退货入口已过期，不重复创建，请通过后台查看原申请。", operation
        return "reuse", "复用该订单已有入口，不重复办理。", operation
    if statuses & {"已提供退货测试入口", "已提供退货入口", "退货操作待核查"}:
        return "pending", "已有旧版退货入口记录但无可核验响应，请后台核查，不重复创建。", None
    return "new", "尚无退货操作，可继续按用户描述查询规则。", None


def validate_return_link(
    response: ReturnLinkResponse, operation: ReturnOperation, gateway: ReturnGateway, now: datetime
) -> None:
    """验证新生成或复用的入口身份、域名、模拟标记和有效期，失败时禁止展示链接。"""
    parsed = urlsplit(response.url or "")
    if (
        response.status != "ready"
        or response.request_id != operation.request_id
        or response.order_id != operation.order_id
        or response.user_id != operation.user_id
        or response.expires_at is None
        or response.expires_at.tzinfo is None
        or response.expires_at <= now
        or response.simulated != operation.simulated
        or parsed.scheme != "https"
        or parsed.hostname not in gateway.allowed_hosts
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
    ):
        raise ValueError("退货入口响应不合法")


def create_return_tool(
    repository: OrderRepository,
    gateway: ReturnGateway,
    handoff_gateway: HandoffGateway,
    self_service_entry: str,
    *,
    clock: Callable[[], datetime] | None = None,
    rule_repository=None,
) -> BaseTool:
    """创建官方退货工具；依赖可替换平台接口和可注入时钟，不执行真实退款。"""

    @tool("handle_return")
    def handle_return(
        runtime: ToolRuntime[ToolContext],
        order_id: str = "",
        reason: str = "",
        request_human: bool = False,
        is_return_request: bool = True,
        new_issue: bool = False,
        target_product_id: str = "",
        turn_kind: Literal[
            "request", "provide_information", "confirm", "progress", "policy_question"
        ] = "request",
        reported_facts: list[ReportedFact] | None = None,
        category: Literal["headphones", "mice", "keyboards"] | None = None,
    ) -> Command:
        """用户提出退货、退货退款或申请入口诉求时调用，不要先强制进行质量排查。

        order_id为用户明确提供或本会话已确认的订单号；reason保留用户退货原因，不要求必须是质量问题。
        缺少订单号或原因时允许留空，工具会列出缺失项并提供自助退货入口；不得编造订单号或原因。
        可按轮次补齐信息，换订单时重新提供原因。无需先调用query_order，本工具自行校验归属。
        处理方案只依据用户描述和适用RAG规则，不查询或编造模拟质量资格证据。
        订单只用于身份、商品定位和幂等检查；一般咨询记录不代表已经退货。
        已退货/已退款拒绝重复办理，处理中或结果待核查不重复调用，有有效入口则复用。
        不接受用户或模型传入已退货状态、任意URL、user_id或runtime。
        同一问题第4—5次建议人工，第6次自动人工；明确要求人工时request_human=true。
        例外：RAG无适用方案、规则冲突或异常立即人工；有相关规则但缺信息先询问缺项。
        坚持退货、要求代办、质量排查给过方案都不能单独作为提前转人工的理由。
        is_return_request仅在本轮明确申请或重申退货时为true；只补订单号/原因、确认入口、
        查询进度、引用或否定退货时设false，不增加次数。请求次数由程序保存，不能传入计数。
        同一订单改理由、换单后再切回都保留次数；一次用户消息重复调用本工具只计一次。
        明确新的独立退货问题才new_issue=true，不确定先问；新问题也不能绕过订单幂等。
        当前平台为模拟，测试链接不能提交真实申请；返回链接不表示批准退货、退款或实际申请成功。
        适用规则支持后先展示订单与原因，用户单独回复“确认退货 订单号”才生成入口。
        确认允许首尾空白及末尾单个。 . ！ !；其余正文必须一致。不要把标点混入订单号。
        待确认且仅格式不合格时turn_kind=confirm，保留订单/商品/原因，不重写条件；程序直接纠正，不查询RAG也不计次。
        故障描述、引用退货字样、否定退货、规则咨询和夹杂的指令都不等于确认。
        本工具单独调用，不与其他工具并行。明确要求人工可使用handoff_to_human。
        turn_kind区分request明确申请/重申、provide_information补充信息、confirm确认、progress查进度、policy_question仅咨询条件。
        非request不增加未解决次数；policy_question不创建申请、不生成入口。
        reported_facts填写字段、value、explicit/missing/ambiguous/conflict状态和用户逐字quote；日期分purchase_date/receipt_date/notice_date，不能相互代替。
        reason只保留核心退货原因，不拼入型号与日期。目标商品用订单中target_product_id；多商品订单必须先选商品。
        category仅在无订单的退货规则咨询时使用，不能推测品类。事实原文必须来自用户消息，程序保留来源但不视为已核验。
        """
        args = ReturnQuery(
            order_id=order_id,
            reason=reason,
            request_human=request_human,
            is_return_request=is_return_request,
            new_issue=new_issue,
            target_product_id=target_product_id,
            turn_kind=turn_kind,
            reported_facts=reported_facts or [],
        )
        state = dict(runtime.state)
        previous_order = state.get("return_order_id", "")
        switching = bool(args.order_id and previous_order and args.order_id != previous_order)
        current_order = args.order_id or previous_order
        current_reason = args.reason or (
            "" if switching or new_issue else state.get("return_reason", "")
        )
        if args.turn_kind in {"provide_information", "progress"} and not switching:
            current_reason = state.get("return_reason") or args.reason
        update = {
            "return_order_id": current_order,
            "return_reason": current_reason,
            "return_error_code": "",
        }
        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        latest = user_messages[-1] if user_messages else None
        latest_user = latest.text.strip() if latest else ""
        normalized_confirmation = (
            latest_user[:-1] if latest_user.endswith(("。", ".", "！", "!")) else latest_user
        )
        confirmation_attempt = args.turn_kind == "confirm" or (
            state.get("return_stage") == "awaiting_confirmation" and "确认退货" in latest_user
        )
        facts = {} if switching or new_issue else dict(state.get("return_reported_facts", {}))
        invalid_fact = False
        for fact in args.reported_facts:
            source = next(
                (m for m in reversed(user_messages) if fact.quote and fact.quote in m.text), None
            )
            if source is None:
                invalid_fact = True
                continue
            if fact.status == "explicit" and (not fact.value or fact.value not in fact.quote):
                invalid_fact = True
                continue
            previous_fact = facts.get(fact.field)
            facts[fact.field] = {
                **fact.model_dump(),
                "source": "user_statement",
                "message_id": source.id,
                "verified": False,
            }
            if (
                previous_fact
                and previous_fact["status"] == "explicit"
                and fact.status == "explicit"
                and previous_fact["value"] != fact.value
            ):
                facts[fact.field]["status"] = "conflict"
                facts[fact.field]["previous_value"] = previous_fact["value"]
        statements = (
            [] if switching or new_issue else list(state.get("return_original_statements", []))
        )
        if latest_user and (not statements or statements[-1] != latest_user):
            statements.append(latest_user)
        target = args.target_product_id or (
            "" if switching else state.get("return_target_product_id", "")
        )
        update.update(
            return_reported_facts=facts,
            return_original_statements=statements[-20:],
            return_target_product_id=target,
        )
        has_request = (not switching and state.get("return_has_request", False)) or (
            args.is_return_request and args.turn_kind == "request"
        )
        update["return_has_request"] = has_request
        update.update(
            advance_question(
                state,
                "return",
                current_order or "__unbound__",
                new_issue=new_issue,
                countable=is_return_request
                and args.turn_kind == "request"
                and not confirmation_attempt,
            )
        )
        entry = {"count": update["return_request_count"]}
        update["support_handoff"] = {}
        combined = {**state, **update}
        thread_id = runtime.config.get("configurable", {}).get("thread_id", "")

        def finish(message: str, stage: str, *, ok: bool = True, **details) -> Command:
            """把确定的用户回复和状态作为官方Command返回，不再让模型续写解释。"""
            if SUGGEST_HANDOFF_FROM <= entry["count"] < AUTO_HANDOFF_FROM and stage != "handoff":
                message += HANDOFF_SUGGESTION
            if (
                current_order
                and has_request
                and args.turn_kind != "policy_question"
                and isinstance(runtime.context, ToolContext)
                and (
                    state.get("return_stage")
                    or (args.is_return_request and latest_user != f"确认退货 {current_order}")
                )
                and getattr(repository, "after_sales", None)
                and repository.find_owned(current_order, runtime.context.user_id)
            ):
                progress = {
                    "collecting": "退货申请待补充",
                    "guided": "退货申请待核查",
                    "awaiting_confirmation": "退货申请待确认",
                    "link_ready": "已提供退货测试入口",
                    "handoff": "退货已转人工（模拟）",
                    "rejected": "退货请求被拒绝",
                }[stage]
                repository.after_sales.record(
                    runtime.context.user_id, current_order, progress, runtime.tool_call_id
                )
            data = {
                "action": "return_result",
                "message": message,
                "request_count": entry["count"],
                "code": update.get("return_error_code") or None,
                **details,
            }
            result = ToolResult(ok=ok, data=data)
            return Command(
                update={
                    **update,
                    "return_stage": stage,
                    "return_result": data,
                    "messages": [
                        ToolMessage(
                            content=result.model_dump_json(),
                            artifact=result.model_dump(mode="json"),
                            tool_call_id=runtime.tool_call_id,
                            name="handle_return",
                            status="success" if ok else "error",
                        )
                    ],
                }
            )

        def handoff(reason: str = "同一退货问题第6次有效求解仍未解决") -> Command:
            """达到计数阈值或明确请求时复用现有交接接口。"""
            nonlocal update
            update = {
                **update,
                **return_handoff(
                    {**combined, **update}, runtime.context, thread_id, handoff_gateway, reason
                ),
            }
            update["return_error_code"] = (update.get("support_handoff") or {}).get(
                "code"
            ) or update.get("return_error_code", "")
            merged = {**combined, **update}
            return finish(
                support_reply(merged), "handoff", ok=merged.get("support_phase") == "transferred"
            )

        if not isinstance(runtime.context, ToolContext):
            return finish(
                "缺少可信用户身份，无法核查退货订单。" + self_service_entry, "guided", ok=False
            )
        prior_stage = None if switching or new_issue else state.get("return_stage")
        if args.request_human:
            return handoff("用户明确要求人工")
        if args.turn_kind == "policy_question":
            policy_category = category
            if current_order:
                policy_order = repository.find_owned(current_order, runtime.context.user_id)
                if policy_order is None:
                    return finish(
                        "订单不存在或不属于当前用户。", "guided", ok=False, code="ORDER_NOT_FOUND"
                    )
                prefixes = {item.product_id.split("-", 1)[0] for item in policy_order.items}
                policy_category = (
                    {"HP": "headphones", "MS": "mice", "KB": "keyboards"}.get(next(iter(prefixes)))
                    if len(prefixes) == 1
                    else None
                )
            if not policy_category:
                return finish("请说明要咨询的商品品类；规则咨询不代表申请退货。", "guided")
            try:
                rule = (
                    rule_repository.find(
                        SupportQuery(
                            category=policy_category,
                            task="return_policy",
                            issue=f"退货规则与适用条件：{args.reason or latest_user}"[:500],
                            original_statements=[latest_user],
                        )
                    )
                    if rule_repository
                    else None
                )
            except Exception as exc:
                update["return_error_code"] = (
                    str(exc) if str(exc) in RAG_ERROR_MESSAGES else "KNOWLEDGE_SERVICE_ERROR"
                )
                return handoff("退货规则咨询无法可靠处理")
            if rule is None:
                return handoff("未找到适用退货规则")
            # 仅知识咨询，不写订单进度，不把guided当作已申请。
            return finish(
                "退货规则说明（未申请退货）：" + rule.answer, "guided", rule_source=rule.source
            )
        if not has_request:
            return finish(
                "尚未收到明确退货申请，不创建入口。如需退货请明确说明订单和原因。"
                + self_service_entry,
                "guided",
            )
        missing = [
            label
            for label, value in (("订单号", current_order), ("退货原因", current_reason))
            if not value
        ]
        if missing:
            if entry["count"] >= AUTO_HANDOFF_FROM or prior_stage == "handoff":
                return handoff()
            return finish(
                "办理退货还需补充："
                + "、".join(missing)
                + "。如希望自行申请，"
                + self_service_entry,
                "collecting",
                missing_fields=missing,
            )
        order = repository.find_owned(current_order, runtime.context.user_id)
        if order is None:
            return finish(
                "订单不存在或不属于当前用户，无法核查退货。" + self_service_entry,
                "guided",
                ok=False,
                code="ORDER_NOT_FOUND",
            )
        now = clock() if clock is not None else datetime.now(CST)
        store = getattr(repository, "after_sales", None)
        if store is None:
            return handoff("缺少持久退货记录，无法确保幂等")
        try:
            action, explanation, existing = check_return_operation(
                order, runtime.context, store, now
            )
        except Exception:
            return handoff("退货记录无法核实，停止自动办理以防重复操作")
        if action == "blocked":
            return finish(explanation, "rejected", ok=False, code="RETURN_ALREADY_COMPLETED")
        if action not in {"new", "reuse"}:
            return finish(
                explanation + self_service_entry,
                "guided",
                ok=False,
                code="RETURN_IN_PROGRESS" if action == "pending" else "RETURN_LINK_EXPIRED",
            )
        if invalid_fact:
            return finish(
                "部分信息缺少可核对的用户原文，请重新说明相关条件。",
                "collecting",
                code="UNSUPPORTED_USER_FACT",
            )
        selected_items = (
            [item for item in order.items if item.product_id == target] if target else order.items
        )
        if len(selected_items) != 1:
            return finish(
                "请明确要退货的商品ID，不合并多个商品的型号和条件。",
                "collecting",
                missing_fields=["目标商品"],
                items=[item.model_dump() for item in order.items],
            )
        target = selected_items[0].product_id
        update["return_target_product_id"] = target
        uncertain = [
            key for key, fact in facts.items() if fact["status"] in {"ambiguous", "conflict"}
        ]
        if uncertain:
            return finish(
                "以下信息模糊或存在冲突，请确认：" + "、".join(uncertain),
                "collecting",
                missing_fields=uncertain,
            )
        # 只对用户逐字提供的ISO日期做确定性一致性检查；不把模糊自然语言换算为日期。
        dates = {}
        for key in ("purchase_date", "receipt_date", "notice_date"):
            fact = facts.get(key)
            if fact and fact["status"] == "explicit":
                try:
                    dates[key] = datetime.strptime(fact["value"], "%Y-%m-%d").date()
                except ValueError:
                    pass
        date_conflict = any(value > now.date() for value in dates.values())
        date_conflict = (
            date_conflict
            or (
                "purchase_date" in dates
                and "receipt_date" in dates
                and dates["purchase_date"] > dates["receipt_date"]
            )
            or (
                "receipt_date" in dates
                and "notice_date" in dates
                and dates["receipt_date"] > dates["notice_date"]
            )
        )
        if date_conflict:
            return finish(
                "日期存在未来时间或先后矛盾，请核对购买、签收和退货通知日期。",
                "collecting",
                code="DATE_CONFLICT",
            )

        def show_link(operation: ReturnOperation) -> Command:
            """统一展示经过验证的原入口或新入口，不把入口生成描述为退货完成。"""
            response = operation.response
            validate_return_link(response, operation, gateway, now)
            prefix = "【模拟退货】" if response.simulated else ""
            notice = (
                "此为测试链接，不可提交真实平台申请。"
                if response.simulated
                else "请进入平台核对后自行提交。"
            )
            return finish(
                prefix
                + "根据用户描述及适用规则提供申请入口："
                + response.url
                + "\n"
                + notice
                + "仅提供入口，不代表申请已提交、质量已鉴定、退货获批或退款完成。",
                "link_ready",
                rule_supported=True,
                url=response.url,
                simulated=response.simulated,
                request_id=response.request_id,
                expires_at=response.expires_at.isoformat(),
            )

        if (
            entry["count"] >= AUTO_HANDOFF_FROM or prior_stage == "handoff"
        ) and not confirmation_attempt:
            return handoff()
        if action == "reuse":
            try:
                return show_link(existing)
            except Exception:
                return handoff("已存退货入口响应无法验证，禁止重复创建")
        if action != "new":
            return finish(
                explanation + self_service_entry,
                "guided",
                ok=False,
                code="RETURN_IN_PROGRESS" if action == "pending" else "RETURN_LINK_EXPIRED",
            )
        binding_unchanged = state.get("return_confirmation_payload") == {
            "order_id": current_order,
            "product_id": target,
            "reason": current_reason,
            "facts": facts,
        }
        if (
            prior_stage == "awaiting_confirmation"
            and binding_unchanged
            and normalized_confirmation != f"确认退货 {current_order}"
        ):
            if confirmation_attempt:
                return finish(
                    "确认格式未通过校验，未生成入口。请单独回复以下指令（允许末尾一个句号或感叹号）：\n"
                    f"确认退货 {current_order}",
                    "awaiting_confirmation",
                    code="RETURN_CONFIRMATION_INVALID",
                )
        if entry["count"] >= AUTO_HANDOFF_FROM or prior_stage == "handoff":
            return handoff()
        confirmed = (
            prior_stage == "awaiting_confirmation"
            and normalized_confirmation == f"确认退货 {current_order}"
            and current_order == previous_order
            and current_reason == state.get("return_reason")
            and binding_unchanged
        )
        if rule_repository is None:
            return handoff("未配置可用的售后规则知识库")
        if rule_repository is not None:
            product_ids = [item.product_id.upper() for item in selected_items]
            category = (
                "headphones"
                if all(value.startswith("HP-P-") for value in product_ids)
                else "mice"
                if all(value.startswith("MS-P-") for value in product_ids)
                else "keyboards"
                if all(value.startswith("KB-P-") for value in product_ids)
                else None
            )
            if category is None:
                return finish(
                    "订单品类无法自动匹配规则，请通过后台核查。" + self_service_entry,
                    "guided",
                    ok=False,
                )
            try:
                rule = rule_repository.find(
                    SupportQuery(
                        category=category,
                        issue=f"退货规则与适用条件：用户报告：{current_reason[:470]}",
                        environment=selected_items[0].name[:200],
                        task="return",
                        reported_facts=list(facts.values()),
                        original_statements=list(
                            dict.fromkeys(
                                [fact["quote"] for fact in facts.values()] + statements[-2:]
                            )
                        ),
                    )
                )
            except Exception as exc:
                code = (
                    str(exc)
                    if isinstance(exc, ValueError) and str(exc) in RAG_ERROR_MESSAGES
                    else "KNOWLEDGE_SERVICE_ERROR"
                )
                update["return_error_code"] = code
                return handoff(f"{code}：{RAG_ERROR_MESSAGES.get(code, '售后规则知识服务异常')}")
            if rule and rule.source.get("handling") == "clarify" and rule.missing_information:
                return finish(
                    "已找到相关退货规则，还需补充：" + "、".join(rule.missing_information),
                    "collecting",
                    missing_fields=rule.missing_information,
                )
            if rule is None or rule.source.get("handling") != "return_eligible":
                update["return_error_code"] = "KNOWLEDGE_NO_RETURN_RULE"
                return handoff("未命中适用退货方案或规则不支持自动处理")
            update["return_rule_source"] = rule.model_dump(mode="json")
        if not confirmed:
            update["return_confirmation_payload"] = {
                "order_id": current_order,
                "product_id": target,
                "reason": current_reason,
                "facts": facts,
            }
            return finish(
                f"请确认是否为订单 {current_order} 的商品 {target} 申请退货，"
                f"原因：{current_reason}。"
                f"\n如确认，请单独回复以下指令：\n确认退货 {current_order}"
                "\n目前仅登记待确认引导，未生成链接或提交申请，仍未完成售后；如不退货可直接说明。",
                "awaiting_confirmation",
            )
        request = ReturnLinkRequest(
            request_id="RETURN-"
            + uuid5(
                NAMESPACE_URL,
                json.dumps([runtime.context.user_id, current_order], ensure_ascii=False),
            ).hex,
            order_id=current_order,
            user_id=runtime.context.user_id,
            reason=current_reason,
        )
        operation = ReturnOperation(
            user_id=runtime.context.user_id,
            order_id=current_order,
            request_id=request.request_id,
            status="creating",
            simulated=getattr(gateway, "simulated", True),
        )
        try:
            claimed = store.claim_return_operation(operation)
        except Exception:
            return handoff("无法原子检查退货记录，禁止调用创建接口")
        if not claimed:
            try:
                action, explanation, existing = check_return_operation(
                    order, runtime.context, store, now
                )
                if action == "reuse":
                    return show_link(existing)
                return finish(
                    explanation,
                    "rejected" if action == "blocked" else "guided",
                    ok=False,
                    code="RETURN_ALREADY_COMPLETED"
                    if action == "blocked"
                    else "RETURN_IN_PROGRESS",
                )
            except Exception:
                return handoff("并发退货结果无法核实，停止重复调用")
        try:
            response = ReturnLinkResponse.model_validate(gateway.create_link(request, now))
            if (
                response.status == "error"
                and response.request_id == operation.request_id
                and response.user_id == operation.user_id
                and response.order_id == operation.order_id
                and response.simulated == operation.simulated
            ):
                store.update_return_operation(operation.model_copy(update={"status": "failed"}))
                return finish(
                    "退货入口生成失败：接口明确返回失败，已登记，不自动重试。" + self_service_entry,
                    "guided",
                    ok=False,
                    code="RETURN_CREATE_FAILED",
                )
            validate_return_link(response, operation, gateway, now)
            completed = operation.model_copy(update={"status": "link_ready", "response": response})
            store.update_return_operation(completed)
        except Exception:
            # 接口可能已经执行，不能删除占用后盲目重试；即使记录更新失败也保留creating。
            try:
                store.update_return_operation(operation.model_copy(update={"status": "uncertain"}))
            except Exception:
                pass
            return finish(
                "退货入口生成失败或结果无法确认，已保留待核查记录，不重复办理。"
                + self_service_entry,
                "guided",
                ok=False,
                code="RETURN_RESULT_UNCERTAIN",
            )
        return show_link(completed)

    return handle_return
