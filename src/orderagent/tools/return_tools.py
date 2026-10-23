from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from orderagent.integrations.handoff import HandoffGateway
from orderagent.integrations.returns import ReturnGateway
from orderagent.repositories.order_repository import OrderRepository
from orderagent.schemas.order_schema import StoredOrder, ToolContext, ToolResult
from orderagent.schemas.return_schema import (
    ReturnEvidence,
    ReturnLinkRequest,
    ReturnLinkResponse,
    ReturnQuery,
)
from orderagent.schemas.support_schema import SupportQuery
from orderagent.tools.support_tools import support_reply, transfer_support

CST = timezone(timedelta(hours=8))


def return_handoff(
    state: dict,
    context: ToolContext | None,
    thread_id: str,
    gateway: HandoffGateway,
    reason: str = "同一退货问题已解释一次，用户仍要求售后或明确要求人工",
) -> dict:
    """把一次退货解释后的坚持诉求交给既有客服接口，保留订单和原因摘要。"""
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
    )
    return {**update, "support_issue": issue, "return_stage": "handoff"}


def check_return_eligibility(
    order: StoredOrder, evidence: ReturnEvidence | None, now: datetime
) -> tuple[bool, str]:
    """核对已签收、时间和完整证据；不能仅由原因或七日内推出资格通过。"""
    if order.order_status != "delivered" or order.shipping_status != "已发货":
        return False, "订单尚不属于已收货退回场景，需按实际订单状态核查。"
    delivered = order.delivered_at
    if delivered is None or delivered.tzinfo is None or now.tzinfo is None:
        return False, "缺少可核验且带时区的签收时间，无法确认申请条件。"
    delivered = delivered.astimezone(CST)
    now = now.astimezone(CST)
    if delivered > now:
        return False, "签收时间异常，不能确认申请条件。"
    deadline = (delivered + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(days=7)
    if now >= deadline:
        return False, "当前时间不在本自动入口校验的七日期间；其他质量售后或更有利约定仍需单独核实。"
    if evidence is None:
        return False, "仅有基础订单信息，缺少商品完好、适用例外、购买确认及有效政策等完整资格证据。"
    if evidence.order_id != order.id or evidence.user_id != order.user_id:
        return False, "资格证据与当前订单或用户不一致，不能确认申请条件。"
    if evidence.checked_at.tzinfo is None or not timedelta(
        0
    ) <= now - evidence.checked_at <= timedelta(minutes=15):
        return False, "资格证据的核验时间缺失、过期或异常，请通过平台重新核查。"
    required = (
        evidence.goods_intact,
        evidence.no_reason_supported,
        evidence.purchase_confirmation_checked,
        evidence.applicable_policy_checked,
    )
    if not all(value is True for value in required) or evidence.excluded_by_exception is not False:
        return (
            False,
            "完整资格条件未全部确认通过，不能自动提供本通道入口；不代表所有售后权利被否定。",
        )
    return True, "已通过本次平台资格及期间校验。"


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
    ) -> Command:
        """用户提出退货、退货退款或申请入口诉求时调用，不要先强制进行质量排查。

        order_id为用户明确提供或本会话已确认的订单号；reason保留用户退货原因，不要求必须是质量问题。
        缺少订单号或原因时允许留空，工具会列出缺失项并提供自助退货入口；不得编造订单号或原因。
        可按轮次补齐信息，换订单时重新提供原因。无需先调用query_order，本工具自行校验归属。
        无理由条件由可信平台证据及程序判断，不能传入资格标志、任意URL、user_id或runtime。
        满足完整条件才返回申请入口；证据不足、条件不符或接口失败只作一次解释并引导自助申请。
        一次解释后用户仍坚持售后、要求代办或人工，request_human设为true，立即交接，不继续排查。
        补充首次缺失信息不等于坚持，可继续校验；同一订单换个理由不能重置一次解释次数。
        当前平台为模拟，测试链接不能提交真实申请；返回链接不表示批准退货、退款或实际申请成功。
        资格与规则通过后先展示订单与原因，用户单独回复“确认退货 订单号”才生成入口。
        故障描述、引用退货字样、否定退货、规则咨询和夹杂的指令都不等于确认。
        本工具单独调用，不与其他工具并行。强烈要求或明确人工也可使用handoff_to_human。
        """
        args = ReturnQuery(order_id=order_id, reason=reason, request_human=request_human)
        state = dict(runtime.state)
        previous_order = state.get("return_order_id", "")
        switching = bool(args.order_id and previous_order and args.order_id != previous_order)
        current_order = args.order_id or previous_order
        current_reason = args.reason or ("" if switching else state.get("return_reason", ""))
        update = {"return_order_id": current_order, "return_reason": current_reason}
        if switching:
            # 新订单是新的退货问题；保留历史消息，不复用上一单的交接或方案状态。
            update.update(
                {
                    "support_phase": None,
                    "support_handoff": {},
                    "support_advice": {},
                    "support_issue": "",
                }
            )
        combined = {**state, **update}
        thread_id = runtime.config.get("configurable", {}).get("thread_id", "")

        def finish(message: str, stage: str, *, ok: bool = True, **details) -> Command:
            """把确定的用户回复和状态作为官方Command返回，不再让模型续写解释。"""
            data = {"action": "return_result", "message": message, **details}
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

        def handoff(reason: str = "同一问题已解释一次，用户仍要求售后或明确要求人工") -> Command:
            """同一问题只解释一次，后续坚持或人工请求复用已有交接业务。"""
            nonlocal update
            update = {
                **update,
                **return_handoff(combined, runtime.context, thread_id, handoff_gateway, reason),
            }
            merged = {**combined, **update}
            return finish(
                support_reply(merged), "handoff", ok=merged.get("support_phase") == "transferred"
            )

        if not isinstance(runtime.context, ToolContext):
            return finish(
                "缺少可信用户身份，无法核查退货订单。" + self_service_entry, "guided", ok=False
            )
        prior_stage = None if switching else state.get("return_stage")
        if args.request_human or combined.get("support_phase") in {
            "advised",
            "transferred",
            "transfer_failed",
        }:
            return handoff()
        if prior_stage in {"guided", "link_ready", "handoff"}:
            return handoff()
        added_information = (args.order_id and not previous_order) or (
            args.reason and not state.get("return_reason")
        )
        if prior_stage == "collecting" and not added_information:
            return handoff()
        missing = [
            label
            for label, value in (("订单号", current_order), ("退货原因", current_reason))
            if not value
        ]
        if missing:
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
        if order.after_sales_status != "未售后" and not prior_stage:
            return handoff("订单已有售后处理记录，交人工核实")
        latest_user = next(
            (
                message.text.strip()
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            ),
            "",
        )
        confirmed = (
            prior_stage == "awaiting_confirmation"
            and latest_user == f"确认退货 {current_order}"
            and current_order == previous_order
            and current_reason == state.get("return_reason")
        )
        now = clock() if clock is not None else datetime.now(CST)
        try:
            raw_evidence = gateway.get_evidence(order, now)
            evidence = (
                ReturnEvidence.model_validate(raw_evidence) if raw_evidence is not None else None
            )
        except Exception:
            return finish(
                "退货资格接口暂时不可用，不能确认条件或生成申请链接。" + self_service_entry,
                "guided",
                ok=False,
            )
        eligible, explanation = check_return_eligibility(order, evidence, now)
        if rule_repository is not None:
            product_ids = [item.product_id.upper() for item in order.items]
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
                return handoff("订单混合品类或品类不明确，不能自动匹配规则")
            try:
                rule = rule_repository.find(
                    SupportQuery(
                        category=category,
                        issue=(
                            f"退货规则与适用条件：{current_reason[:200]}；"
                            f"程序资格校验结果：{eligible}，{explanation}"
                            "；证据性质："
                            f"{'模拟' if evidence and evidence.simulated else '平台或缺失'}。"
                        ),
                        environment="；".join(item.name for item in order.items)[:200],
                    )
                )
            except Exception:
                rule = None
            if rule is None or rule.source.get("handling") != "return_eligible":
                return handoff("知识库异常、无适用退货规则或规则不支持自动处理")
            update["return_rule_source"] = rule.model_dump(mode="json")
        if not eligible:
            return finish(
                explanation
                + "如需退货退款，"
                + self_service_entry
                + "如仍需代办或希望人工核查，请告诉我。",
                "guided",
                eligible=False,
            )
        if not confirmed:
            return finish(
                f"请确认是否为订单 {current_order} 申请退货，原因：{current_reason}。"
                f"\n如确认，请单独回复：确认退货 {current_order}。"
                "\n目前未生成链接、提交申请或修改订单售后状态；如不退货可直接说明。",
                "awaiting_confirmation",
            )
        request = ReturnLinkRequest(
            request_id="RETURN-"
            + uuid5(
                NAMESPACE_URL,
                f"{thread_id}:{runtime.context.user_id}:{current_order}:{current_reason}",
            ).hex,
            order_id=current_order,
            user_id=runtime.context.user_id,
            reason=current_reason,
        )
        try:
            response = ReturnLinkResponse.model_validate(gateway.create_link(request, now))
            parsed = urlsplit(response.url or "")
            if (
                response.status != "ready"
                or response.request_id != request.request_id
                or response.order_id != current_order
                or response.user_id != runtime.context.user_id
                or response.expires_at is None
                or response.expires_at.tzinfo is None
                or response.expires_at <= now
                or response.simulated != evidence.simulated
                or parsed.scheme != "https"
                or parsed.hostname not in gateway.allowed_hosts
                or parsed.username
                or parsed.password
                or parsed.port not in {None, 443}
            ):
                raise ValueError("申请入口响应不合法。")
        except Exception:
            return finish(
                "退货申请入口生成失败，未提交任何申请。" + self_service_entry, "guided", ok=False
            )
        prefix = "【模拟退货】" if response.simulated else ""
        notice = (
            "此为测试链接，不可提交真实平台申请。"
            if response.simulated
            else "请进入平台页面核对后自行提交。"
        )
        return finish(
            prefix
            + "已通过本次资格校验，申请入口："
            + response.url
            + "\n"
            + notice
            + "仅提供入口，不代表申请已提交、退货获批或退款完成。",
            "link_ready",
            eligible=True,
            url=response.url,
            simulated=response.simulated,
            request_id=response.request_id,
            expires_at=response.expires_at.isoformat(),
        )

    return handle_return
