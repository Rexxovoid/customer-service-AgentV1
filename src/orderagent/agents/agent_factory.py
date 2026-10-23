from collections.abc import Callable
from datetime import datetime

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from orderagent.integrations.handoff import HandoffGateway, MockHandoffGateway
from orderagent.integrations.ragflow import RagflowKnowledge
from orderagent.integrations.returns import MockReturnGateway, ReturnGateway
from orderagent.middleware import BusinessToolPolicyMiddleware
from orderagent.models import create_deepseek_model
from orderagent.prompts.order_prompt import ORDER_SYSTEM_PROMPT
from orderagent.repositories import JsonOrderRepository, JsonProductRepository
from orderagent.repositories.after_sales_repository import AfterSalesRepository
from orderagent.repositories.support_repository import JsonSupportRepository
from orderagent.schemas.order_schema import ToolContext
from orderagent.schemas.return_schema import ReturnAgentState
from orderagent.settings import Settings
from orderagent.tools import ProductQueryTool, create_order_query_tool
from orderagent.tools.return_tools import create_return_tool
from orderagent.tools.support_tools import create_support_tools


def create_order_agent(
    settings: Settings,
    *,
    model: BaseChatModel | None = None,
    handoff_gateway: HandoffGateway | None = None,
    return_gateway: ReturnGateway | None = None,
    return_clock: Callable[[], datetime] | None = None,
    support_repository=None,
):
    """组装官方Agent图、业务工具、中间件和内存检查点，支持注入测试模型或客服适配器。"""
    after_sales = AfterSalesRepository(settings.after_sales_db_path)
    order_repository = JsonOrderRepository(settings.orders_path, after_sales)
    agent_model = model if model is not None else create_deepseek_model(settings)
    query_tool = create_order_query_tool(order_repository)
    product_tool = ProductQueryTool(JsonProductRepository(settings.products_path))
    gateway = (
        handoff_gateway
        if handoff_gateway is not None
        else MockHandoffGateway(settings.mock_handoff_scenario)
    )
    if support_repository is None:
        if settings.ragflow_api_url:
            if not settings.ragflow_api_key.get_secret_value():
                raise ValueError("缺少RAGFLOW_API_KEY")
            support_repository = RagflowKnowledge(settings, agent_model)
        else:
            support_repository = JsonSupportRepository([settings.support_path])
    support_tools = create_support_tools(
        support_repository, gateway, order_repository=order_repository
    )
    return_tool = create_return_tool(
        order_repository,
        return_gateway
        if return_gateway is not None
        else MockReturnGateway(settings.mock_return_scenario),
        gateway,
        settings.return_self_service_entry,
        clock=return_clock,
        rule_repository=support_repository if settings.ragflow_api_url else None,
    )
    return create_deep_agent(
        model=agent_model,
        tools=[query_tool, product_tool.as_langchain_tool(), *support_tools, return_tool],
        system_prompt=ORDER_SYSTEM_PROMPT,
        context_schema=ToolContext,
        state_schema=ReturnAgentState,
        checkpointer=InMemorySaver(),
        middleware=[
            # 框架内部需要read_file；业务策略同时隐藏并拦截该工具。
            FilesystemMiddleware(tools=["read_file"]),
            BusinessToolPolicyMiddleware(gateway, order_repository, after_sales),
            ModelCallLimitMiddleware(run_limit=settings.max_model_calls, exit_behavior="error"),
            ToolCallLimitMiddleware(run_limit=settings.max_tool_calls, exit_behavior="error"),
        ],
        # 使用默认内存后端，不让模型通过框架工具访问本机文件系统。
        name="order_query_agent",
    )
