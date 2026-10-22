from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from orderagent.integrations.handoff import HandoffGateway, MockHandoffGateway
from orderagent.middleware import BusinessToolPolicyMiddleware
from orderagent.models import create_deepseek_model
from orderagent.prompts.order_prompt import ORDER_SYSTEM_PROMPT
from orderagent.repositories import JsonOrderRepository, JsonProductRepository
from orderagent.repositories.support_repository import JsonSupportRepository
from orderagent.schemas.order_schema import ToolContext
from orderagent.schemas.support_schema import SupportAgentState
from orderagent.settings import Settings
from orderagent.tools import ProductQueryTool, create_order_query_tool
from orderagent.tools.support_tools import create_support_tools


def create_order_agent(
    settings: Settings,
    *,
    model: BaseChatModel | None = None,
    handoff_gateway: HandoffGateway | None = None,
):
    """组装官方Agent图、业务工具、中间件和内存检查点，支持注入测试模型或客服适配器。"""
    query_tool = create_order_query_tool(JsonOrderRepository(settings.orders_path))
    product_tool = ProductQueryTool(JsonProductRepository(settings.products_path))
    gateway = (
        handoff_gateway
        if handoff_gateway is not None
        else MockHandoffGateway(settings.mock_handoff_scenario)
    )
    support_tools = create_support_tools(JsonSupportRepository([settings.support_path]), gateway)
    return create_deep_agent(
        model=model if model is not None else create_deepseek_model(settings),
        tools=[query_tool, product_tool.as_langchain_tool(), *support_tools],
        system_prompt=ORDER_SYSTEM_PROMPT,
        context_schema=ToolContext,
        state_schema=SupportAgentState,
        checkpointer=InMemorySaver(),
        middleware=[
            # 框架内部需要read_file；业务策略同时隐藏并拦截该工具。
            FilesystemMiddleware(tools=["read_file"]),
            BusinessToolPolicyMiddleware(gateway),
            ModelCallLimitMiddleware(run_limit=settings.max_model_calls, exit_behavior="error"),
            ToolCallLimitMiddleware(run_limit=settings.max_tool_calls, exit_behavior="error"),
        ],
        # 使用默认内存后端，不让模型通过框架工具访问本机文件系统。
        name="order_query_agent",
    )
