from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from orderagent.middleware import OrderToolPolicyMiddleware
from orderagent.models import create_deepseek_model
from orderagent.prompts.order_prompt import ORDER_SYSTEM_PROMPT
from orderagent.repositories import JsonOrderRepository
from orderagent.schemas.order_schema import ToolContext
from orderagent.settings import Settings
from orderagent.tools import OrderQueryTool


def create_order_agent(settings: Settings, *, model: BaseChatModel | None = None):
    """Build a real Deep Agents graph; LangGraph owns the execution loop."""
    query_tool = OrderQueryTool(JsonOrderRepository(settings.orders_path))
    return create_deep_agent(
        model=model if model is not None else create_deepseek_model(settings),
        tools=[query_tool.as_langchain_tool()],
        system_prompt=ORDER_SYSTEM_PROMPT,
        context_schema=ToolContext,
        checkpointer=InMemorySaver(),
        middleware=[
            # Framework requires read_file internally; policy hides AND blocks it.
            FilesystemMiddleware(tools=["read_file"]),
            OrderToolPolicyMiddleware(),
            ModelCallLimitMiddleware(run_limit=settings.max_model_calls, exit_behavior="error"),
            ToolCallLimitMiddleware(run_limit=settings.max_tool_calls, exit_behavior="error"),
        ],
        # Default in-memory backend, never a local FilesystemBackend.
        name="order_query_agent",
    )
