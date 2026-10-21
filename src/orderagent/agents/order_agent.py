import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from orderagent.exceptions import AgentExecutionError
from orderagent.schemas.order_schema import AgentReply, ToolContext, ToolResult, ToolTrace


class OrderAgent:
    """Session adapter, not a tool execution loop. One instance per trusted user."""

    def __init__(self, graph, context: ToolContext):
        self.graph = graph
        self.context = context
        self._messages = []

    def reset(self) -> None:
        self._messages.clear()

    def chat(self, text: str) -> AgentReply:
        if not text.strip() or len(text) > 4000:
            raise ValueError("请输入1—4000字符的消息。")
        before = {message.id for message in self._messages if message.id}
        try:
            state = self.graph.invoke(
                {"messages": [*self._messages, HumanMessage(content=text.strip())]},
                context=self.context,
                config={"recursion_limit": 40},
            )
        except GraphRecursionError:
            raise AgentExecutionError("Agent执行步数达到上限，请缩小查询范围。") from None
        except Exception:
            raise AgentExecutionError(
                "Agent执行失败，请检查模型配置、调用限制或稍后重试。"
            ) from None
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage) or not messages[-1].text:
            raise AgentExecutionError("Agent没有生成有效回复，请重试。")
        traces = []
        for message in messages:
            if not isinstance(message, ToolMessage) or message.id in before:
                continue
            try:
                payload = message.artifact or json.loads(message.content)
                result = ToolResult.model_validate(payload)
            except (ValueError, TypeError, ValidationError):
                result = ToolResult.failure("TOOL_ERROR", "工具未返回有效结构化结果。")
            traces.append(
                ToolTrace(call_id=message.tool_call_id, tool=message.name or "", result=result)
            )
        # Retain complete recent turns; never split an assistant/tool pair.
        turns = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
        self._messages = messages[turns[-8] :] if len(turns) > 8 else messages
        return AgentReply(answer=messages[-1].text, tool_trace=traces)
