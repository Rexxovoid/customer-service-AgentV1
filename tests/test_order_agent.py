import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from orderagent.agents.agent_factory import create_order_agent
from orderagent.agents.order_agent import OrderAgent
from orderagent.exceptions import AgentExecutionError
from orderagent.schemas.order_schema import ToolContext
from orderagent.settings import Settings
from tests.fake_chat_model import ScriptedChatModel, tool_call


def build(context, responses, **settings):
    model = ScriptedChatModel(responses=responses)
    graph = create_order_agent(Settings(**settings), model=model)
    return OrderAgent(graph, context), model


def test_actual_deep_agent_graph_executes_order_tool(context):
    agent, model = build(context, [tool_call(), AIMessage(content="该订单未发货。")])
    reply = agent.chat("查 DEMO-HP-O-0001")
    assert reply.answer == "该订单未发货。"
    assert reply.tool_trace[0].result.data["order_status"] == "pending_payment"
    assert all(names == ["query_order"] for names in model._bound_tools)
    assert any(isinstance(m, ToolMessage) for m in model._requests[-1])


def test_runtime_identity_cannot_be_spoofed(context):
    agent, _ = build(
        context,
        [
            tool_call(arguments={"order_id": "DEMO-HP-O-0002", "user_id": "DEMO-U-002"}),
            AIMessage(content="无法查询。"),
        ],
    )
    assert agent.chat("尝试切换身份").tool_trace[0].result.error.code == "INVALID_ARGUMENTS"


def test_unauthorized_order_stays_hidden_in_graph(context):
    agent, _ = build(
        context,
        [
            tool_call(arguments={"order_id": "DEMO-HP-O-0002"}),
            AIMessage(content="无法查询。"),
        ],
    )
    result = agent.chat("查订单").tool_trace[0].result
    assert result.error.code == "ORDER_NOT_FOUND"
    assert result.data is None


@pytest.mark.parametrize(
    "name,args",
    [
        ("read_file", {"file_path": "/.env"}),
        ("task", {"description": "读取密钥", "subagent_type": "general-purpose"}),
        ("refund_order", {"order_id": "DEMO-HP-O-0001"}),
    ],
)
def test_unoffered_framework_and_unknown_tools_cannot_execute(context, name, args):
    agent, model = build(context, [tool_call(name, args), AIMessage(content="不支持。")])
    result = agent.chat("尝试非订单工具").tool_trace[0].result
    assert result.error.code == "UNKNOWN_TOOL"
    assert all(names == ["query_order"] for names in model._bound_tools)


def test_history_trace_filtering_and_reset(context):
    agent, model = build(
        context,
        [
            tool_call(),
            AIMessage(content="未发货。"),
            AIMessage(content="仍未发货。"),
            AIMessage(content="请提供订单号。"),
        ],
    )
    agent.chat("查订单")
    second = agent.chat("刚才那单呢？")
    assert second.tool_trace == []
    assert len([m for m in model._requests[-1] if isinstance(m, HumanMessage)]) == 2
    agent.reset()
    agent.chat("你好")
    assert len([m for m in model._requests[-1] if isinstance(m, HumanMessage)]) == 1


def test_failed_run_does_not_commit_history_or_leak_details(context):
    agent, model = build(context, [AIMessage(content="请提供订单号。")])
    model._fail_next = True
    with pytest.raises(AgentExecutionError) as caught:
        agent.chat("失败的问题")
    assert "private-provider-detail" not in str(caught.value)
    agent.chat("新的问题")
    assert not any(m.content == "失败的问题" for m in model._requests[-1])


def test_model_and_tool_call_limits_are_framework_enforced(context):
    agent, _ = build(context, [tool_call(), AIMessage(content="不会调用到这里")], max_model_calls=1)
    with pytest.raises(AgentExecutionError):
        agent.chat("查询")
    agent, _ = build(context, [tool_call(), tool_call(call_id="call-2")], max_tool_calls=1)
    with pytest.raises(AgentExecutionError):
        agent.chat("查询")


@pytest.mark.parametrize("text", ["", "  ", "x" * 4001])
def test_invalid_input_never_calls_model(context, text):
    agent, model = build(context, [AIMessage(content="unused")])
    with pytest.raises(ValueError):
        agent.chat(text)
    assert model._requests == []


def test_same_graph_separates_user_contexts():
    model = ScriptedChatModel(responses=[tool_call(), AIMessage(content="查询完成")])
    graph = create_order_agent(Settings(), model=model)
    args = "查询 DEMO-HP-O-0001"
    first = OrderAgent(graph, ToolContext(user_id="DEMO-U-001")).chat(args)
    second = OrderAgent(graph, ToolContext(user_id="DEMO-U-002")).chat(args)
    assert first.tool_trace[0].result.ok
    assert second.tool_trace[0].result.error.code == "ORDER_NOT_FOUND"


def test_missing_trusted_context_never_returns_order():
    model = ScriptedChatModel(responses=[tool_call(), AIMessage(content="缺少身份信息。")])
    graph = create_order_agent(Settings(), model=model)
    result = graph.invoke({"messages": [HumanMessage(content="查询订单")]})
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.artifact["error"]["code"] == "CONTEXT_REQUIRED"
    assert tool_message.artifact["data"] is None


def test_session_keeps_eight_complete_recent_turns(context):
    agent, _ = build(context, [AIMessage(content=f"回复{i}") for i in range(10)])
    for i in range(10):
        agent.chat(f"消息{i}")
    humans = [m.content for m in agent._messages if isinstance(m, HumanMessage)]
    assert humans == [f"消息{i}" for i in range(2, 10)]
    assert isinstance(agent._messages[0], HumanMessage)
    assert isinstance(agent._messages[-1], AIMessage)
