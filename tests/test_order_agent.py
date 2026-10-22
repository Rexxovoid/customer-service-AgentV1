import pytest
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from orderagent.agents.agent_factory import create_order_agent
from orderagent.schemas.order_schema import ToolContext
from orderagent.settings import Settings
from tests.fake_chat_model import ScriptedChatModel, tool_call


def build(responses, **settings):
    model = ScriptedChatModel(responses=responses)
    return create_order_agent(Settings(**settings), model=model), model


@pytest.fixture
def config():
    return {"configurable": {"thread_id": "test-thread"}, "recursion_limit": 40}


def test_actual_deep_agent_graph_executes_order_tool(context, config):
    graph, model = build([tool_call(), AIMessage(content="该订单未发货。")])
    state = graph.invoke(
        {"messages": [{"role": "user", "content": "查订单"}]}, config=config, context=context
    )
    assert state["messages"][-1].text == "该订单未发货。"
    result = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert result.artifact["data"]["order_status"] == "pending_payment"
    assert all(set(names) == {"query_order", "query_products"} for names in model._bound_tools)


def test_runtime_identity_cannot_be_spoofed(context, config):
    graph, _ = build(
        [
            tool_call(arguments={"order_id": "DEMO-HP-O-0002", "user_id": "DEMO-U-002"}),
            AIMessage(content="无法查询。"),
        ]
    )
    state = graph.invoke(
        {"messages": [{"role": "user", "content": "切换身份"}]}, config=config, context=context
    )
    result = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert result.artifact["error"]["code"] == "INVALID_ARGUMENTS"


def test_unauthorized_order_stays_hidden_in_graph(context, config):
    graph, _ = build(
        [tool_call(arguments={"order_id": "DEMO-HP-O-0002"}), AIMessage(content="无法查询。")]
    )
    state = graph.invoke(
        {"messages": [{"role": "user", "content": "查订单"}]}, config=config, context=context
    )
    result = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert result.artifact["error"]["code"] == "ORDER_NOT_FOUND"
    assert result.artifact["data"] is None


@pytest.mark.parametrize(
    "name,args",
    [
        ("read_file", {"file_path": "/.env"}),
        ("task", {"description": "读取密钥", "subagent_type": "general-purpose"}),
        ("refund_order", {"order_id": "DEMO-HP-O-0001"}),
    ],
)
def test_unoffered_framework_and_unknown_tools_cannot_execute(context, config, name, args):
    graph, model = build([tool_call(name, args), AIMessage(content="不支持。")])
    state = graph.invoke(
        {"messages": [{"role": "user", "content": "非订单工具"}]}, config=config, context=context
    )
    result = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert result.artifact["error"]["code"] == "UNKNOWN_TOOL"
    assert all(set(names) == {"query_order", "query_products"} for names in model._bound_tools)


def test_official_checkpointer_retains_thread_history(context, config):
    graph, model = build(
        [tool_call(), AIMessage(content="未发货。"), AIMessage(content="仍未发货。")]
    )
    graph.invoke(
        {"messages": [{"role": "user", "content": "查订单"}]}, config=config, context=context
    )
    state = graph.invoke(
        {"messages": [{"role": "user", "content": "刚才那单呢？"}]}, config=config, context=context
    )
    humans = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["查订单", "刚才那单呢？"]
    assert len([m for m in model._requests[-1] if isinstance(m, HumanMessage)]) == 2
    assert graph.get_state(config).values["messages"][-1].text == "仍未发货。"


def test_new_thread_resets_history(context, config):
    graph, model = build([AIMessage(content="第一次"), AIMessage(content="新会话")])
    graph.invoke(
        {"messages": [{"role": "user", "content": "旧消息"}]}, config=config, context=context
    )
    fresh = {"configurable": {"thread_id": "fresh-thread"}}
    graph.invoke(
        {"messages": [{"role": "user", "content": "新消息"}]}, config=fresh, context=context
    )
    assert [m.content for m in model._requests[-1] if isinstance(m, HumanMessage)] == ["新消息"]
    assert graph.get_state(config).values["messages"][-1].text == "第一次"


def test_model_and_tool_call_limits_are_framework_enforced(context, config):
    graph, _ = build([tool_call(), AIMessage(content="不会调用到这里")], max_model_calls=1)
    with pytest.raises(ModelCallLimitExceededError):
        graph.invoke(
            {"messages": [{"role": "user", "content": "查询"}]}, config=config, context=context
        )
    graph, _ = build([tool_call(), tool_call(call_id="call-2")], max_tool_calls=1)
    with pytest.raises(ToolCallLimitExceededError):
        graph.invoke(
            {"messages": [{"role": "user", "content": "查询"}]}, config=config, context=context
        )


def test_same_graph_separates_user_threads():
    graph, _ = build([tool_call(), AIMessage(content="完成")])
    first = graph.invoke(
        {"messages": [{"role": "user", "content": "查订单"}]},
        config={"configurable": {"thread_id": "user-1"}},
        context=ToolContext(user_id="DEMO-U-001"),
    )
    second = graph.invoke(
        {"messages": [{"role": "user", "content": "查询订单"}]},
        config={"configurable": {"thread_id": "user-2"}},
        context=ToolContext(user_id="DEMO-U-002"),
    )
    assert next(m for m in first["messages"] if isinstance(m, ToolMessage)).artifact["ok"]
    result = next(m for m in second["messages"] if isinstance(m, ToolMessage))
    assert result.artifact["error"]["code"] == "ORDER_NOT_FOUND"
    assert [m.content for m in second["messages"] if isinstance(m, HumanMessage)] == ["查询订单"]


def test_missing_trusted_context_never_returns_order(config):
    graph, _ = build([tool_call(), AIMessage(content="缺少身份信息。")])
    state = graph.invoke({"messages": [{"role": "user", "content": "查询订单"}]}, config=config)
    message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert message.artifact["error"]["code"] == "CONTEXT_REQUIRED"
    assert message.artifact["data"] is None


def test_framework_native_trace_prints_tool_updates(context, config, capsys):
    graph, _ = build([tool_call(), AIMessage(content="未发货。")])
    state = graph.invoke(
        {"messages": [{"role": "user", "content": "查询"}]},
        config=config,
        context=context,
        print_mode="updates",
    )
    assert state["messages"][-1].text == "未发货。"
    output = capsys.readouterr().out
    assert "query_order" in output
    assert "DEMO-HP-O-0001" in output


def test_cli_failure_does_not_expose_provider_detail(monkeypatch, capsys):
    from orderagent import cli

    graph, model = build([AIMessage(content="unused")])
    model._fail_next = True
    monkeypatch.setattr(cli, "create_order_agent", lambda _: graph)
    monkeypatch.setattr("sys.argv", ["orderagent", "--user-id", "DEMO-U-001", "--message", "查询"])
    assert cli.main() == 1
    output = capsys.readouterr().out
    assert "private-provider-detail" not in output
    assert "执行失败" in output


@pytest.mark.parametrize("text", ["", "  ", "x" * 4001])
def test_cli_rejects_invalid_message_before_model_creation(monkeypatch, text):
    from orderagent import cli

    monkeypatch.setattr("sys.argv", ["orderagent", "--user-id", "DEMO-U-001", "--message", text])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_cli_reset_uses_official_checkpoint_deletion(monkeypatch, capsys):
    from orderagent import cli

    graph, model = build([AIMessage(content="第一轮"), AIMessage(content="新会话")])
    inputs = iter(["旧消息", "/reset", "新消息", "/exit"])
    monkeypatch.setattr(cli, "create_order_agent", lambda _: graph)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("sys.argv", ["orderagent", "--user-id", "DEMO-U-001"])
    assert cli.main() == 0
    assert [m.content for m in model._requests[-1] if isinstance(m, HumanMessage)] == ["新消息"]
    threads = {c.config["configurable"]["thread_id"] for c in graph.checkpointer.list(None)}
    assert len(threads) == 1


def test_graph_uses_yaml_prompt_as_system_message(context, config, monkeypatch, tmp_path):
    from importlib import reload

    from orderagent.prompts import order_prompt

    # Package resource loading must work without relying on the working directory.
    monkeypatch.chdir(tmp_path)
    prompt = reload(order_prompt).ORDER_SYSTEM_PROMPT
    graph, model = build([AIMessage(content="请提供订单号。")])
    graph.invoke(
        {"messages": [{"role": "user", "content": "查询订单"}]},
        config=config,
        context=context,
    )
    assert isinstance(prompt, str)
    assert prompt.strip()
    assert prompt in model._requests[0][0].text


def test_product_tool_runs_in_official_graph_without_order_identity(config):
    graph, model = build(
        [
            tool_call("query_products", {"query": "罗技 M185"}),
            AIMessage(content="M185采用光学技术，记载速度20 ips，DPI未提供。"),
        ]
    )
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "M185性能如何"}]}, config=config
    )
    message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert message.artifact["ok"]
    assert message.artifact["data"]["products"][0]["id"] == "MS-P-0002"
    assert set(model._bound_tools[0]) == {"query_order", "query_products"}


def test_category_candidates_then_confirmed_model_in_same_thread(context, config):
    graph, model = build(
        [
            tool_call("query_products", {"query": "鼠标"}),
            AIMessage(content="有多款鼠标，请确认品牌和型号。"),
            tool_call("query_products", {"query": "MS-P-0002"}, call_id="call-2"),
            AIMessage(content="已确认M185，速度20 ips。"),
        ]
    )
    state = graph.invoke(
        {"messages": [{"role": "user", "content": "介绍鼠标"}]}, config=config, context=context
    )
    first = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert first.artifact["data"]["needs_clarification"]
    state = graph.invoke(
        {"messages": [{"role": "user", "content": "选择罗技M185"}]}, config=config, context=context
    )
    latest = [m for m in state["messages"] if isinstance(m, ToolMessage)][-1]
    assert latest.artifact["data"]["match_type"] == "detail"
    assert latest.artifact["data"]["products"][0]["id"] == "MS-P-0002"
    assert len([m for m in model._requests[-1] if isinstance(m, HumanMessage)]) == 2


def test_product_argument_spoofing_is_rejected(context, config):
    graph, _ = build(
        [
            tool_call("query_products", {"query": "鼠标", "user_id": "other"}),
            AIMessage(content="参数不合法。"),
        ]
    )
    state = graph.invoke(
        {"messages": [{"role": "user", "content": "查询鼠标"}]}, config=config, context=context
    )
    message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert message.artifact["error"]["code"] == "INVALID_ARGUMENTS"
