import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from orderagent.agents.agent_factory import create_order_agent
from orderagent.integrations.handoff import HandoffResponse, MockHandoffGateway
from orderagent.repositories.order_repository import DataLoadError
from orderagent.repositories.support_repository import JsonSupportRepository
from orderagent.schemas.order_schema import ToolContext
from orderagent.schemas.support_schema import HandoffQuery, SupportQuery
from orderagent.settings import ConfigurationError, Settings
from orderagent.tools.support_tools import create_support_tools
from tests.fake_chat_model import ScriptedChatModel, tool_call


class RecordingGateway:
    def __init__(self, *, fail=False, raises=False):
        self.requests = []
        self.fail = fail
        self.raises = raises

    def transfer(self, request):
        self.requests.append(request)
        if self.raises:
            raise RuntimeError("private-platform-key")
        return HandoffResponse(
            request_id=request.request_id,
            status="error" if self.fail else "connected",
            simulated=True,
            agent_id=None if self.fail else "001",
        )


@pytest.fixture
def repository():
    settings = Settings()
    return JsonSupportRepository([settings.support_path])


def build(responses, *, gateway=None, **settings):
    model = ScriptedChatModel(responses=responses)
    graph = create_order_agent(Settings(**settings), model=model, handoff_gateway=gateway)
    return graph, model


def ask(graph, text, thread="quality", *, context=ToolContext(user_id="DEMO-U-001")):
    return graph.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread}},
        context=context,
    )


def quality_call(issue="左耳没有声音", **kwargs):
    return tool_call("query_quality_support", {"category": "headphones", "issue": issue, **kwargs})


def test_quality_advice_is_source_based_and_program_rendered():
    graph, model = build([quality_call(), AIMessage(content="不应调用的第二套排查方案")])
    state = ask(graph, "我的耳机左耳没有声音了")
    assert state["support_phase"] == "advised"
    assert state["support_advice"]["id"] == "DEMO-HP-K-0001"
    assert "左右声道平衡" in state["messages"][-1].text
    assert "模拟排查案例" in state["messages"][-1].text
    assert len(model._requests) == 1
    message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert message.artifact["data"]["action"] == "advice"


@pytest.mark.parametrize("feedback", ["没有解决", "还是没声音", "试了没用", "方案无效"])
def test_unsuccessful_attempt_hands_off_without_another_model_call(feedback):
    gateway = RecordingGateway()
    graph, model = build([quality_call()], gateway=gateway)
    ask(graph, "我的耳机左耳没有声音了")
    state = ask(graph, feedback)
    assert state["support_phase"] == "transferred"
    assert state["messages"][-1].text == "【模拟客服】对接人工客服成功，由工号001为你服务。"
    assert len(model._requests) == 1
    assert len(gateway.requests) == 1
    request = gateway.requests[0]
    assert request.user_id == "DEMO-U-001"
    assert request.issue == "左耳没有声音"
    assert request.attempted_solution["id"] == "DEMO-HP-K-0001"


def test_unrecognized_feedback_can_be_routed_by_model_to_handoff():
    gateway = RecordingGateway()
    graph, model = build(
        [
            quality_call(),
            tool_call("handoff_to_human", {"reason": "用户反馈照做后依然无法正常收听"}, "h-2"),
        ],
        gateway=gateway,
    )
    ask(graph, "耳机左耳没有声音")
    state = ask(graph, "照着做完了，故障依旧")
    assert state["support_phase"] == "transferred"
    assert len(model._requests) == 2
    assert len(gateway.requests) == 1


@pytest.mark.parametrize(
    "scenario,phase,expected",
    [
        ("success", "transferred", "对接人工客服成功，由工号001"),
        ("error", "transfer_failed", "转人工错误"),
    ],
)
def test_no_knowledge_uses_configured_mock(scenario, phase, expected):
    graph, model = build([quality_call("不明故障XYZ")], mock_handoff_scenario=scenario)
    state = ask(graph, "耳机不明故障XYZ")
    assert state["support_phase"] == phase
    assert expected in state["messages"][-1].text
    assert "模拟客服" in state["messages"][-1].text
    assert "support_advice" not in state
    assert len(model._requests) == 1


@pytest.mark.parametrize("fail,raises", [(True, False), (False, True)])
def test_failure_never_claims_success_or_retries(fail, raises):
    gateway = RecordingGateway(fail=fail, raises=raises)
    graph, model = build([quality_call()], gateway=gateway)
    ask(graph, "耳机左耳没有声音")
    state = ask(graph, "没有解决")
    assert state["support_phase"] == "transfer_failed"
    assert "转人工错误" in state["messages"][-1].text
    assert "成功" not in state["messages"][-1].text
    assert "private-platform-key" not in str(state)
    again = ask(graph, "转人工")
    assert again["support_handoff"] == state["support_handoff"]
    assert len(gateway.requests) == 1
    assert len(model._requests) == 1


def test_successful_handoff_is_idempotent():
    gateway = RecordingGateway()
    graph, _ = build([quality_call()], gateway=gateway)
    ask(graph, "耳机左耳没有声音")
    first = ask(graph, "没有解决")
    second = ask(graph, "再转人工客服")
    assert first["support_handoff"] == second["support_handoff"]
    assert len(gateway.requests) == 1


def test_second_quality_lookup_cannot_provide_second_solution():
    gateway = RecordingGateway()
    graph, model = build([quality_call(), quality_call("换个方法解决")], gateway=gateway)
    ask(graph, "耳机左耳没有声音")
    state = ask(graph, "还有别的方法吗")
    assert state["support_phase"] == "transferred"
    assert len(gateway.requests) == 1
    assert "可先进行" not in state["messages"][-1].text
    assert len(model._requests) == 2


def test_after_handoff_quality_tool_stays_closed():
    gateway = RecordingGateway()
    graph, _ = build([quality_call("陌生问题"), quality_call()], gateway=gateway)
    ask(graph, "耳机有陌生问题")
    state = ask(graph, "再检查左耳没有声音")
    assert state["support_phase"] == "transferred"
    assert "support_advice" not in state
    assert len(gateway.requests) == 1


def test_not_tried_and_resolved_feedback_do_not_transfer():
    gateway = RecordingGateway()
    graph, model = build(
        [
            quality_call(),
            AIMessage(content="可以先试刚才那一次方案。"),
            AIMessage(content="好的。"),
        ],
        gateway=gateway,
    )
    ask(graph, "耳机左耳没有声音")
    ask(graph, "还没用这个办法")
    state = ask(graph, "已经解决了")
    assert state["support_phase"] == "advised"
    assert not gateway.requests
    assert len(model._requests) == 3


def test_explicit_handoff_needs_no_model_and_passes_issue():
    gateway = RecordingGateway()
    graph, model = build([AIMessage(content="不应调用")], gateway=gateway)
    state = ask(graph, "转人工")
    assert state["support_phase"] == "transferred"
    assert gateway.requests[0].issue == "转人工"
    assert not model._requests


def test_missing_context_never_contacts_gateway():
    gateway = RecordingGateway()
    graph, _ = build([quality_call("无匹配")], gateway=gateway)
    state = ask(graph, "耳机无匹配问题", context=None)
    assert state["support_phase"] == "transfer_failed"
    assert "缺少可信用户" in state["messages"][-1].text
    assert not gateway.requests


def test_new_thread_and_checkpoint_deletion_reset_attempt():
    graph, _ = build([quality_call()])
    ask(graph, "耳机左耳没有声音", thread="old")
    state = ask(graph, "耳机左耳没有声音", thread="fresh")
    assert state["support_phase"] == "advised"
    graph.checkpointer.delete_thread("old")
    state = ask(graph, "耳机左耳没有声音", thread="old")
    assert state["support_phase"] == "advised"
    assert len([m for m in state["messages"] if isinstance(m, ToolMessage)]) == 1


def test_runtime_identity_and_result_cannot_be_supplied_by_model():
    graph, _ = build(
        [
            tool_call(
                "handoff_to_human", {"reason": "故障", "user_id": "other", "status": "connected"}
            ),
            AIMessage(content="参数错误。"),
        ]
    )
    state = ask(graph, "耳机坏了")
    result = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert result.artifact["error"]["code"] == "INVALID_ARGUMENTS"
    assert "support_handoff" not in state


def test_parallel_quality_calls_are_rejected():
    first = quality_call()
    second = quality_call()
    second.tool_calls[0]["id"] = "parallel-2"
    first.tool_calls.extend(second.tool_calls)
    graph, _ = build([first, AIMessage(content="请逐个处理。")])
    state = ask(graph, "耳机左耳没有声音")
    assert "support_advice" not in state
    results = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(results) == 2
    assert all(m.artifact["error"]["code"] == "INVALID_ARGUMENTS" for m in results)


def test_model_schemas_hide_identity_and_mock_controls(repository):
    tools = create_support_tools(repository, MockHandoffGateway())
    assert set(tools[0].tool_call_schema.model_json_schema()["properties"]) == {
        "category",
        "issue",
        "environment",
        "order_id",
    }
    assert set(tools[1].tool_call_schema.model_json_schema()["properties"]) == {
        "reason",
        "topic",
        "request_human",
    }


def test_environment_constraints_and_specific_case_priority(repository):
    specific = repository.find(
        SupportQuery(category="headphones", issue="左耳没有声音", environment="AirPods iPhone")
    )
    assert specific.id == "DEMO-HP-K-0002"
    assert specific.data_origin == "source_grounded_summary"
    assert (
        repository.find(SupportQuery(category="mice", issue="USB鼠标没有反应", environment="macOS"))
        is None
    )
    assert (
        repository.find(
            SupportQuery(category="mice", issue="USB鼠标没有反应", environment="Windows 11")
        ).id
        == "DEMO-MS-K-0001"
    )


@pytest.mark.parametrize(
    "category,issue",
    [
        ("mice", "无线鼠标没反应"),
        ("keyboards", "无线键盘没反应"),
        ("headphones", "右耳没声音"),
    ],
)
def test_added_cases_cover_three_categories(repository, category, issue):
    assert repository.find(SupportQuery(category=category, issue=issue)) is not None


@pytest.mark.parametrize("raw", [{}, [{"id": "bad"}], None])
def test_invalid_knowledge_fails_safely(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataLoadError, match="故障知识"):
        JsonSupportRepository([path])


def test_duplicate_knowledge_is_rejected():
    path = Settings().support_path
    with pytest.raises(DataLoadError):
        JsonSupportRepository([path, path])


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"category": "other", "issue": "x"},
        {"category": "headphones", "issue": " "},
        {"category": "mice", "issue": "x", "environment": "x", "user_id": "other"},
    ],
)
def test_support_parameters_are_strict(arguments):
    with pytest.raises(ValidationError):
        SupportQuery.model_validate(arguments)
    with pytest.raises(ValidationError):
        HandoffQuery(reason=" ")


def test_mock_scenario_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("MOCK_HANDOFF_SCENARIO", raising=False)
    env = tmp_path / ".env"
    env.write_text("MOCK_HANDOFF_SCENARIO=error\n", encoding="utf-8")
    assert Settings.load(env).mock_handoff_scenario == "error"
    monkeypatch.setenv("MOCK_HANDOFF_SCENARIO", "success")
    assert Settings.load(env).mock_handoff_scenario == "success"
    monkeypatch.setenv("MOCK_HANDOFF_SCENARIO", "invalid")
    with pytest.raises(ConfigurationError):
        Settings.load(env)


@pytest.mark.parametrize(
    "bad_result",
    [
        {"status": "connected", "simulated": False},
        {"request_id": "wrong-id", "status": "connected", "simulated": False, "agent_id": "999"},
    ],
)
def test_invalid_platform_response_is_a_failure(bad_result):
    class InvalidGateway:
        def transfer(self, request):
            return bad_result

    graph, _ = build([quality_call("无适用方案")], gateway=InvalidGateway())
    state = ask(graph, "耳机无适用方案")
    assert state["support_phase"] == "transfer_failed"
    assert "转人工错误" in state["messages"][-1].text
    assert "成功" not in state["messages"][-1].text


def test_handoff_requests_are_isolated_by_user_thread():
    gateway = RecordingGateway()
    graph, _ = build([quality_call("无适用方案")], gateway=gateway)
    for user in ("DEMO-U-001", "DEMO-U-002"):
        state = ask(graph, "耳机无适用方案", thread=user, context=ToolContext(user_id=user))
        assert state["support_phase"] == "transferred"
    assert {r.user_id for r in gateway.requests} == {"DEMO-U-001", "DEMO-U-002"}
    assert len({r.request_id for r in gateway.requests}) == 2
