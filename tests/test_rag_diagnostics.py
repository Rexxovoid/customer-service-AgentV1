from types import SimpleNamespace

import pytest
import requests
from langchain_core.runnables import RunnableLambda
from openai import APITimeoutError, LengthFinishReasonError
from openai.types.chat import ChatCompletion
from pydantic import SecretStr, ValidationError

from orderagent.agents.agent_factory import create_order_agent
from orderagent.evaluation.runner import Collector
from orderagent.integrations.ragflow import RagflowKnowledge
from orderagent.models import create_deepseek_model
from orderagent.repositories.support_repository import JsonSupportRepository
from orderagent.schemas.support_schema import SupportQuery
from orderagent.settings import Settings
from tests.fake_chat_model import ScriptedChatModel
from tests.test_ragflow_after_sales import FakeRagClient, Selector
from tests.test_return_tools import ask, call
from tests.test_return_tools import graph_builder as graph_builder
from tests.test_return_tools import order_rows as order_rows
from tests.test_session_policy import quality
from tests.test_support_tools import ask as ask_quality


class FailingClient(FakeRagClient):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def retrieve(self, **kwargs):
        raise self.error


@pytest.mark.parametrize("mode", ["disabled", "saved", "limit_reached", "write_failed"])
def test_local_evidence_capture_is_opt_in_and_never_changes_rejection(tmp_path, monkeypatch, mode):
    import json

    monkeypatch.setattr("orderagent.integrations.ragflow.PROJECT_ROOT", tmp_path)
    debug_dir = tmp_path / "evals" / "debug" / "rag_evidence"
    if mode == "limit_reached":
        debug_dir.mkdir(parents=True)
        for i in range(20):
            (debug_dir / f"evidence-{i}.json").write_text("{}", encoding="utf-8")
    if mode == "write_failed":
        (tmp_path / "evals").write_text("blocked", encoding="utf-8")
    repo = repository(
        selector=Selector(
            {
                "applicable": True,
                "chunk_id": "chunk",
                "paragraph_start": 1,
                "paragraph_end": 99,
                "handling": "clarify",
                "missing_information": ["condition"],
            }
        )
    )
    repo.evidence_debug = mode != "disabled"
    repo.debug_secrets = ["private-test-key"]
    collector = Collector()
    with pytest.raises(ValueError, match="RAG_EVIDENCE_INVALID"):
        RunnableLambda(repo.find).invoke(
            SupportQuery(category="keyboards", issue="private-user-question"),
            config={"callbacks": [collector]},
        )
    diagnostic = next(
        e["data"] for e in collector.events if e["data"]["stage"] == "selection_rejected"
    )
    assert diagnostic["debug_status"] == mode
    if mode == "saved":
        body = (debug_dir / diagnostic["debug_record"]).read_text(encoding="utf-8")
        payload = json.loads(body)
        assert payload["candidate_text"]
        assert payload["paragraph_end"] == 99
        assert payload["extracted_excerpt"] == ""
        assert "--- candidate" in payload["diff"]
        assert "private-test-key" not in body
        assert "private-user-question" in body  # 调用现场现在显式包含实际选择输入。
        assert "candidate_text" not in diagnostic
    elif mode == "disabled":
        assert not debug_dir.exists()


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"chunk_id": "unknown"}, ["UNKNOWN_CHUNK"]),
        ({"paragraph_start": 0}, ["INVALID_PARAGRAPH_RANGE"]),
        ({"paragraph_end": 99}, ["INVALID_PARAGRAPH_RANGE"]),
        ({"paragraph_start": 2, "paragraph_end": 1}, ["INVALID_PARAGRAPH_RANGE"]),
        (
            {"handling": "clarify", "missing_information": []},
            ["MISSING_CLARIFICATION_FIELDS"],
        ),
    ],
)
def test_evidence_rejection_records_specific_reasons(changes, expected):
    selection = {
        "applicable": True,
        "chunk_id": "chunk",
        "paragraph_start": 1,
        "paragraph_end": 1,
        "handling": "basic_troubleshooting",
        **changes,
    }
    collector = Collector()
    with pytest.raises(ValueError, match="RAG_EVIDENCE_INVALID"):
        RunnableLambda(repository(selector=Selector(selection)).find).invoke(
            SupportQuery(category="headphones", issue="左耳无声"),
            config={"callbacks": [collector]},
        )
    diagnostic = next(
        e["data"] for e in collector.events if e["data"]["stage"] == "selection_rejected"
    )
    assert diagnostic["failure_reasons"] == expected
    assert diagnostic["excerpt_length"] >= 0
    assert diagnostic["excerpt_limit"] == 2500
    assert "excerpt" not in diagnostic


def test_evidence_failure_is_available_in_native_custom_stream():
    from langgraph.graph import END, START, StateGraph

    def node(state):
        with pytest.raises(ValueError, match="RAG_EVIDENCE_INVALID"):
            repository(
                selector=Selector(
                    {
                        "applicable": True,
                        "chunk_id": "unknown",
                        "paragraph_start": 1,
                        "paragraph_end": 1,
                        "handling": "return_eligible",
                    }
                )
            ).find(SupportQuery(category="keyboards", issue="退货"))
        return state

    graph = StateGraph(dict).add_node("check", node)
    graph.add_edge(START, "check")
    graph.add_edge("check", END)
    events = list(graph.compile().stream({}, stream_mode="custom"))
    rejected = [e for e in events if e["rag_diagnostic"]["stage"] == "selection_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["rag_diagnostic"]["failure_reasons"] == ["UNKNOWN_CHUNK"]


class FailingSelector:
    def __init__(self, error):
        self.error = error

    def invoke(self, messages):
        raise self.error


@pytest.mark.parametrize("with_response", [True, False])
def test_selection_failure_saves_input_before_call_and_available_response(
    tmp_path, monkeypatch, with_response
):
    import json

    monkeypatch.setattr("orderagent.integrations.ragflow.PROJECT_ROOT", tmp_path)
    debug_dir = tmp_path / "evals" / "debug" / "rag_evidence"
    completion = ChatCompletion(
        id="test",
        created=0,
        model="test",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": '{"applicable":',
                    "reasoning_content": "private-reasoning-not-for-log",
                },
            }
        ],
        usage={"prompt_tokens": 50, "completion_tokens": 99, "total_tokens": 149},
    )

    class InspectSelector:
        def invoke(self, messages):
            paths = list(debug_dir.glob("evidence-*.json"))
            assert len(paths) == 1
            before = json.loads(paths[0].read_text(encoding="utf-8"))
            assert before["phase"] == "before_call"
            assert before["messages"] == [list(m) for m in messages]
            assert before["effective_max_tokens"] == 4096
            if with_response:
                raise LengthFinishReasonError(completion=completion)
            raise APITimeoutError(request=SimpleNamespace())

    repo = repository(selector=InspectSelector())
    repo.evidence_debug = True
    repo.selection_budget = 4096
    with pytest.raises(
        ValueError, match="RAG_SELECTION_TRUNCATED" if with_response else "RAG_SELECTION_TIMEOUT"
    ):
        repo.find(SupportQuery(category="keyboards", issue="退货"))
    paths = list(debug_dir.glob("evidence-*.json"))
    assert len(paths) == 1
    body = paths[0].read_text(encoding="utf-8")
    after = json.loads(body)
    assert after["phase"] == "exception"
    assert "private-reasoning-not-for-log" not in body
    if with_response:
        assert after["finish_reasons"] == ["length"]
        assert after["token_usage"]["completion_tokens"] == 99
        assert after["partial_response"][0]["content"] == '{"applicable":'
    else:
        assert after["token_usage"] is None
        assert after["finish_reasons"] is None
        assert after["response_availability"] == "unavailable_from_exception"


def test_successful_selection_also_saves_structured_result(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr("orderagent.integrations.ragflow.PROJECT_ROOT", tmp_path)
    repo = repository()
    repo.evidence_debug = True
    repo.find(SupportQuery(category="headphones", issue="左耳无声"))
    record = next((tmp_path / "evals/debug/rag_evidence").glob("evidence-*.json"))
    result = json.loads(record.read_text(encoding="utf-8"))
    assert result["phase"] == "returned"
    assert result["structured_result"]["chunk_id"] == "chunk"
    assert result["response_availability"] == "structured_only_raw_unavailable"


def repository(client=None, selector=None):
    result = object.__new__(RagflowKnowledge)
    result.client = client if client is not None else FakeRagClient()
    result.selector = (
        selector
        if selector is not None
        else Selector(
            {
                "applicable": True,
                "chunk_id": "chunk",
                "paragraph_start": 1,
                "paragraph_end": 1,
                "handling": "basic_troubleshooting",
                "customer_reply": "如果支持声道平衡，请恢复居中。",
            }
        )
    )
    result.dataset_name = "售后规则知识库"
    result.assistant_name = "售后助手"
    return result


@pytest.mark.parametrize(
    "error,code",
    [
        (requests.ConnectionError("private-provider-message"), "RAG_CONNECTION_ERROR"),
        (requests.Timeout("private-provider-message"), "RAG_RETRIEVAL_TIMEOUT"),
        (RuntimeError("private-provider-message"), "RAG_RETRIEVAL_ERROR"),
    ],
)
def test_retrieval_failure_code_does_not_expose_vendor_body(error, code):
    collector = Collector()
    with pytest.raises(ValueError, match=code) as raised:
        RunnableLambda(repository(client=FailingClient(error)).find).invoke(
            SupportQuery(category="mice", issue="左键没反应"), config={"callbacks": [collector]}
        )
    assert str(raised.value) == code
    events = [e["data"] for e in collector.events if e["data"]["stage"] == "rag_exception"]
    assert events[0]["error_code"] == code
    assert "private-provider-message" not in str(events)


@pytest.mark.parametrize(
    "status,code", [(401, "RAG_AUTH_ERROR"), (403, "RAG_AUTH_ERROR"), (500, "RAG_HTTP_ERROR")]
)
def test_http_status_has_separate_classification(status, code):
    response = requests.Response()
    response.status_code = status
    with pytest.raises(ValueError, match=code):
        repository(client=FailingClient(requests.HTTPError(response=response))).find(
            SupportQuery(category="mice", issue="左键没反应")
        )


@pytest.mark.parametrize(
    "error,code",
    [
        (
            LengthFinishReasonError(
                completion=ChatCompletion(
                    id="test", created=0, model="test", object="chat.completion", choices=[]
                )
            ),
            "RAG_SELECTION_TRUNCATED",
        ),
        (APITimeoutError(request=SimpleNamespace()), "RAG_SELECTION_TIMEOUT"),
        (ValueError("invalid-json-private-provider-message"), "RAG_SELECTION_ERROR"),
    ],
)
def test_selection_failures_are_not_retrieval_misses(error, code):
    with pytest.raises(ValueError, match=code):
        repository(selector=FailingSelector(error)).find(
            SupportQuery(category="headphones", issue="左耳无声")
        )


def test_no_candidates_and_no_applicable_rule_are_different():
    client = FakeRagClient()
    client.retrieve = lambda **kwargs: []
    with pytest.raises(ValueError, match="RAG_NO_CANDIDATES"):
        repository(client=client).find(SupportQuery(category="mice", issue="左键没反应"))
    with pytest.raises(ValueError, match="RAG_NO_APPLICABLE_RULE"):
        repository(
            selector=Selector(
                {
                    "applicable": False,
                    "chunk_id": "",
                    "paragraph_start": 0,
                    "paragraph_end": 0,
                    "handling": "human",
                }
            )
        ).find(SupportQuery(category="mice", issue="左键没反应"))


def test_selection_budget_does_not_mutate_main_model(monkeypatch):
    settings = Settings(api_key=SecretStr("fake-test-key"), ragflow_selection_max_tokens=6144)
    model = create_deepseek_model(settings)
    observed = []

    def structured(self, schema, *, method, include_raw):
        observed.append((self.max_tokens, method))
        assert include_raw is True
        assert self.reasoning_effort == "low"
        assert self.extra_body["thinking"]["type"] == "enabled"
        return Selector({})

    monkeypatch.setattr(type(model), "with_structured_output", structured)
    RagflowKnowledge(settings, model)
    assert model.max_tokens == 1200
    assert model.reasoning_effort is None
    assert not model.extra_body
    assert observed == [(6144, "json_mode")]


def test_unverified_provider_is_not_silently_given_low_effort():
    from orderagent.settings import ConfigurationError

    settings = Settings(api_key=SecretStr("test"), base_url="https://example.com")
    with pytest.raises(ConfigurationError, match="DeepSeek"):
        RagflowKnowledge(settings, create_deepseek_model(settings))


def test_native_raw_response_metrics_are_recorded_without_private_reasoning():
    from langchain_core.messages import AIMessage

    selected = {
        "applicable": True,
        "chunk_id": "chunk",
        "paragraph_start": 1,
        "paragraph_end": 1,
        "handling": "basic_troubleshooting",
        "customer_reply": "如果支持声道平衡，请恢复居中。",
    }
    repo = repository()
    repo.selector = SimpleNamespace(
        invoke=lambda messages: {
            "raw": AIMessage(
                content="{}",
                additional_kwargs={"reasoning_content": "private"},
                usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                response_metadata={"finish_reason": "stop", "model_name": "test"},
            ),
            "parsed": selected,
            "parsing_error": None,
        }
    )
    collector = Collector()
    RunnableLambda(repo.find).invoke(
        SupportQuery(category="headphones", issue="左耳无声"), config={"callbacks": [collector]}
    )
    metrics = next(e["data"] for e in collector.events if e["data"]["stage"] == "selection_metrics")
    assert metrics["usage"]["output_tokens"] == 20
    assert "private" not in str(metrics)


@pytest.mark.parametrize("failure", ["json_format", "field_type", "enum_value", "success"])
@pytest.mark.parametrize("storage", ["saved", "disabled", "limit_reached", "write_failed"])
def test_native_parser_body_and_error_are_saved_locally(tmp_path, monkeypatch, failure, storage):
    import json

    from langchain_core.messages import AIMessage
    from langchain_core.output_parsers import PydanticOutputParser

    from orderagent.integrations.ragflow import RuleSelection

    monkeypatch.setattr("orderagent.integrations.ragflow.PROJECT_ROOT", tmp_path)
    debug_dir = tmp_path / "evals/debug/rag_evidence"
    if storage == "limit_reached":
        debug_dir.mkdir(parents=True)
        for index in range(20):
            (debug_dir / f"evidence-{index}.json").write_text("{}", encoding="utf-8")
    elif storage == "write_failed":
        (tmp_path / "evals").write_text("blocked", encoding="utf-8")
    payload = {
        "applicable": True,
        "chunk_id": "chunk",
        "paragraph_start": 1,
        "paragraph_end": 1,
        "handling": "basic_troubleshooting",
        "customer_reply": "如果支持声道平衡，请恢复居中。",
    }
    if failure == "field_type":
        payload["missing_information"] = 123
    elif failure == "enum_value":
        payload["handling"] = "invalid-value"
    content = (
        '{"applicable":not-json}'
        if failure == "json_format"
        else json.dumps(payload, ensure_ascii=False)
    )
    parsed = None
    error = None
    try:
        parsed = PydanticOutputParser(pydantic_object=RuleSelection).parse(content)
    except Exception as exc:
        error = exc
    raw = AIMessage(
        content=content,
        additional_kwargs={"reasoning_content": "hidden-do-not-save"},
        response_metadata={"finish_reason": "stop", "model_name": "test"},
    )
    repo = repository(
        selector=SimpleNamespace(
            invoke=lambda messages: {"raw": raw, "parsed": parsed, "parsing_error": error}
        )
    )
    repo.evidence_debug = storage != "disabled"
    if error:
        with pytest.raises(ValueError, match="RAG_SELECTION_ERROR"):
            repo.find(SupportQuery(category="headphones", issue="左耳无声"))
    else:
        repo.find(SupportQuery(category="headphones", issue="左耳无声"))
    if storage != "saved":
        if storage == "limit_reached":
            assert len(list(debug_dir.glob("evidence-*.json"))) == 20
            assert all(
                p.read_text(encoding="utf-8") == "{}" for p in debug_dir.glob("evidence-*.json")
            )
        else:
            assert not debug_dir.exists()
        return
    record = next((tmp_path / "evals/debug/rag_evidence").glob("evidence-*.json"))
    body = record.read_text(encoding="utf-8")
    saved = json.loads(body)
    assert saved["raw_content"] == content
    assert "hidden-do-not-save" not in body
    if error:
        assert failure in saved["parsing_error"]["categories"]
        assert saved["parsing_error"]["detail"]
        assert saved["parsing_error"]["type"] == "OutputParserException"
        assert saved["structured_result"] is None
    else:
        assert saved["parsing_error"] is None
        assert saved["structured_result"]["handling"] == "basic_troubleshooting"


def test_inconsistent_return_decision_is_rejected():
    repo = repository(
        selector=Selector(
            {
                "applicable": True,
                "chunk_id": "chunk",
                "paragraph_start": 1,
                "paragraph_end": 1,
                "handling": "return_eligible",
                "missing_information": ["签收日期"],
                "decision": "missing_information",
            }
        )
    )
    with pytest.raises(ValueError, match="RAG_EVIDENCE_INVALID"):
        repo.find(SupportQuery(category="headphones", issue="退货"))


@pytest.mark.parametrize("source_break", ["\r\n", "\r", "\n"])
def test_evidence_matching_normalizes_line_endings_only(source_break):
    client = FakeRagClient()
    client.retrieve = lambda **kwargs: [
        SimpleNamespace(
            id="chunk", document_name="test", content=f"条件A{source_break}{source_break}步骤B"
        )
    ]
    repo = repository(
        client=client,
        selector=Selector(
            {
                "applicable": True,
                "chunk_id": "chunk",
                "paragraph_start": 1,
                "paragraph_end": 2,
                "handling": "basic_troubleshooting",
                "customer_reply": "满足条件A时执行步骤B。",
            }
        ),
    )
    collector = Collector()
    answer = RunnableLambda(repo.find).invoke(
        SupportQuery(category="headphones", issue="左耳无声"), config={"callbacks": [collector]}
    )
    assert answer.answer == client.retrieve()[0].content
    diagnostic = next(
        e["data"] for e in collector.events if e["data"]["stage"] == "evidence_checked"
    )
    assert diagnostic["normalized_excerpt_match"] is True


def test_search_description_is_not_presented_as_user_fact_to_selector():
    import json

    repo = repository()
    original = repo.client.retrieve
    requests = []
    repo.client.retrieve = lambda **kwargs: (requests.append(kwargs), original(**kwargs))[1]
    selected = repo.selector.selection

    def invoke(messages):
        data = json.loads(messages[1][1])
        assert "search_description" not in data["query"]
        return selected

    repo.selector = SimpleNamespace(invoke=invoke)
    repo.find(
        SupportQuery(category="headphones", issue="左耳无声", search_description="耳机单侧声音异常")
    )
    assert "耳机单侧声音异常" in requests[0]["question"]


@pytest.mark.parametrize("start,end", [(0, 1), (-1, 1), (2, 1), (1, 3)])
def test_invalid_paragraph_ranges_are_rejected(start, end):
    repo = repository(
        selector=Selector(
            {
                "applicable": True,
                "chunk_id": "chunk",
                "paragraph_start": start,
                "paragraph_end": end,
                "handling": "clarify",
                "missing_information": ["system"],
            }
        )
    )
    with pytest.raises(ValueError, match="RAG_EVIDENCE_INVALID"):
        repo.find(SupportQuery(category="headphones", issue="test"))


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_range_extraction_keeps_intermediate_conditions_and_exact_text(newline):
    import json

    # E51 regression: the model no longer copies/pastes conditions + steps while skipping P2.
    source = (newline * 2).join(
        [
            "Conditions: Windows USB mouse.",
            "Required: Windows and USB; no model number required.",
            "Steps: reconnect; try another USB port.",
            "Limits: no hardware repair guarantee.",
        ]
    )
    client = FakeRagClient()
    client.retrieve = lambda **kwargs: [
        SimpleNamespace(id="chunk", document_name="mouse", content=source)
    ]

    def select(messages):
        paragraphs = json.loads(messages[1][1])["evidence"]["chunk"]["paragraphs"]
        assert [p["id"] for p in paragraphs] == [1, 2, 3, 4]
        return {
            "applicable": True,
            "chunk_id": "chunk",
            "paragraph_start": 1,
            "paragraph_end": 4,
            "handling": "basic_troubleshooting",
            "customer_reply": "Reconnect the USB receiver.",
        }

    repo = repository(client=client, selector=SimpleNamespace(invoke=select))
    events = []
    repo._diagnostic = events.append
    answer = repo.find(SupportQuery(category="mice", issue="left button not responding"))
    assert answer.answer == source
    event = next(e for e in events if e["stage"] == "evidence_checked")
    assert event["extracted_excerpt"] == source
    assert event["paragraph_start"] == 1 and event["paragraph_end"] == 4


@pytest.mark.parametrize("value", [True, "1", 1.5])
def test_range_numbers_are_strict_integers(value):
    from orderagent.integrations.ragflow import RuleSelection

    with pytest.raises(ValidationError):
        RuleSelection(
            applicable=True,
            chunk_id="chunk",
            paragraph_start=value,
            paragraph_end=1,
            handling="clarify",
        )


def test_extracted_evidence_length_limit_is_preserved():
    client = FakeRagClient()
    client.retrieve = lambda **kwargs: [
        SimpleNamespace(id="chunk", document_name="test", content="x" * 2501)
    ]
    repo = repository(client=client)
    events = []
    repo._diagnostic = events.append
    with pytest.raises(ValueError, match="RAG_EVIDENCE_INVALID"):
        repo.find(SupportQuery(category="mice", issue="test"))
    assert next(e for e in events if e["stage"] == "selection_rejected")["failure_reasons"] == [
        "EXCERPT_TOO_LONG"
    ]


@pytest.mark.parametrize("value", [0, 1023, 16385])
def test_selection_budget_is_bounded(value):
    with pytest.raises(ValidationError):
        Settings(ragflow_selection_max_tokens=value)


@pytest.mark.parametrize(
    "category,issue",
    [
        ("mice", "我鼠标左键没反应是为什么"),
        ("mice", "左键点不动"),
        ("keyboards", "我的键盘空格没有反应怎么办"),
        ("keyboards", "spacebar not working"),
    ],
)
def test_synonyms_match_but_unknown_environment_only_clarifies(category, issue):
    knowledge = JsonSupportRepository([Settings().support_path]).find(
        SupportQuery(category=category, issue=issue)
    )
    assert knowledge is not None
    assert knowledge.source["handling"] == "clarify"
    assert knowledge.missing_information


@pytest.mark.parametrize("category,issue", [("mice", "左键没有反应"), ("keyboards", "空格键失灵")])
def test_known_and_partially_known_environment_do_not_get_confused(category, issue):
    source = JsonSupportRepository([Settings().support_path])
    partial = source.find(SupportQuery(category=category, issue=issue, environment="Windows 11"))
    assert partial is not None and partial.source["handling"] == "clarify"
    complete = source.find(
        SupportQuery(category=category, issue=issue, environment="Windows 11 USB")
    )
    assert complete is not None and complete.source["handling"] == "basic_troubleshooting"
    incompatible = source.find(
        SupportQuery(category=category, issue=issue, environment="macOS ROCCAT USB")
    )
    assert incompatible is None


def test_quality_handoff_includes_classified_reason(tmp_path):
    graph = create_order_agent(
        Settings(after_sales_db_path=tmp_path / "test.sqlite3"),
        model=ScriptedChatModel(responses=[quality()]),
        support_repository=repository(
            client=FailingClient(requests.Timeout("private-provider-message"))
        ),
    )
    state = ask_quality(graph, "耳机左耳没有声音")
    assert state["support_error_code"] == "RAG_RETRIEVAL_TIMEOUT"
    assert "RAG_RETRIEVAL_TIMEOUT" in state["support_handoff"]["reason"]
    assert "private-provider-message" not in str(state)


def test_return_handoff_includes_classified_reason(graph_builder):
    graph, _ = graph_builder(
        [call()], knowledge=repository(client=FailingClient(requests.ConnectionError()))
    )
    state = ask(graph, "我要退货")
    assert state["return_error_code"] == "RAG_CONNECTION_ERROR"
    assert state["return_result"]["code"] == "RAG_CONNECTION_ERROR"
    assert "RAG_CONNECTION_ERROR" in state["support_handoff"]["reason"]
    repeated = ask(graph, "还是同一个退货问题")
    assert repeated["return_result"]["code"] == "RAG_CONNECTION_ERROR"
