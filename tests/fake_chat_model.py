from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import PrivateAttr


class ScriptedChatModel(FakeMessagesListChatModel):
    _bound_tools: list[list[str]] = PrivateAttr(default_factory=list)
    _requests: list[list] = PrivateAttr(default_factory=list)
    _fail_next: bool = PrivateAttr(default=False)

    def bind_tools(self, tools, **kwargs):
        self._bound_tools.append([t.name for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._requests.append(list(messages))
        if self._fail_next:
            self._fail_next = False
            raise RuntimeError("private-provider-detail")
        return super()._generate(messages, stop, run_manager, **kwargs)


def tool_call(name="query_order", arguments=None, call_id="call-1"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": call_id,
                "name": name,
                "args": arguments if arguments is not None else {"order_id": "DEMO-HP-O-0001"},
                "type": "tool_call",
            }
        ],
    )
