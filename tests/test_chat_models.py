"""Unit tests for ChatClaudeCode."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_claude_code.chat_models import ChatClaudeCode, _convert_messages


def test_convert_messages_basic():
    msgs = [
        SystemMessage(content="Be helpful."),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
        HumanMessage(content="How are you?"),
    ]
    system, api_msgs = _convert_messages(msgs)
    assert system == "Be helpful."
    assert len(api_msgs) == 3
    assert api_msgs[0] == {"role": "user", "content": "Hello"}
    assert api_msgs[1] == {"role": "assistant", "content": "Hi there"}
    assert api_msgs[2] == {"role": "user", "content": "How are you?"}


def test_convert_messages_no_system():
    msgs = [HumanMessage(content="Hello")]
    system, api_msgs = _convert_messages(msgs)
    assert system is None
    assert len(api_msgs) == 1


def test_llm_type():
    llm = ChatClaudeCode()
    assert llm._llm_type == "claude-code"


def test_identifying_params():
    llm = ChatClaudeCode(model="claude-opus-4-20250514", temperature=0.5)
    params = llm._identifying_params
    assert params["model"] == "claude-opus-4-20250514"
    assert params["temperature"] == 0.5
