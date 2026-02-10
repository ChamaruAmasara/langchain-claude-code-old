# langchain-claude-code

Use your **Claude Pro/Max subscription** as a LangChain ChatModel — no API key needed.

Uses the Claude Code CLI under the hood, so if you can run `claude`, you can use this.

## Installation

```bash
pip install langchain-claude-code
```

### Prerequisites

- Claude Code CLI installed and authenticated: `npm install -g @anthropic-ai/claude-code`
- A Claude Pro or Max subscription
- Python 3.10+

## Quick Start

```python
from langchain_claude_code import ChatClaudeCode

llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
response = llm.invoke("What is the capital of France?")
print(response.content)
```

### With LangChain Chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_claude_code import ChatClaudeCode

llm = ChatClaudeCode(model="claude-sonnet-4-20250514", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

chain = prompt | llm
response = chain.invoke({"input": "Explain OAuth2 in 2 sentences"})
print(response.content)
```

### System Messages

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_claude_code import ChatClaudeCode

llm = ChatClaudeCode(model="claude-sonnet-4-20250514")

messages = [
    SystemMessage(content="You are a Python expert. Be concise."),
    HumanMessage(content="Write a function to reverse a string."),
]

response = llm.invoke(messages)
print(response.content)
```

## How It Works

Claude Code CLI stores OAuth tokens in the system keychain (macOS Keychain, etc.) with scopes like `user:inference`. However, **these tokens are restricted to the Claude Code CLI** — they cannot be used for direct API calls to `api.anthropic.com`.

This package works by shelling out to the `claude` CLI via `claude-code-sdk`, which handles all authentication internally. The tradeoff is subprocess overhead, but it's the only way to use subscription-based inference programmatically.

## API Reference

### `ChatClaudeCode`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"claude-sonnet-4-20250514"` | Anthropic model ID |
| `max_tokens` | `int` | `4096` | Maximum tokens to generate |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `system_prompt` | `str` | `None` | System prompt override |
| `permission_mode` | `str` | `None` | `default`, `acceptEdits`, `bypassPermissions` |
| `max_turns` | `int` | `1` | Max conversation turns |

## LangGraph Agent (create_react_agent)

Build a ReAct agent with tools — powered by your Claude subscription:

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_claude_code import ChatClaudeCode

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"25°C, sunny in {city}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

llm = ChatClaudeCode(model="claude-sonnet-4-20250514", max_turns=5)
agent = create_react_agent(model=llm, tools=[get_weather, calculate])

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]}
)
print(response["messages"][-1].content)
```

See [`examples/agent.py`](examples/agent.py) for a full working example.

## Streaming

```python
from langchain_claude_code import ChatClaudeCode

llm = ChatClaudeCode(model="claude-sonnet-4-20250514")

for chunk in llm.stream("Count from 1 to 5"):
    print(chunk.content, end="", flush=True)
```

## Image Input

```python
import base64
from langchain_core.messages import HumanMessage
from langchain_claude_code import ChatClaudeCode

llm = ChatClaudeCode(model="claude-sonnet-4-20250514")

# From base64
with open("photo.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = llm.invoke([HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
])])

# From URL
response = llm.invoke([HumanMessage(content=[
    {"type": "text", "text": "Describe this"},
    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
])])
```

## Why Not Direct API?

Claude Code's OAuth tokens (`sk-ant-oat01-*`) have `user:inference` scope but are server-side restricted:

```
"This credential is only authorized for use with Claude Code
and cannot be used for other API requests."
```

The only way to use subscription-based inference is through the CLI.

## License

MIT
