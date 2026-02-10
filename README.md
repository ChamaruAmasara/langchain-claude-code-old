# langchain-claude-code

**Drop-in replacement for `ChatAnthropic`** that uses your Claude Pro/Max subscription — no API key needed.

Uses the Claude Code CLI under the hood, so if you can run `claude`, you can use this.

```bash
pip install langchain-claude-code
```

## Quick Start

```python
from langchain_claude_code import ChatClaudeCode

# Just like ChatAnthropic, but no API key needed
llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
response = llm.invoke("What is the capital of France?")
print(response.content)
```

## Feature Comparison with ChatAnthropic

| Feature | ChatAnthropic | ChatClaudeCode | Notes |
|---|---|---|---|
| `invoke` | ✅ | ✅ | |
| `stream` | ✅ | ✅ | Real token-by-token streaming |
| `batch` | ✅ | ✅ | Via LangChain base class |
| `ainvoke` / `astream` | ✅ | ✅ | Via LangChain base class |
| Image input | ✅ | ✅ | Base64 + URL |
| Audio input | ❌ | ❌ | |
| Video input | ❌ | ❌ | |
| System messages | ✅ | ✅ | |
| Tool calling (`bind_tools`) | ✅ | ✅ | Via system prompt injection |
| Structured output | ✅ | ✅ | Via `with_structured_output` |
| Extended thinking | ✅ | ✅ | Via `thinking` param |
| Effort levels | ✅ | ✅ | `effort="high"` / `"medium"` / `"low"` |
| Token usage | ✅ | ❌ | CLI doesn't expose per-call usage |
| `stop_sequences` | ✅ | ⚠️ | Param accepted, limited CLI support |
| `temperature` | ✅ | ⚠️ | Param accepted, CLI uses its defaults |
| `top_k` / `top_p` | ✅ | ⚠️ | Param accepted, CLI uses its defaults |
| `max_retries` | ✅ | ❌ | CLI handles retries internally |
| Logprobs | ❌ | ❌ | |
| API key auth | ✅ | N/A | Uses subscription via CLI OAuth |
| Strict tool use | ✅ | ❌ | |
| MCP servers | ❌ | ✅ | Via Claude Code's MCP support |
| Agentic mode | ❌ | ✅ | Built-in filesystem, bash, etc. |
| Tool access control | N/A | ✅ | `allowed_tools` / `disallowed_tools` |
| Computer use | ✅ | ❌ | API-only feature |
| Web search | ✅ | ❌ | API-only feature |

## Prerequisites

- **Claude Code CLI** installed and authenticated: `npm install -g @anthropic-ai/claude-code`
- **Claude Pro or Max subscription**
- **Python 3.10+**
- **Node.js 18+** (required by Claude Code CLI)
- CLI must run in a **TTY** (terminal) — doesn't work when backgrounded

### Platform Support

| Platform | Status | Credential Storage |
|---|---|---|
| **macOS** | ✅ Fully supported | macOS Keychain |
| **Linux** | ✅ Fully supported | `~/.claude/credentials.json` |
| **Windows (WSL)** | ✅ Works in WSL | `~/.claude/credentials.json` |
| **Windows (native)** | ⚠️ Untested | — |

#### Linux Setup

```bash
# 1. Install Node.js 18+ (if not installed)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs

# 2. Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# 3. Authenticate (opens browser for OAuth)
claude auth login

# 4. Install the package
pip install langchain-claude-code
```

> **Note:** On headless Linux servers, `claude auth login` will print a URL to open in your browser. Complete the OAuth flow there, and the CLI will store credentials locally.

## Usage

### Basic Invocation

```python
from langchain_claude_code import ChatClaudeCode

llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
llm.invoke("Hello, Claude!")
```

### System Messages

```python
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
llm.invoke([
    SystemMessage(content="You are a Python expert. Be concise."),
    HumanMessage(content="Write a function to reverse a string."),
])
```

### Streaming

```python
llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
for chunk in llm.stream("Count from 1 to 5"):
    print(chunk.content, end="", flush=True)
```

### Chains

```python
from langchain_core.prompts import ChatPromptTemplate

llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])
chain = prompt | llm
chain.invoke({"input": "Explain OAuth2 briefly"})
```

### Image Input

```python
import base64
from langchain_core.messages import HumanMessage

llm = ChatClaudeCode(model="claude-sonnet-4-20250514")

# From URL
llm.invoke([HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
])])

# From base64
with open("photo.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

llm.invoke([HumanMessage(content=[
    {"type": "text", "text": "Describe this image"},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
])])
```

### Tool Calling (bind_tools)

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"25°C, sunny in {city}"

llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("What's the weather in Tokyo?")
```

### Structured Output

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
structured_llm = llm.with_structured_output(Answer)
result = structured_llm.invoke("What is the capital of France?")
# result.answer == "Paris", result.confidence == 1.0
```

### Extended Thinking

```python
llm = ChatClaudeCode(
    model="claude-sonnet-4-20250514",
    thinking={"type": "enabled", "budget_tokens": 10000},
)
response = llm.invoke("Solve this step by step: what is 127 * 389?")
```

### Effort Levels

```python
# Quick response
llm = ChatClaudeCode(model="claude-sonnet-4-20250514", effort="low")

# Thorough response
llm = ChatClaudeCode(model="claude-sonnet-4-20250514", effort="high")
```

### Agentic Mode (Filesystem, Bash, and more)

By default, `ChatClaudeCode` runs with `max_turns=1` — pure text completion, no tool execution. Increase `max_turns` to unlock Claude Code's built-in tools:

```python
from langchain_claude_code import ChatClaudeCode

# Full agent with filesystem + bash access
agent = ChatClaudeCode(
    model="claude-sonnet-4-20250514",
    max_turns=10,
    permission_mode="bypassPermissions",
    cwd="/path/to/project",
)
response = agent.invoke("Read main.py, find the bug, and fix it")
```

#### Available Built-in Tools

When `max_turns > 1`, Claude Code can use its built-in tools:

| Tool | Description |
|---|---|
| `Read` | Read file contents |
| `Write` | Create or overwrite files |
| `Edit` | Make precise edits to files |
| `Bash` | Run shell commands |
| `Glob` | Find files by pattern |
| `Grep` | Search file contents |
| `LS` | List directory contents |

#### Controlling Tool Access

```python
# Read-only agent (safe for untrusted prompts)
reader = ChatClaudeCode(
    model="claude-sonnet-4-20250514",
    max_turns=5,
    allowed_tools=["Read", "Glob", "Grep", "LS"],
)

# Everything except shell access
no_bash = ChatClaudeCode(
    model="claude-sonnet-4-20250514",
    max_turns=5,
    disallowed_tools=["Bash"],
    permission_mode="bypassPermissions",
)
```

#### Permission Modes

| Mode | Description |
|---|---|
| `default` | Prompts user for permission (interactive only) |
| `acceptEdits` | Auto-accept file edits, prompt for bash |
| `plan` | Read-only, no writes or bash |
| `bypassPermissions` | Auto-accept everything ⚠️ |

> **⚠️ Security Note:** With `max_turns > 1` and `bypassPermissions`, the model has full access to the filesystem and can execute arbitrary shell commands in `cwd`. Only use with trusted prompts. Use `allowed_tools` or `plan` mode to restrict access.

### LangGraph ReAct Agent

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

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
    {"messages": [{"role": "user", "content": "What's the weather in Colombo?"}]}
)
print(response["messages"][-1].content)
```

See [`examples/agent.py`](examples/agent.py) for a full working example.

## API Reference

### `ChatClaudeCode`

#### Core Parameters (ChatAnthropic-compatible)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"claude-sonnet-4-20250514"` | Model ID or alias |
| `max_tokens` | `int` | `4096` | Maximum tokens to generate |
| `temperature` | `float` | `None` | Sampling temperature |
| `top_k` | `int` | `None` | Top-K sampling |
| `top_p` | `float` | `None` | Nucleus sampling |
| `stop_sequences` | `list[str]` | `None` | Stop sequences |
| `streaming` | `bool` | `False` | Stream by default |
| `thinking` | `dict` | `None` | Extended thinking config |
| `effort` | `str` | `None` | `"high"`, `"medium"`, or `"low"` |

#### Claude Code-specific Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | `None` | System prompt override |
| `permission_mode` | `str` | `None` | `default`, `acceptEdits`, `plan`, `bypassPermissions` |
| `max_turns` | `int` | `1` | Max conversation turns. `1` = text-only, `>1` = agentic |
| `cwd` | `str` | `None` | Working directory for CLI and file operations |
| `cli_path` | `str` | `None` | Path to claude binary |
| `allowed_tools` | `list[str]` | `None` | Whitelist of tools (e.g. `["Read", "Glob"]`) |
| `disallowed_tools` | `list[str]` | `None` | Blacklist of tools (e.g. `["Bash", "Write"]`) |

## How It Works

Claude Code CLI stores OAuth tokens in the system credential store (macOS Keychain, or `~/.claude/credentials.json` on Linux). These tokens are **restricted to the Claude Code CLI** — they return:

> *"This credential is only authorized for use with Claude Code and cannot be used for other API requests."*

This package works by shelling out to `claude` via `claude-code-sdk`, which handles all authentication. The tradeoff is subprocess overhead per call, but it's the only way to use subscription-based inference programmatically.

## Migration from ChatAnthropic

```python
# Before (requires API key)
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key="sk-ant-...")

# After (uses your subscription)
from langchain_claude_code import ChatClaudeCode
llm = ChatClaudeCode(model="claude-sonnet-4-20250514")

# Everything else stays the same:
llm.invoke("Hello!")
llm.stream("Count to 5")
llm.bind_tools([my_tool])
llm.with_structured_output(MySchema)
prompt | llm | parser  # chains work identically
```

## License

MIT
