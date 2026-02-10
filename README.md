# langchain-claude-code

Use your **Claude Pro/Max subscription** as a LangChain ChatModel — no API key needed.

This package provides `ChatClaudeCode`, a drop-in replacement for `ChatAnthropic` that authenticates via OAuth (the same way Claude Code CLI does).

## Installation

```bash
pip install langchain-claude-code

# With CLI backend support
pip install 'langchain-claude-code[cli]'
```

## Quick Start

### API Backend (default)

Uses OAuth Bearer tokens to call the Anthropic Messages API directly. First run opens your browser for authentication; tokens are cached and auto-refreshed.

```python
from langchain_claude_code import ChatClaudeCode

llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
response = llm.invoke("What is the capital of France?")
print(response.content)
```

### CLI Backend

Uses the Claude Code CLI as a subprocess. Requires `claude` CLI installed and authenticated.

```python
from langchain_claude_code import ChatClaudeCode

llm = ChatClaudeCode(backend="cli", model="claude-sonnet-4-20250514")
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

### Streaming

```python
from langchain_claude_code import ChatClaudeCode

llm = ChatClaudeCode(model="claude-sonnet-4-20250514")

for chunk in llm.stream("Count from 1 to 5"):
    print(chunk.content, end="", flush=True)
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

### API Backend (`backend="api"`)

1. On first use, opens your browser to `console.anthropic.com` for OAuth authorization
2. You log in with your Claude Pro/Max account and grant access
3. Tokens are cached at `~/.langchain-claude-code/auth.json` and auto-refreshed
4. All API calls use `Authorization: Bearer <token>` instead of `x-api-key`

This uses the same OAuth client ID as Claude Code CLI (`9d1c250a-e61b-44d9-88ed-5944d1962f5e`), so if you're already logged into Claude Code, you can reuse those tokens.

### CLI Backend (`backend="cli"`)

Shells out to the `claude` CLI via `claude-code-sdk`. The CLI handles all authentication (including OAuth token management). This is simpler but adds subprocess overhead.

## API Reference

### `ChatClaudeCode`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"claude-sonnet-4-20250514"` | Anthropic model ID |
| `max_tokens` | `int` | `4096` | Maximum tokens to generate |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `backend` | `str` | `"api"` | `"api"` or `"cli"` |
| `base_url` | `str` | `"https://api.anthropic.com"` | API base URL |
| `oauth_manager` | `ClaudeOAuthManager` | `None` | Custom OAuth manager |
| `system_prompt` | `str` | `None` | System prompt (CLI backend) |
| `permission_mode` | `str` | `None` | CLI permission mode |

### `ClaudeOAuthManager`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `client_id` | `str` | Claude Code's ID | OAuth client ID |
| `redirect_uri` | `str` | `http://127.0.0.1:8912/callback` | Redirect URI |
| `token_file` | `Path` | `~/.langchain-claude-code/auth.json` | Token cache path |

## Requirements

- Python 3.10+
- A Claude Pro or Max subscription
- For CLI backend: `claude` CLI installed (`npm install -g @anthropic-ai/claude-code`)

## License

MIT
