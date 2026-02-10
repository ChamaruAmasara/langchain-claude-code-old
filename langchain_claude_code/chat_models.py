"""
ChatClaudeCode — LangChain BaseChatModel backed by Claude Code authentication.

Supports two inference backends:
  1. "api" (default) — Direct Anthropic Messages API with OAuth Bearer tokens.
     No API key needed; uses your Claude Pro/Max subscription.
  2. "cli" — Shells out to Claude Code CLI via claude-code-sdk.
     Requires `claude` CLI installed and authenticated.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Iterator, List, Optional, Union

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from langchain_claude_code.oauth import ClaudeOAuthManager


def _convert_messages(
    messages: List[BaseMessage],
) -> tuple[Optional[str], list[dict]]:
    """Convert LangChain messages → Anthropic API format (system, messages)."""
    system: Optional[str] = None
    api_msgs: list[dict] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system = str(msg.content)
        elif isinstance(msg, HumanMessage):
            api_msgs.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            api_msgs.append({"role": "assistant", "content": str(msg.content)})
        else:
            # Fallback — treat unknown as user
            api_msgs.append({"role": "user", "content": str(msg.content)})

    return system, api_msgs


class ChatClaudeCode(BaseChatModel):
    """LangChain ChatModel using Claude Code authentication.

    Use your Claude Pro/Max subscription directly — no API key needed.

    Two backends:
      - ``backend="api"`` (default): Direct API calls with OAuth Bearer tokens.
        First run opens a browser for authorization; tokens are cached and auto-refreshed.
      - ``backend="cli"``: Uses the Claude Code CLI subprocess via ``claude-code-sdk``.
        Requires ``claude`` CLI installed and already authenticated.

    Examples:
        .. code-block:: python

            from langchain_claude_code import ChatClaudeCode

            # API backend (default) — opens browser on first use
            llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
            llm.invoke("Hello, Claude!")

            # CLI backend — uses claude CLI subprocess
            llm = ChatClaudeCode(backend="cli", model="claude-sonnet-4-20250514")
            llm.invoke("Hello, Claude!")

            # Works with chains
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are helpful."),
                ("human", "{input}"),
            ])
            chain = prompt | llm
            chain.invoke({"input": "Explain OAuth2 briefly"})
    """

    model: str = "claude-sonnet-4-20250514"
    """Anthropic model ID."""

    max_tokens: int = 4096
    """Maximum tokens to generate."""

    temperature: float = 0.0
    """Sampling temperature."""

    backend: str = "api"
    """Inference backend: "api" (OAuth + direct API) or "cli" (Claude Code CLI)."""

    base_url: str = "https://api.anthropic.com"
    """Anthropic API base URL (only used with api backend)."""

    oauth_manager: Optional[ClaudeOAuthManager] = None
    """Custom OAuth manager (only used with api backend). Auto-created if None."""

    streaming: bool = False
    """Whether to stream by default."""

    # CLI backend options
    cli_path: Optional[str] = None
    """Path to claude CLI binary (cli backend only)."""

    system_prompt: Optional[str] = None
    """System prompt override for CLI backend."""

    permission_mode: Optional[str] = None
    """Permission mode for CLI backend (default, acceptEdits, bypassPermissions)."""

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _get_oauth_manager(self) -> ClaudeOAuthManager:
        if self.oauth_manager is None:
            self.oauth_manager = ClaudeOAuthManager()
        return self.oauth_manager

    # ── API Backend ──────────────────────────────────────────────

    def _api_generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
    ) -> ChatResult:
        """Generate via direct Anthropic API with OAuth Bearer token."""
        token = self._get_oauth_manager().get_access_token()
        system, api_messages = _convert_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": api_messages,
        }
        if system:
            payload["system"] = system
        if stop:
            payload["stop_sequences"] = stop

        resp = httpx.post(
            f"{self.base_url}/v1/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=120,
        )

        if resp.status_code == 401:
            # Token might have expired between check and use — force refresh
            self._get_oauth_manager()._tokens = None
            token = self._get_oauth_manager().get_access_token()
            resp = httpx.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
                timeout=120,
            )

        if resp.status_code != 200:
            raise RuntimeError(f"Anthropic API error ({resp.status_code}): {resp.text}")

        data = resp.json()
        text = "".join(
            block["text"] for block in data.get("content", []) if block["type"] == "text"
        )

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=text),
                    generation_info={
                        "model": data.get("model"),
                        "usage": data.get("usage"),
                        "stop_reason": data.get("stop_reason"),
                    },
                )
            ]
        )

    def _api_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream via direct Anthropic API with OAuth Bearer token."""
        token = self._get_oauth_manager().get_access_token()
        system, api_messages = _convert_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": api_messages,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if stop:
            payload["stop_sequences"] = stop

        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Anthropic API error ({resp.status_code}): {resp.read().decode()}"
                )
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data["type"] == "content_block_delta":
                    text = data["delta"].get("text", "")
                    if text:
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=text)
                        )
                        if run_manager:
                            run_manager.on_llm_new_token(text)
                        yield chunk

    # ── CLI Backend ──────────────────────────────────────────────

    def _cli_generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
    ) -> ChatResult:
        """Generate via Claude Code CLI subprocess."""
        try:
            from claude_code_sdk import ClaudeCodeOptions, query as claude_query
        except ImportError:
            raise ImportError(
                "CLI backend requires claude-code-sdk. "
                "Install with: pip install 'langchain-claude-code[cli]'"
            )

        system, api_messages = _convert_messages(messages)

        # Build prompt from messages
        if len(api_messages) == 1:
            prompt = api_messages[0]["content"]
        else:
            # Multi-turn: format as conversation
            parts = []
            for msg in api_messages:
                role = msg["role"].capitalize()
                parts.append(f"{role}: {msg['content']}")
            prompt = "\n\n".join(parts)

        options = ClaudeCodeOptions(
            model=self.model,
            system_prompt=system or self.system_prompt,
            max_turns=1,
        )

        if self.permission_mode:
            options.permission_mode = self.permission_mode  # type: ignore

        # Run async query synchronously
        text_parts: list[str] = []

        async def _run() -> None:
            async for msg in claude_query(prompt=prompt, options=options):
                if hasattr(msg, "content"):
                    for block in msg.content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(lambda: asyncio.run(_run())).result()
        else:
            asyncio.run(_run())

        text = "".join(text_parts)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=text),
                    generation_info={"model": self.model, "backend": "cli"},
                )
            ]
        )

    # ── LangChain Interface ──────────────────────────────────────

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.backend == "cli":
            return self._cli_generate(messages, stop)
        return self._api_generate(messages, stop)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.backend == "cli":
            # CLI backend doesn't support streaming yet — fall back to full generate
            result = self._cli_generate(messages, stop)
            text = result.generations[0].message.content
            yield ChatGenerationChunk(message=AIMessageChunk(content=str(text)))
            return
        yield from self._api_stream(messages, stop, run_manager)
