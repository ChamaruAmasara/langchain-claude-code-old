"""
ChatClaudeCode — LangChain BaseChatModel backed by Claude Code CLI.

Uses the Claude Code CLI as the inference backend via claude-code-sdk.
No API key needed; uses your Claude Pro/Max subscription.

Note: Claude Code OAuth tokens are restricted to the Claude Code CLI —
they cannot be used for direct API calls to the Anthropic Messages API.
This is why inference goes through the CLI subprocess.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import (
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
            api_msgs.append({"role": "user", "content": str(msg.content)})

    return system, api_msgs


class ChatClaudeCode(BaseChatModel):
    """LangChain ChatModel using Claude Code CLI — no API key needed.

    Uses the Claude Code CLI subprocess via ``claude-code-sdk`` to run inference
    with your Claude Pro/Max subscription. The CLI handles all authentication
    (OAuth tokens stored in the system keychain).

    Requirements:
      - ``claude`` CLI installed and authenticated (``npm install -g @anthropic-ai/claude-code``)
      - ``claude-code-sdk`` Python package (``pip install claude-code-sdk``)

    Examples:
        .. code-block:: python

            from langchain_claude_code import ChatClaudeCode

            llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
            llm.invoke("Hello, Claude!")

            # With system message
            from langchain_core.messages import HumanMessage, SystemMessage
            llm.invoke([
                SystemMessage(content="You are helpful."),
                HumanMessage(content="What is OAuth2?"),
            ])

            # With chains
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

    system_prompt: Optional[str] = None
    """System prompt override."""

    permission_mode: Optional[str] = None
    """Permission mode: default, acceptEdits, bypassPermissions."""

    cli_path: Optional[str] = None
    """Path to claude CLI binary. Auto-detected if None."""

    max_turns: Optional[int] = None
    """Maximum conversation turns. Defaults to 1 for single-shot."""

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response via Claude Code CLI subprocess."""
        try:
            from claude_code_sdk import ClaudeCodeOptions
            from claude_code_sdk import query as claude_query
        except ImportError:
            raise ImportError(
                "claude-code-sdk is required. Install with: pip install claude-code-sdk"
            )

        system, api_messages = _convert_messages(messages)

        # Build prompt from messages
        if len(api_messages) == 1:
            prompt = api_messages[0]["content"]
        else:
            parts = []
            for msg in api_messages:
                role = msg["role"].capitalize()
                parts.append(f"{role}: {msg['content']}")
            prompt = "\n\n".join(parts)

        options = ClaudeCodeOptions(
            model=self.model,
            system_prompt=system or self.system_prompt,
            max_turns=self.max_turns or 1,
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

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream — currently falls back to full generate."""
        result = self._generate(messages, stop, run_manager, **kwargs)
        text = result.generations[0].message.content
        yield ChatGenerationChunk(message=AIMessageChunk(content=str(text)))
