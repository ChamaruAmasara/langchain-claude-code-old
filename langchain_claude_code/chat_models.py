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
import base64
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, List, Optional, Union

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


def _content_to_anthropic_blocks(content: Union[str, list]) -> Union[str, list[dict]]:
    """Convert LangChain message content to Anthropic content blocks.

    Handles:
      - Plain strings → returned as-is
      - List of dicts with type "text" → text blocks
      - List of dicts with type "image_url" → image blocks (base64 or URL)
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content)

    blocks: list[dict] = []
    for item in content:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            item_type = item.get("type", "")

            if item_type == "text":
                blocks.append({"type": "text", "text": item.get("text", "")})

            elif item_type == "image_url":
                image_url = item.get("image_url", {})
                if isinstance(image_url, str):
                    url = image_url
                else:
                    url = image_url.get("url", "")

                if url.startswith("data:"):
                    # data:image/png;base64,iVBOR...
                    header, b64data = url.split(",", 1)
                    media_type = header.split(":")[1].split(";")[0]
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64data,
                        },
                    })
                else:
                    # Regular URL
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": url,
                        },
                    })

            elif item_type == "image":
                # Direct Anthropic-style image block (passthrough)
                blocks.append(item)
            else:
                blocks.append({"type": "text", "text": str(item)})
        else:
            blocks.append({"type": "text", "text": str(item)})

    return blocks


def _convert_messages(
    messages: List[BaseMessage],
) -> tuple[Optional[str], list[dict], bool]:
    """Convert LangChain messages → Anthropic API format.

    Returns (system, messages, has_multimodal).
    """
    system: Optional[str] = None
    api_msgs: list[dict] = []
    has_multimodal = False

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system = str(msg.content)
        elif isinstance(msg, HumanMessage):
            content = _content_to_anthropic_blocks(msg.content)
            if isinstance(content, list):
                has_multimodal = True
            api_msgs.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            api_msgs.append({"role": "assistant", "content": str(msg.content)})
        else:
            api_msgs.append({"role": "user", "content": str(msg.content)})

    return system, api_msgs, has_multimodal


def _build_prompt_string(api_messages: list[dict]) -> str:
    """Build a plain text prompt from messages (text-only fallback)."""
    if len(api_messages) == 1:
        content = api_messages[0]["content"]
        return content if isinstance(content, str) else str(content)

    parts = []
    for msg in api_messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        if isinstance(content, str):
            parts.append(f"{role}: {content}")
        else:
            # Extract text from blocks
            texts = [b["text"] for b in content if b.get("type") == "text"]
            parts.append(f"{role}: {' '.join(texts)}")
    return "\n\n".join(parts)


class ChatClaudeCode(BaseChatModel):
    """LangChain ChatModel using Claude Code CLI — no API key needed.

    Uses the Claude Code CLI subprocess via ``claude-code-sdk`` to run inference
    with your Claude Pro/Max subscription. The CLI handles all authentication
    (OAuth tokens stored in the system keychain).

    Requirements:
      - ``claude`` CLI installed and authenticated
      - ``claude-code-sdk`` Python package

    Supports multimodal input (images) via LangChain's standard format:

    Examples:
        .. code-block:: python

            from langchain_claude_code import ChatClaudeCode
            from langchain_core.messages import HumanMessage

            llm = ChatClaudeCode(model="claude-sonnet-4-20250514")

            # Text only
            llm.invoke("Hello!")

            # With image (base64)
            import base64
            with open("image.png", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            llm.invoke([HumanMessage(content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}"
                }},
            ])])

            # With image URL
            llm.invoke([HumanMessage(content=[
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {
                    "url": "https://example.com/photo.jpg"
                }},
            ])])
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

        system, api_messages, has_multimodal = _convert_messages(messages)

        options = ClaudeCodeOptions(
            model=self.model,
            system_prompt=system or self.system_prompt,
            max_turns=self.max_turns or 1,
        )

        if self.permission_mode:
            options.permission_mode = self.permission_mode  # type: ignore

        text_parts: list[str] = []

        if has_multimodal:
            # Use streaming mode to send multimodal content
            async def _run_multimodal() -> None:
                async def _input_stream() -> AsyncIterator[dict[str, Any]]:
                    for msg in api_messages:
                        yield {
                            "type": "user",
                            "message": {"role": msg["role"], "content": msg["content"]},
                        }

                async for resp in claude_query(
                    prompt=_input_stream(), options=options
                ):
                    if hasattr(resp, "content"):
                        for block in resp.content:
                            if hasattr(block, "text"):
                                text_parts.append(block.text)

            self._run_async(_run_multimodal())
        else:
            # Simple string prompt for text-only
            prompt = _build_prompt_string(api_messages)

            async def _run_text() -> None:
                async for msg in claude_query(prompt=prompt, options=options):
                    if hasattr(msg, "content"):
                        for block in msg.content:
                            if hasattr(block, "text"):
                                text_parts.append(block.text)

            self._run_async(_run_text())

        text = "".join(text_parts)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=text),
                    generation_info={"model": self.model, "backend": "cli"},
                )
            ]
        )

    @staticmethod
    def _run_async(coro: Any) -> None:
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(lambda: asyncio.run(coro)).result()
        else:
            asyncio.run(coro)

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
