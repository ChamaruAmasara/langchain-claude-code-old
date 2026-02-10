"""LangChain integration for Claude Code - use Claude Pro/Max subscription as a LangChain ChatModel."""

from langchain_claude_code.chat_models import ChatClaudeCode
from langchain_claude_code.oauth import ClaudeOAuthManager

__all__ = ["ChatClaudeCode", "ClaudeOAuthManager"]
__version__ = "0.1.0"
