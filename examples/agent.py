"""
LangGraph ReAct Agent powered by Claude Code subscription.

Uses `create_react_agent` from langgraph with ChatClaudeCode as the LLM.
No API key needed — uses your Claude Pro/Max subscription via the CLI.

Requirements:
    pip install langchain-claude-code langgraph langchain-core

Usage:
    python agent.py
"""

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_claude_code import ChatClaudeCode


# ── Define tools ─────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Fake implementation for demo
    weather_data = {
        "london": "15°C, cloudy with light rain",
        "tokyo": "22°C, sunny and clear",
        "new york": "18°C, partly cloudy",
        "colombo": "30°C, humid with thunderstorms",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. E.g. '2 + 2' or '(10 * 5) / 2'."""
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def search_knowledge(query: str) -> str:
    """Search a knowledge base for information."""
    # Fake implementation for demo
    knowledge = {
        "donely": "Donely is an AI-driven business process automation platform that runs securely on your own infrastructure.",
        "langchain": "LangChain is a framework for developing applications powered by large language models.",
        "oauth2": "OAuth 2.0 is an authorization framework that enables third-party applications to obtain limited access to a web service.",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"No results found for: {query}"


# ── Create the agent ─────────────────────────────────────────

def create_agent():
    """Create a ReAct agent with Claude Code as the LLM."""
    llm = ChatClaudeCode(
        model="claude-sonnet-4-20250514",
        max_turns=5,  # allow multiple tool calls
    )

    tools = [get_weather, calculate, search_knowledge]

    agent = create_react_agent(
        model=llm,
        tools=tools,
    )

    return agent


# ── Run examples ─────────────────────────────────────────────

def main():
    agent = create_agent()

    print("=" * 60)
    print("Example 1: Simple question (no tools needed)")
    print("=" * 60)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What is LangChain?"}]}
    )
    last_msg = response["messages"][-1]
    print(f"Agent: {last_msg.content}\n")

    print("=" * 60)
    print("Example 2: Weather lookup (tool use)")
    print("=" * 60)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Colombo?"}]}
    )
    last_msg = response["messages"][-1]
    print(f"Agent: {last_msg.content}\n")

    print("=" * 60)
    print("Example 3: Multi-step reasoning")
    print("=" * 60)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What is (25 * 4) + (10 / 2)? Also, what is Donely?"}]}
    )
    last_msg = response["messages"][-1]
    print(f"Agent: {last_msg.content}\n")


if __name__ == "__main__":
    main()
