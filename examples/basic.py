"""Basic usage examples for langchain-claude-code."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_claude_code import ChatClaudeCode


def example_invoke():
    """Simple invoke."""
    llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
    response = llm.invoke("What is 2 + 2?")
    print(f"Response: {response.content}")


def example_messages():
    """System + Human messages."""
    llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
    messages = [
        SystemMessage(content="You are a Python expert. Be concise."),
        HumanMessage(content="Write a one-liner to flatten a nested list."),
    ]
    response = llm.invoke(messages)
    print(f"Response: {response.content}")


def example_streaming():
    """Streaming output — tokens arrive incrementally."""
    llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
    print("Streaming: ", end="", flush=True)
    for chunk in llm.stream("Count from 1 to 5, one per line."):
        print(chunk.content, end="", flush=True)
    print()


def example_chain():
    """LangChain chain."""
    llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Explain things simply in 2 sentences."),
        ("human", "{topic}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"topic": "OAuth2"})
    print(f"Response: {response.content}")


def example_image():
    """Multimodal — passing an image via URL."""
    llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
    response = llm.invoke([
        HumanMessage(content=[
            {"type": "text", "text": "What colors do you see in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/en/a/a9/Example.jpg"
                },
            },
        ])
    ])
    print(f"Response: {response.content}")


if __name__ == "__main__":
    print("=== Invoke ===")
    example_invoke()
    print("\n=== Messages ===")
    example_messages()
    print("\n=== Streaming ===")
    example_streaming()
    print("\n=== Chain ===")
    example_chain()
    print("\n=== Image ===")
    example_image()
