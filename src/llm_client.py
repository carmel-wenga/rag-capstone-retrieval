from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import settings

SYSTEM_PROMPT = """You are a helpful HR policy assistant.
Answer the user's question using only the context provided below.

Rules:
- Use the retrieved context as your only source of truth.
- If the context contains the answer, respond clearly and concisely.
- If the answer is not present in the context, say you do not have enough information from the provided policy material.
- Do not invent policy details, exceptions, dates, or country-specific rules.
- When helpful, mention relevant policy metadata such as title, country, section, or effective date.

Context:
{context}
"""


def create_embedding_model() -> OpenAIEmbeddings:
    """Create the LangChain embedding model configured for this project."""
    return OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model=settings.EMBEDDING_MODEL,
    )


def create_chat_model(llm_model: str | None = None) -> ChatOpenAI:
    """Create the LangChain chat model configured for this project."""
    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=llm_model or settings.LLM_MODEL,
    )


def get_embedding(text: str, embedding_model: OpenAIEmbeddings | None = None) -> list[float]:
    """Generate an embedding for a single input string using the configured model."""
    active_model = embedding_model or create_embedding_model()
    return active_model.embed_query(text)


def _format_context_chunk(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    metadata_parts = [
        f"title={metadata.get('title')}" if metadata.get("title") else None,
        f"category={metadata.get('category')}" if metadata.get("category") else None,
        f"country={metadata.get('country')}" if metadata.get("country") else None,
        f"section={metadata.get('section')}" if metadata.get("section") else None,
        f"effective_date={metadata.get('effective_date')}" if metadata.get("effective_date") else None,
        f"source={metadata.get('source')}" if metadata.get("source") else None,
        f"page_number={metadata.get('page_number')}" if metadata.get("page_number") is not None else None,
    ]
    metadata_line = ", ".join(part for part in metadata_parts if part)
    text = chunk.get("text", "")
    return f"Metadata: {metadata_line}\nText: {text}" if metadata_line else f"Text: {text}"


def call_llm(
    user_question: str,
    context_chunks: list[dict[str, Any]],
    chat_model: ChatOpenAI | None = None,
) -> str:
    """Generate a grounded answer using the retrieved context chunks."""
    context = "\n\n---\n\n".join(_format_context_chunk(chunk) for chunk in context_chunks)

    active_chat_model = chat_model or create_chat_model()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=user_question),
    ]

    response = active_chat_model.invoke(messages)

    return response.text
