from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import settings

SYSTEM_PROMPT = """You are a helpful support assistant for Utrains training programs.
Answer the user's question using only the context provided below.

Rules:
- Use the retrieved context as your only source of truth.
- If the context contains the answer, respond clearly and concisely.
- If the answer is not present in the context, say you do not have enough information from the provided course material.
- Do not invent program details, policies, or timing.
- Prefer a direct answer, with a short note when the response is based on retrieved support context.

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
    print(f"[EMBEDDINGS] Get Embeddings For: {text}")
    return active_model.embed_query(text)


def call_llm(
        user_question: str,
        context_chunks: list[dict],
        chat_model: ChatOpenAI | None = None,
    ) -> str:
    """Generate a grounded answer using the retrieved context chunks."""

    # Prepare the retrieved Q&A records as context
    context = "\n\n---\n\n".join([f"Q: {chunk['question']}\nA: {chunk['answer']}" for chunk in context_chunks])

    active_chat_model = chat_model or create_chat_model()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=user_question),
    ]

    response = active_chat_model.invoke(messages)

    return response.text
