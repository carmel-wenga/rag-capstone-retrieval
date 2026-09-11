from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch
from langchain_openai import OpenAIEmbeddings

from src.config import settings
from src.llm_client import get_embedding

client = Elasticsearch(settings.ELASTICSEARCH_HOST)


def _serialize_hit(hit: dict[str, Any]) -> dict[str, Any]:
    source = hit.get("_source", {})
    metadata = source.get("metadata", {})
    context = source.get("context", {})

    return {
        "text": context.get("text", ""),
        "metadata": metadata,
        "score": hit.get("_score"),
    }


def vector_search(
    query: str,
    top_k: int = 3,
    embedding_model: OpenAIEmbeddings | None = None,
) -> list[dict[str, Any]]:
    """Embed the query and retrieve the nearest HR policy chunks."""
    embedding = get_embedding(query, embedding_model=embedding_model)
    response = client.search(
        index=settings.INDEX_NAME,
        knn={
            "field": settings.VECTOR_FIELD,
            "query_vector": embedding,
            "k": top_k,
            "num_candidates": max(top_k * 10, 20),
        },
        source_includes=["metadata", "context.text"],
    )
    return [
        serialized_hit
        for hit in response.get("hits", {}).get("hits", [])
        if (serialized_hit := _serialize_hit(hit)).get("text")
    ]
