from __future__ import annotations

from typing import Any

import streamlit as st

from src.llm_client import create_chat_model, create_embedding_model, call_llm
from src.rag import vector_search


def _render_context_chunks(context_chunks: list[dict[str, Any]]) -> None:
    for index, result in enumerate(context_chunks, start=1):
        metadata = result.get("metadata", {})
        st.markdown(f"**Result {index}**")
        if metadata.get("title"):
            st.write(f"Title: {metadata['title']}")
        if metadata.get("category"):
            st.write(f"Category: {metadata['category']}")
        if metadata.get("country"):
            st.write(f"Country: {metadata['country']}")
        if metadata.get("section"):
            st.write(f"Section: {metadata['section']}")
        if metadata.get("source"):
            st.write(f"Source: {metadata['source']}")
        if metadata.get("page_number") is not None:
            st.write(f"Page: {metadata['page_number']}")
        st.write(f"Text: {result.get('text', '')}")
        if result.get("score") is not None:
            st.caption(f"Score: {result['score']:.4f}")


def main() -> None:
    st.set_page_config(
        page_title="HR Policy Retrieval Chat",
        page_icon="📘",
        layout="centered",
    )

    st.title("HR Policy Retrieval Chat")
    st.caption("Ask questions grounded in retrieved HR policy chunks from Elasticsearch.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_model" not in st.session_state:
        st.session_state.chat_model = create_chat_model()

    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = create_embedding_model()

    for item in st.session_state.chat_history:
        if item["role"] == "user":
            with st.chat_message("user"):
                st.write(item["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(item["content"])
                if item.get("context"):
                    with st.expander("Show retrieved context", expanded=False):
                        _render_context_chunks(item["context"])

    user_question = str(st.chat_input("Ask a question about an HR policy"))

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Executing…"):
            retrieved_chunks = vector_search(
                user_question,
                top_k=3,
                embedding_model=st.session_state.embedding_model,
            )

            if retrieved_chunks:
                answer_text = call_llm(
                    user_question=user_question,
                    context_chunks=retrieved_chunks,
                    chat_model=st.session_state.chat_model,
                )
            else:
                answer_text = (
                    "I couldn't find enough relevant policy context in Elasticsearch to answer that question confidently."
                )

        assistant_message = {
            "role": "assistant",
            "content": answer_text,
            "context": retrieved_chunks,
        }
        st.session_state.chat_history.append(assistant_message)

        with st.chat_message("assistant"):
            st.markdown(answer_text)
            if retrieved_chunks:
                with st.expander("Show retrieved context", expanded=False):
                    _render_context_chunks(retrieved_chunks)

if __name__ == "__main__":
    main()
