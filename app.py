from __future__ import annotations

from typing import Any

import streamlit as st

from src.llm_client import create_chat_model, create_embedding_model, call_llm
from src.rag import vector_search


def _render_context_sources(context_chunks: list[dict[str, Any]]) -> None:
    source = "**Sources:**\n"
    for index, result in enumerate(context_chunks, start=1):
        metadata = result.get("metadata", {})
        source += f"{index}. Title: {metadata['title']}, {metadata['source']}, Page: {metadata['page_number']}\n"
    st.markdown(f"{source}")

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
                        _render_context_sources(item["context"])

    user_question = st.chat_input("Ask a question about an HR policy")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Executing…"):
            retrieved_chunks = vector_search(
                str(user_question),
                top_k=3,
                embedding_model=st.session_state.embedding_model,
            )

            if retrieved_chunks:
                answer_text = call_llm(
                    user_question=str(user_question),
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
                    _render_context_sources(retrieved_chunks)

if __name__ == "__main__":
    main()
