from __future__ import annotations

import streamlit as st

from src.llm_client import create_chat_model, create_embedding_model, call_llm
from src.rag import vector_search


def main() -> None:
    st.set_page_config(
        page_title="Utrains Support Chatbot",
        page_icon="🎓",
        layout="centered"
    )

    st.title("Utrains Support Chatbot")
    st.caption("Ask questions about training programs, support, schedules, and platform access.")

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
                        for index, result in enumerate(item["context"], start=1):
                            st.markdown(f"**Result {index}**")
                            st.write(f"Question: {result['question']}")
                            st.write(f"Answer: {result['answer']}")
                            if result.get("score") is not None:
                                st.caption(f"Score: {result['score']:.4f}")

    # 1. Get user prompt from the chat input box
    user_question = st.chat_input("Ask a question about a Utrains training program")

    if user_question:

        # add user question to the chat history and display on the UI
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)


        with st.spinner("Executing…"):
            # 2. search relevant context from the indexed Q&A data using the selected search mode
            retrieved_chunks = vector_search(
                str(user_question),
                top_k=3,
                embedding_model=st.session_state.embedding_model,
            )

            # 3. Use the retrieved context to call the LLM. If the context is empty, return a message indicating that
            # the answer cannot be provided.
            if retrieved_chunks:
                answer_text = call_llm(
                    user_question=str(user_question), # User Prompt
                    context_chunks=retrieved_chunks,  # The retrieved context
                    chat_model=st.session_state.chat_model,
                )
            else:
                answer_text = "I couldn't find enough context in the indexed Q&A data to answer that question confidently."

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
                    for index, result in enumerate(retrieved_chunks, start=1):
                        st.markdown(f"**Result {index}**")
                        st.write(f"Question: {result['question']}")
                        st.write(f"Answer: {result['answer']}")
                        if result.get("score") is not None:
                            st.caption(f"Score: {result['score']:.4f}")

if __name__ == "__main__":
    main()
