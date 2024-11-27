from typing import List

import streamlit as st
from langchain_community.llms.vllm import VLLMOpenAI

from src.embeddings import queryDB


def main() -> None:
    # Connect to VLLMOpenAI
    llm: VLLMOpenAI = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base="http://localhost:2020/v1",
        model="./llm/model.gguf",
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.set_page_config(page_title="RAG Demo")

    st.markdown(body="# RAG Demo")
    st.markdown(body="> Powered by Llama3.2, ChromaDB, and Streamlit")

    with st.container(border=True):
        # Display chat messages
        message: dict
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt: str
        if prompt := st.chat_input("What is up?"):
            # Display chat message from user
            st.chat_message(name="user", avatar="🤔").markdown(prompt)
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            # Query ChromaDB
            queryResults: List[str] = queryDB(prompt=prompt, top_n=2)[
                "documents"
            ][  # noqa: E501
                0
            ]

            # Show documents
            st.chat_message(name="ai", avatar="📖").markdown(
                f"Found the following documents:\n\n* {queryResults[0]}\n\n* {queryResults[1]}",  # noqa: E501
            )

            # Update prompt
            updatedPrompt: str = f"Respond with the following context:\n\n* {queryResults[0]}\n\n* {queryResults[1]}\n\n{prompt}"  # noqa: E501

            # Show updated prompt
            st.chat_message(name="ai", avatar="✏️").markdown(
                f"Here is the updated prompt:\n\n {updatedPrompt}",
            )

            response = llm.invoke(input=updatedPrompt)

            # Display assistant response in chat message container
            st.chat_message(name="assistant", avatar="🤖").markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )  # noqa: E501


if __name__ == "__main__":
    main()
