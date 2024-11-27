import streamlit


def main() -> None:
    streamlit.set_page_config(page_title="RAG Demo")

    streamlit.markdown(body="# Retrieval Augmented Generation (RAG) Demo")
    streamlit.markdown(
        body="> Demo web app of using LLMs *offline* to become subject matter experts on private documents"
    )
    streamlit.divider()

    streamlit.markdown(body="## What is RAG?")
    streamlit.markdown(
        body="Retrieval-Augmented Generation (RAG) is a technique that enhances the output of large language models (LLMs) by allowing them to reference external knowledge sources or authoritative databases before generating a response. This enables LLMs to provide more relevant, accurate, and context-specific outputs for various domains without needing to retrain the model, making it a cost-effective approach."
    )
    streamlit.markdown(
        body="> [Source (AWS)](https://aws.amazon.com/what-is/retrieval-augmented-generation/)"
    )
    streamlit.divider()


if __name__ == "__main__":
    main()
