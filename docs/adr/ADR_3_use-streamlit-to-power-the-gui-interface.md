# 3. Use Streamlit to power the GUI interface

## Context

LLMs are typically interfaced via a chat interface. We can create a simple chat
interface with libraries like Gradio and Streamlit.

## Decision

We will use Streamlit to handle the GUI between vLLM and the user.

## Consequences

Streamlit (as far as I know) cannot be bundled as an installable application.
