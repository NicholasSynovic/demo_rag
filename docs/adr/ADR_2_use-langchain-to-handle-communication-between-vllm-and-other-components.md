# 2. Use LangChain to handle communication between vLLM and other components

## Context

To inference with vLLM, we would be limited to the bundled HTTP REST server.
While this is OpenAI compatible, it makes more sense to abstract this
implementation behind a different service. That way, we have the flexibility of
changing our backends without having to rewrite API logic.

## Decision

We will use LangChain to abstract the interface details of vLLM. We will still
need to interface directly with vLLM to load and download models. However, the
actual user-LLM interactions will be handled by LangChain.

## Consequences

LangChain is non-trivial to setup, and is still changing more frequently than
desired. Thus, we should lock te version of LangChain after installation.
