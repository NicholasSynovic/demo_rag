# 1. Use vLLM to host LLM models

## Context

LLMs are technically difficult to host on a single device. Inference engines
such as Ollama, Onnx Runtime, and vLLM ease the technical hurdle to leverage
LLMs in applications.

## Decision

We will use vLLM as the inference engine for this demo. The reason is that vLLM
supports advanced and optimized inference engine features such as paged
attention and FlashAttention algorithms.

## Consequences

We will be constrained to the supported models of vLLM. Additionally, this may
not be deployable at scale (although there is a Docker image to review).
