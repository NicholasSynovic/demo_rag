#!/bin/bash

wget -O ./llm/model.gguf \
    --progress bar \
    -nc \
    https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

vllm serve \
    --device cuda \
    --dtype auto \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --load-format gguf \
    --seed 42 \
    --tokenizer "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --port 2020 \
    ./llm/model.gguf
