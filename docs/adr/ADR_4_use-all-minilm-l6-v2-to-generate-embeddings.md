# 4. Use `all-MiniLM-L6-v2` to generate embeddings

## Context

To generate embeddings of the document, we need to use an embedding model. This
model does not need to be the same model that we inference with, but does need
to be choosen to ensure that efficient similarity searches can be made.

## Decision

We will go with the `all-MiniLM-L6-v2` embedding model with the Sentence
Transformer library. This model has decent performance both in accuracy and in
speed, and is the default model used in the libraries documentation. We will run
this model on the GPU in order to improve inference performance.

## Consequences

When we are doing our similarity searches, we will have to use this model in
order to compatible embeddings.
