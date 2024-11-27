from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams


def downloadTinyLlama() -> str:
    hfRepo: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    modelFilename: str = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

    modelPath: str = hf_hub_download(
        repo_id=hfRepo, filename=modelFilename, local_dir="."
    )
    return modelPath


def main() -> None:
    model_path = downloadTinyLlama()
    PROMPT_TEMPLATE = "<|system|>\n{system_message}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"  # noqa: E501

    system_message = "You are a friendly chatbot who always responds in the style of a pirate."  # noqa: E501

    # Sample prompts.

    prompts = [
        "How many helicopters can a human eat in one sitting?",
        "What's the future of AI?",
    ]

    prompts = [
        PROMPT_TEMPLATE.format(system_message=system_message, prompt=prompt)
        for prompt in prompts
    ]

    # Create a sampling params object.

    sampling_params = SamplingParams(temperature=0, max_tokens=128)

    # Create an LLM.

    llm = LLM(
        model=model_path,
        tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        gpu_memory_utilization=0.95,
    )

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.

    for output in outputs:
        prompt = output.prompt

        generated_text = output.outputs[0].text

        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
