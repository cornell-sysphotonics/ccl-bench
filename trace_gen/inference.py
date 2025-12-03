from vllm import LLM
def main():
    llm = LLM(model="Qwen/Qwen3-32B", tensor_parallel_size=2, max_model_len=4096)
    outputs = llm.generate("Hello, my name is")

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()