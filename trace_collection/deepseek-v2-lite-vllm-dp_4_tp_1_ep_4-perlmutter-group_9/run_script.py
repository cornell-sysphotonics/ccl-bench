import time
import json
import textwrap
import requests
from datasets import load_dataset

# ----------------------------
# Configuration
# ----------------------------
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "/pscratch/sd/c/cp724/DeepSeek-V2-Lite"
CACHE_DIR = "/pscratch/sd/c/cp724/hf_cache"

NUM_SAMPLES = 10
GEN_MAX_TOKENS = 512


# ===========================================================
# Stream a single prompt
# ===========================================================
def stream_chat_completion(prompt: str):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": GEN_MAX_TOKENS,
    }

    t_start = time.time()
    timestamps = []
    tokens = []

    with requests.post(VLLM_API_URL, json=payload, stream=True) as resp:
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue

            if line.startswith(b"data: "):
                data = line[len(b"data: ") :]

                if data == b"[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta = obj["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    tok = delta["content"]
                    tokens.append(tok)
                    timestamps.append(time.time())

    # If no tokens are generated, use the HTTP end time
    t_end = timestamps[-1] if timestamps else time.time()

    return tokens, timestamps, t_start, t_end


# ===========================================================
# Latency metrics
# ===========================================================
def compute_latency_metrics(tokens, timestamps, t_start):
    if not tokens:
        return None

    ttft = timestamps[0] - t_start

    itl = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    tpot = sum(itl) / len(itl) if itl else None

    return ttft, itl, tpot


# ===========================================================
# Single prompt test
# ===========================================================
def process_one_text(text):
    prompt = text.strip()
    if not prompt:
        return

    print("\n" + "=" * 80)
    print("Input text:")
    print(textwrap.fill(prompt, width=80))

    tokens, timestamps, t_start, t_end = stream_chat_completion(prompt)

    metrics = compute_latency_metrics(tokens, timestamps, t_start)
    if metrics is None:
        print("⚠ Empty generation")
        return

    ttft, itl, tpot = metrics
    e2e = t_end - t_start  # End-to-end request latency

    print(f"\nGenerated tokens: {len(tokens)}")
    print(f"TTFT: {ttft:.4f} s")
    print(f"E2E Latency: {e2e:.4f} s")

    if tpot is None:
        print("TPOT: N/A (only one token generated)")
    else:
        print(f"TPOT: {tpot:.6f} s")

    print("\nITL list:")
    if itl:
        print([round(x, 4) for x in itl])
    else:
        print("N/A (only one token)")

    print("\n[Continuations]:")
    print(textwrap.fill("".join(tokens), width=80))

    print("=" * 80)


# ===========================================================
# Dataset loading
# ===========================================================
def load_dataset_10(dataset_name):
    print(f"\n📌 Loading dataset: {dataset_name}")

    if dataset_name == "wikitext":
        from datasets import Dataset

        LOCAL_WIKITEXT = (
            "/pscratch/sd/c/cp724/hf_cache/wikitext/"
            "wikitext-2-raw-v1/0.0.0/"
            "b08601e04326c79dfdd32d625aee71d232d685c3"
        )

        ds = Dataset.from_file(f"{LOCAL_WIKITEXT}/wikitext-test.arrow")
        raw = ds["text"]

    elif dataset_name == "c4":
        print("📌 Using local C4 shard")
        local_path = (
            "/pscratch/sd/c/cp724/datasets/c4/en/c4-validation.00000-of-00008.json.gz"
        )

        raw = []
        import gzip

        with gzip.open(local_path, "rt") as f:
            for line in f:
                obj = json.loads(line)
                txt = obj.get("text", "").strip()
                raw.append(txt)

    elif dataset_name == "redpajama":
        ds = load_dataset(
            "togethercomputer/RedPajama-Data-1T-Sample",
            split="train",
            cache_dir=CACHE_DIR,
        )
        raw = ds["text"]

    else:
        raise ValueError("Unknown dataset name")

    filtered = [t.strip() for t in raw if t and t.strip() and len(t.strip()) > 20]

    texts = filtered[:NUM_SAMPLES]
    print(f"→ Loaded {len(texts)} usable prompts.")
    return texts


# ===========================================================
# Main function
# ===========================================================
def main():
    DATASET = "wikitext"

    texts = load_dataset_10(DATASET)[:10]

    for i, t in enumerate(texts):
        print(f"\n### === Processing input line {i} === ###")
        process_one_text(t)


if __name__ == "__main__":
    main()
