import time
import json
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset


# ----------------------------
# Configuration (can be overridden by CLI)
# ----------------------------
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "/pscratch/sd/c/cp724/DeepSeek-V2-Lite"
CACHE_DIR = "/pscratch/sd/c/cp724/hf_cache"

GEN_MAX_TOKENS = 512


# ===========================================================
# Stream a single prompt (return chunk timestamps)
# ===========================================================
def stream_chat_completion(prompt: str, api_url: str, model_name: str, max_tokens: int):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": max_tokens,
    }

    t_start = time.perf_counter()
    timestamps = []
    chunks = []

    with requests.post(api_url, json=payload, stream=True, timeout=600) as resp:
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue

            if line.startswith(b"data:"):
                data = line[len(b"data:") :].strip()

                if data == b"[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta = obj["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    chunks.append(content)
                    timestamps.append(time.perf_counter())

    t_end = timestamps[-1] if timestamps else time.perf_counter()
    return chunks, timestamps, t_start, t_end


# ===========================================================
# Latency computation (based on streaming chunk arrival time)
# ===========================================================
def compute_latency_metrics(chunks, timestamps, t_start):
    if not chunks or not timestamps:
        return None

    ttft = timestamps[0] - t_start
    itl = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    tpot = (sum(itl) / len(itl)) if itl else None
    return ttft, itl, tpot


# ===========================================================
# Dataset loading (ensure non-empty text only)
# ===========================================================
def load_prompts(dataset_name: str, num_prompts: int):
    print(f"\n📌 Loading dataset: {dataset_name}")

    if dataset_name == "wikitext":
        from datasets import Dataset

        LOCAL_WIKITEXT = (
            "/pscratch/sd/c/cp724/hf_cache/wikitext/wikitext-2-raw-v1/0.0.0/"
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
                raw.append(obj.get("text", ""))

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
    prompts = filtered[:num_prompts]
    print(f"→ Loaded {len(prompts)} usable prompts.")
    return prompts


# ===========================================================
# Statistics utilities
# ===========================================================
def pctl(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def summarize_latencies(results):
    # results: list of dicts with keys: ttft, e2e, tpot
    ttft = [r["ttft"] for r in results if r["ttft"] is not None]
    e2e = [r["e2e"] for r in results if r["e2e"] is not None]
    tpot = [r["tpot"] for r in results if r["tpot"] is not None]

    def fmt(name, xs):
        if not xs:
            return f"{name}: N/A"
        return (
            f"{name}: mean={statistics.mean(xs):.4f}s  "
            f"p50={pctl(xs, 50):.4f}s  p90={pctl(xs, 90):.4f}s  p99={pctl(xs, 99):.4f}s"
        )

    return "\n".join([fmt("TTFT", ttft), fmt("E2E", e2e), fmt("TPOT", tpot)])


# ===========================================================
# Load test: controlled request rate + concurrency (batch size)
# ===========================================================
def run_load_test(
    prompts,
    api_url,
    model_name,
    max_tokens,
    num_requests,
    request_rate,
    concurrency,
    verbose=False,
    out_jsonl=None,
):
    # Generate prompt sequence (cycled from the prompt pool)
    seq = [prompts[i % len(prompts)] for i in range(num_requests)]

    results = []
    started_at = []
    completed_at = []

    fout = open(out_jsonl, "w") if out_jsonl else None

    def one_call(req_id, prompt):
        chunks, timestamps, t0, t1 = stream_chat_completion(
            prompt, api_url, model_name, max_tokens
        )
        metrics = compute_latency_metrics(chunks, timestamps, t0)
        ttft, itl, tpot = metrics if metrics else (None, [], None)
        e2e = t1 - t0
        return {
            "req_id": req_id,
            "ttft": ttft,
            "tpot": tpot,
            "e2e": e2e,
            "num_chunks": len(chunks),
            "gen_chars": len("".join(chunks)),
        }

    t_test_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = []

        # Send requests at the target request_rate
        # (request_rate <= 0 means send as fast as possible)
        for i, prompt in enumerate(seq):
            if request_rate and request_rate > 0:
                target = t_test_start + (i / request_rate)
                now = time.perf_counter()
                if target > now:
                    time.sleep(target - now)

            started_at.append(time.perf_counter())
            futures.append(ex.submit(one_call, i, prompt))

        # Collect completed requests
        for fut in as_completed(futures):
            r = fut.result()
            completed_at.append(time.perf_counter())
            results.append(r)

            if fout:
                fout.write(json.dumps(r) + "\n")
                fout.flush()

            if verbose:
                print(
                    f"[done] req={r['req_id']:4d} "
                    f"ttft={r['ttft'] if r['ttft'] is not None else 'NA'} "
                    f"e2e={r['e2e']:.4f}s "
                    f"tpot={r['tpot'] if r['tpot'] is not None else 'NA'} "
                    f"chunks={r['num_chunks']}"
                )

    t_test_end = time.perf_counter()
    if fout:
        fout.close()

    # Throughput = completed requests / wall-clock time
    wall = t_test_end - t_test_start
    throughput = (len(results) / wall) if wall > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"Requests: {len(results)}")
    print(f"Wall time: {wall:.4f} s")
    print(f"Throughput (completed): {throughput:.3f} req/s")
    print(summarize_latencies(results))
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=["wikitext", "c4", "redpajama"],
    )
    parser.add_argument(
        "--num-prompts", type=int, default=50, help="Size of the prompt pool (cycled)"
    )
    parser.add_argument(
        "--num-requests", type=int, default=30, help="Total number of requests"
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=5.0,
        help="Target request rate (req/s); <=0 means send as fast as possible",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Concurrency level (batch size)"
    )
    parser.add_argument("--max-tokens", type=int, default=GEN_MAX_TOKENS)
    parser.add_argument("--api-url", type=str, default=VLLM_API_URL)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default=None,
        help="Write per-request results to a JSONL file",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.dataset, args.num_prompts)
    if not prompts:
        raise RuntimeError("No usable prompts loaded.")

    run_load_test(
        prompts=prompts,
        api_url=args.api_url,
        model_name=args.model,
        max_tokens=args.max_tokens,
        num_requests=args.num_requests,
        request_rate=args.request_rate,
        concurrency=args.batch_size,
        verbose=args.verbose,
        out_jsonl=args.out_jsonl,
    )


if __name__ == "__main__":
    main()
