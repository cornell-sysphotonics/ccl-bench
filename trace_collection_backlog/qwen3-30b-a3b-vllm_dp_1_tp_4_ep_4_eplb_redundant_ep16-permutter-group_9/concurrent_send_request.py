#!/usr/bin/env python3
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
import requests

def load_one_prompt(path: str, min_chars: int = 20) -> str:
    """Load the first non-empty line as the fixed prompt."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and len(line) >= min_chars:
                return line
    raise ValueError(f"No usable prompt found in {path}")


def percentile(sorted_vals, p: float):
    """p in [0,100]. Uses nearest-rank style."""
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[k]


def send_once(i: int, url: str, model: str, prompt: str, max_tokens: int, timeout_s: int):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        t1 = time.time()
        return i, (t1 - t0), resp.status_code, resp.text[:200]
    except Exception as e:
        t1 = time.time()
        return i, (t1 - t0), -1, repr(e)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/pscratch/sd/j/jy2222/wikitext103/wiki.test.raw")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", type=str, default="Qwen", help="--served-model-name")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=500, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=3, help="Number of concurrent requests")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup requests")
    parser.add_argument("--timeout", type=int, default=600, help="Per-request timeout")
    args = parser.parse_args()

    prompt = load_one_prompt(args.data)
    logger.info("Using ONE fixed prompt (first non-empty line) from: %s", args.data)
    logger.info("Prompt preview: %s", prompt[:120].replace("\n", " "))

    if args.warmup > 0:
        logger.info("Warmup: %d requests (serial)...", args.warmup)
        for w in range(args.warmup):
            _, lat, code, msg = send_once(w, args.url, args.model, prompt, args.max_tokens, args.timeout)
            if code != 200:
                logger.warning("Warmup[%d] failed: code=%s msg=%s", w, code, msg)

    logger.info(
        "Running load: repeat=%d, concurrency=%d, max_tokens=%d, url=%s, model=%s",
        args.repeat, args.concurrency, args.max_tokens, args.url, args.model
    )

    latencies = []
    failed = 0

    t_global0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [
            ex.submit(send_once, i, args.url, args.model, prompt, args.max_tokens, args.timeout)
            for i in range(args.repeat)
        ]
        for fut in as_completed(futs):
            i, lat, code, msg = fut.result()
            if code == 200:
                latencies.append(lat)
            else:
                failed += 1
                if failed <= 10:
                    logger.error("Fail[%d] code=%s msg=%s", i, code, msg)

    t_global1 = time.time()

    if not latencies:
        logger.error("All requests failed.")
        return

    lat_sorted = sorted(latencies)
    p50 = percentile(lat_sorted, 50)
    p95 = percentile(lat_sorted, 95)
    p99 = percentile(lat_sorted, 99)

    ok = len(latencies)
    total = args.repeat
    wall = t_global1 - t_global0

    logger.info("Done. ok=%d/%d (fail=%d). Wall=%.3fs", ok, total, failed, wall)
    logger.info(  "Latency(s): avg=%.3f p50=%.3f p95=%.3f p99=%.3f min=%.3f max=%.3f", mean(lat_sorted), p50, p95, p99, lat_sorted[0], lat_sorted[-1])
    logger.info( "Throughput: %.3f req/s (approx %.1f tok/s using max_tokens)",   ok / wall, (ok / wall) * args.max_tokens  )


if __name__ == "__main__":
    main()
