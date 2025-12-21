#!/usr/bin/env python3
"""
vLLM Profiling Wrapper with PyTorch ET and Kineto Trace Collection

Usage:
    python vllm_profiler.py --config experiments/configs/E1.1.yaml
"""

import argparse
import json
import os
import time

import torch
from torch.profiler import (
    ExecutionTraceObserver,
    ProfilerActivity,
    profile,
    schedule,
)
from vllm import LLM, SamplingParams
import yaml


class VLLMProfiler:
    def __init__(self, config_path):
        """Initialize vLLM profiler with experiment configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # Save configuration to output directory for analysis
        with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)

        # Get rank for multi-GPU setup
        self.rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))

    def create_llm_engine(self):
        """Create vLLM engine with parallelism configuration."""
        model_config = self.config["model"]
        parallel_config = self.config["parallelism"]

        print("[DEBUG] Initializing LLM with disable_log_stats=False")
        llm = LLM(
            model=model_config["name"],
            tensor_parallel_size=parallel_config.get("tp", 1),
            pipeline_parallel_size=parallel_config.get("pp", 1),
            trust_remote_code=True,
            dtype=model_config.get("precision", "bfloat16"),
            max_model_len=self.config["data"]["seq_len"],
            gpu_memory_utilization=model_config.get(
                "gpu_memory_utilization", 0.9
            ),
            enforce_eager=model_config.get("enforce_eager", False),
            disable_log_stats=False,
        )

        return llm

    def prepare_prompts(self):
        """Prepare input prompts for profiling."""
        batch_size = self.config["data"]["batch_size"]
        # seq_len = self.config["data"]["seq_len"]  # not really needed for dummy text

        base_prompt = "The future of artificial intelligence is "
        prompts = [base_prompt] * batch_size
        return prompts

    def run_profiled_inference(self):
        """Run vLLM inference with profiling enabled."""
        print(f"[Rank {self.rank}] Starting profiled inference...")

        # Create LLM engine
        llm = self.create_llm_engine()
        prompts = self.prepare_prompts()

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=self.config["data"].get("max_tokens", 100),
        )

        # Warm-up runs
        warmup_iters = self.config.get("warmup_iterations", 2)
        print(f"[Rank {self.rank}] Running {warmup_iters} warmup iterations...")
        for i in range(warmup_iters):
            _ = llm.generate(prompts, sampling_params)

        # Inspect LLM engine internals
        print(f"[DEBUG] LLM Engine dir: {dir(llm.llm_engine)}")
        if hasattr(llm.llm_engine, "stat_logger"):
            print(f"[DEBUG] Stat Logger: {llm.llm_engine.stat_logger}")
            print(
                f"[DEBUG] Stat Logger dir: {dir(llm.llm_engine.stat_logger)}"
            )

        # Profiled runs
        profile_iters = self.config.get("profile_iterations", 3)

        # Setup PyTorch ET observer
        et_file = os.path.join(
            self.output_dir, f"torch_et_{self.rank}.json"
        )
        et = ExecutionTraceObserver()
        et.register_callback(et_file)

        # Kineto trace handler
        def trace_handler(prof):
            kineto_file = os.path.join(
                self.output_dir,
                f"kineto_trace_{self.rank}.json",
            )
            prof.export_chrome_trace(kineto_file)
            print(
                f"[Rank {self.rank}] Saved Kineto trace to {kineto_file}"
            )

        print(f"[Rank {self.rank}] Starting profiled iterations...")

        iteration_times = []
        all_outputs = []

        # Start execution trace observer
        et.start()

        # Kineto profiler context
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=0, active=profile_iters),
            record_shapes=False,
            on_trace_ready=trace_handler,
        ) as prof:
            for iter_idx in range(profile_iters):
                start_time = time.perf_counter()

                outputs = llm.generate(prompts, sampling_params)
                all_outputs.extend(outputs)

                torch.cuda.synchronize()
                end_time = time.perf_counter()
                iter_time = end_time - start_time
                iteration_times.append(iter_time)

                prof.step()

                print(
                    f"[Rank {self.rank}] Iteration {iter_idx}: {iter_time:.3f}s"
                )

        # Stop execution trace observer
        et.stop()
        et.unregister_callback()
        print(f"[Rank {self.rank}] Saved PyTorch ET trace to {et_file}")

        # ---- Metrics extraction (no estimation / no fallback) ----
        ttft_list = []
        tpot_list = []

        # 1) Try StatLogger histograms (true aggregated metrics)
        def _extract_histogram_avg(logger_obj, attr_name: str):
            if not hasattr(logger_obj, attr_name):
                return []
            hist = getattr(logger_obj, attr_name)
            histograms = hist.values() if isinstance(hist, dict) else [hist]
            results = []
            for histogram in histograms:
                if hasattr(histogram, "collect"):
                    metrics_data = histogram.collect()
                    if metrics_data:
                        samples = metrics_data[0].samples
                        sum_val = next(
                            (s.value for s in samples if s.name.endswith("_sum")),
                            0,
                        )
                        count_val = next(
                            (s.value for s in samples if s.name.endswith("_count")),
                            0,
                        )
                        if count_val > 0:
                            results.append(sum_val / count_val)
            return results

        if hasattr(llm.llm_engine, "stat_logger"):
            print("[DEBUG] Attempting to extract metrics from StatLogger...")
            logger = llm.llm_engine.stat_logger
            ttft_list.extend(_extract_histogram_avg(logger, "histogram_time_to_first_token"))
            tpot_list.extend(_extract_histogram_avg(logger, "histogram_time_per_output_token"))

        # 2) Try per-request RequestOutput.metrics (true per-request metrics)
        metrics_found = 0
        if all_outputs:
            print(f"[DEBUG] Output count: {len(all_outputs)}")
            for request_output in all_outputs:
                metrics = getattr(request_output, "metrics", None)
                if metrics is None:
                    continue

                metrics_found += 1

                # TTFT: Time to first token (arrival to first token)
                if (
                    metrics.first_token_time is not None
                    and metrics.arrival_time is not None
                ):
                    ttft = (
                        metrics.first_token_time - metrics.arrival_time
                    )
                    ttft_list.append(ttft)

                # TPOT: Time per output token
                if (
                    metrics.finished_time is not None
                    and metrics.first_token_time is not None
                ):
                    gen_time = (
                        metrics.finished_time - metrics.first_token_time
                    )
                    output_len = len(request_output.outputs[0].token_ids)
                    if output_len > 1:
                        tpot = gen_time / (output_len - 1)
                        tpot_list.append(tpot)

            print(
                f"[DEBUG] Request outputs with metrics: "
                f"{metrics_found}/{len(all_outputs)}"
            )

        # ---- Aggregate stats (if lists are empty, just report 0) ----
        def _percentile(values, pct):
            if not values:
                return None
            vals = sorted(values)
            k = (pct / 100.0) * (len(vals) - 1)
            f = int(k)
            c = min(f + 1, len(vals) - 1)
            if c == f:
                return vals[f]
            return vals[f] + (vals[c] - vals[f]) * (k - f)

        stats = {
            "iteration_times": iteration_times,
            "avg_iteration_time": (
                sum(iteration_times) / len(iteration_times)
                if iteration_times
                else None
            ),
            "min_iteration_time": min(iteration_times)
            if iteration_times
            else None,
            "max_iteration_time": max(iteration_times)
            if iteration_times
            else None,
            "ttft_avg": (
                sum(ttft_list) / len(ttft_list) if ttft_list else None
            ),
            "tpot_avg": (
                sum(tpot_list) / len(tpot_list) if tpot_list else None
            ),
            "ttft_p99": _percentile(ttft_list, 99),
            "tpot_p99": _percentile(tpot_list, 99),
        }

        stats_file = os.path.join(
            self.output_dir, f"timing_stats_{self.rank}.json"
        )
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"[Rank {self.rank}] Profiling complete!")
        print(
            f"[Rank {self.rank}] Average iteration time: "
            f"{stats['avg_iteration_time']:.3f}s"
        )
        print(
            f"[Rank {self.rank}] Average TTFT: "
            f"{stats['ttft_avg']:.4f}s"
        )
        print(
            f"[Rank {self.rank}] Average TPOT: "
            f"{stats['tpot_avg']:.4f}s"
        )


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Profiling with Trace Collection"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML",
    )

    args = parser.parse_args()

    profiler = VLLMProfiler(args.config)
    profiler.run_profiled_inference()


if __name__ == "__main__":
    main()
