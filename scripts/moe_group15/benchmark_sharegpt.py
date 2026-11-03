"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    python -m vllm.entrypoints.openai.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000
        
    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.

Multi-turn (prompt chain) benchmarking with ShareGPT:
    add `--sharegpt-max-turns N` (0 = every turn in the conversation). Each
    conversation is replayed as a chain of dependent requests: turn k is only
    sent once turn k-1 has finished, and its prompt contains the full history
    including the text the server *actually generated* for the earlier turns.
    `--request-rate` then controls how fast new conversations start; follow-up
    turns are sent as soon as the previous response completes.
"""
import argparse
import asyncio
import json
import os
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np
from bench_utils_sharegpt import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer

def get_tokenizer(tokenizer_name, tokenizer_mode="auto", trust_remote_code=False, **kwargs):
        return AutoTokenizer.from_pretrained(
                        tokenizer_name, trust_remote_code=trust_remote_code
                            )


# Length limits used when sampling ShareGPT conversations. The first prompt must
# be at most MAX_PROMPT_LEN tokens, and the whole context (all earlier prompts
# and responses plus the current prompt and response) must fit in MAX_SEQ_LEN.
# Later turns that would overflow MAX_SEQ_LEN are dropped, which truncates the
# conversation at that point.
SHAREGPT_MAX_PROMPT_LEN = 1024
SHAREGPT_MAX_SEQ_LEN = 2048


@dataclass
class Turn:
    """One user message in a conversation."""
    prompt: str  # The user message only (no history).
    prompt_len: int  # Tokens in `prompt` alone.
    output_len: int  # Max tokens to generate for this turn.


@dataclass
class Conversation:
    """A chain of dependent requests. A single-prompt request is simply a
    conversation with one turn."""
    turns: List[Turn]


@dataclass
class ConversationResult:
    conversation_id: int
    planned_turns: int
    # One entry per request actually sent, in turn order. A chain stops at the
    # first failed request, so this can be shorter than `planned_turns`.
    outputs: List[RequestFuncOutput] = field(default_factory=list)
    # Wall time from sending the first request to receiving the last response.
    latency: float = 0.0

    @property
    def success(self) -> bool:
        return (len(self.outputs) == self.planned_turns
                and all(o.success for o in self.outputs))


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    p99_itl_ms: float
    # Conversation-level metrics. For single-turn workloads these mirror the
    # per-request numbers.
    completed_conversations: int
    mean_conv_latency_ms: float
    median_conv_latency_ms: float
    p99_conv_latency_ms: float


def to_single_turn_conversations(
        requests: List[Tuple[str, int, int]]) -> List[Conversation]:
    """Wrap (prompt, prompt_len, output_len) tuples as one-turn conversations."""
    return [
        Conversation(turns=[Turn(prompt, prompt_len, output_len)])
        for prompt, prompt_len, output_len in requests
    ]


def render_history_as_prompt(messages: List[Dict[str, str]]) -> str:
    """Flatten a chat history into one string for the plain-text backends.

    Each turn's prompt is the previous turn's prompt plus the generated reply
    plus the new user message, so earlier turns are always a strict prefix of
    later ones (which is what prefix caching relies on). With a single message
    this is just that message, so single-turn behaviour is unchanged.
    """
    return "\n\n".join(m["content"] for m in messages)


def generate_random_prompt(tokenizer: PreTrainedTokenizerBase, target_tokens: int) -> str:
    """Generate a random prompt with approximately target_tokens tokens."""
    # Get vocabulary size and sample random token IDs
    vocab_size = tokenizer.vocab_size
    
    # Sample random token IDs (avoiding special tokens)
    special_tokens = set(tokenizer.all_special_ids) if hasattr(tokenizer, 'all_special_ids') else set()
    
    # Generate random token IDs, excluding special tokens
    token_ids = []
    for _ in range(target_tokens):
        token_id = random.randint(0, vocab_size - 1)
        # Skip special tokens to avoid issues with decoding
        while token_id in special_tokens:
            token_id = random.randint(0, vocab_size - 1)
        token_ids.append(token_id)
    
    # Decode token IDs to text
    try:
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception:
        # Fallback to a simple repeated token if decoding fails
        print('Failed')
        prompt = "hello " * (target_tokens // 2)
    
    return prompt

def sample_dummy_dataset(num_requests: int, tokenizer: PreTrainedTokenizerBase, long_prompts: int, long_prompt_len: int):
    
    # Input: 42, output: 13
    prompts = []
    with open('tokens.txt', 'r') as log_file:
        for line in log_file:
            parts = line.split()
            input_tokens = int(parts[1].replace(',', ''))
            output_tokens = int(parts[-1])
            if input_tokens > 10 and input_tokens < 10000 and output_tokens < 10000:
                prompt = generate_random_prompt(tokenizer, input_tokens)
                prompts.append((prompt, input_tokens, output_tokens))
                if len(prompts) >= num_requests:
                    break
    
    random.shuffle(prompts)
    dataset = []
    for prompt, input_tokens, output_tokens in prompts:
        if len(dataset) == num_requests - long_prompts:
            break
        dataset.append((prompt, input_tokens, output_tokens))
        print('Appending: {}, {}'.format(input_tokens, output_tokens))
    
    while len(dataset) < num_requests:
        prompt = generate_random_prompt(tokenizer, long_prompt_len)
        dataset.append((prompt, long_prompt_len, 256))

    random.shuffle(dataset)
    return dataset

def _build_sharegpt_conversation(
    messages: List[dict],
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
    max_turns: Optional[int],
) -> Optional[Conversation]:
    """Turn a raw ShareGPT message list into a Conversation.

    Messages are paired positionally: even indices are user prompts and odd
    indices are the reference replies (which only supply the output length).
    Later turns are dropped, ending the chain there, as soon as one is too
    short or the accumulated context would exceed SHAREGPT_MAX_SEQ_LEN. A
    chain cannot skip a turn because every turn depends on the one before it.
    Returns None if not even the first turn qualifies.
    """
    num_pairs = len(messages) // 2
    if num_pairs < 5: # only long conversations
        return None
    if max_turns is not None:
        num_pairs = min(num_pairs, max_turns)

    turns: List[Turn] = []
    context_len = 0  # Planned tokens of history preceding the current turn.
    for i in range(num_pairs):
        prompt = messages[2 * i]["value"]
        prompt_len = len(tokenizer(prompt).input_ids)
        if fixed_output_len is not None:
            output_len = fixed_output_len
        else:
            completion = messages[2 * i + 1]["value"]
            output_len = len(tokenizer(completion).input_ids)

        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            break
        if i == 0 and prompt_len > SHAREGPT_MAX_PROMPT_LEN:
            # Prune too long first prompts.
            break
        if context_len + prompt_len + output_len > SHAREGPT_MAX_SEQ_LEN:
            # Prune too long sequences.
            break

        turns.append(Turn(prompt, prompt_len, output_len))
        context_len += prompt_len + output_len

    return Conversation(turns=turns) if turns else None


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    max_turns: Optional[int] = 1,
) -> List[Conversation]:
    """Sample `num_requests` conversations from a ShareGPT json file.

    `max_turns` caps the number of (user, assistant) exchanges kept per
    conversation; None means no cap. With max_turns=1 every conversation is a
    single prompt, matching the old behaviour.
    """
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    if max_turns is not None and max_turns < 1:
        raise ValueError("max_turns must be >= 1 (or None for no limit)")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 messages (one exchange).
    dataset = [
        data["conversations"] for data in dataset
        if len(data["conversations"]) >= 2
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Conversation] = []
    for messages in dataset:
        if len(filtered_dataset) == num_requests:
            break
        conversation = _build_sharegpt_conversation(messages, tokenizer,
                                                    fixed_output_len,
                                                    max_turns)
        if conversation is not None:
            filtered_dataset.append(conversation)

    return filtered_dataset


def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int]]:
    assert (
        input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path) as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(
        len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{
        "role": "user",
        "content": base_prompt,
    }]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (
        input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round(
        (input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
        prefix_len > base_prompt_offset
    ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List[Tuple[str, int, int]] = []
    for _ in range(num_requests):
        sampled_lines = "".join(
            prefix_lines +
            random.sample(poem_lines, num_input_lines - num_prefix_lines))

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False)
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append(
            (prompt, prompt_formatted, prompt_len, output_len))

    return sampled_requests


async def get_request(
    input_requests: List[Conversation],
    request_rate: float,
) -> AsyncGenerator[Conversation, None]:
    """Yield conversations at `request_rate` conversations per second."""
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    conversation_results: List[ConversationResult],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    """Compute request-level and conversation-level metrics.

    Request-level metrics treat every turn as its own request. The returned
    list of output lengths is aligned with the flattened outputs, i.e. the
    requests of conversation 0 in turn order, then conversation 1, and so on.
    """
    actual_output_lens = []
    total_input = 0
    completed = 0
    itls = []
    tpots = []
    ttfts = []
    latencies = []
    sts = []
    conv_latencies = []
    for conv in conversation_results:
        if conv.success:
            conv_latencies.append(conv.latency * 1000)
        for turn_idx, output in enumerate(conv.outputs):
            if output.success:
                # We use the tokenizer to count the number of output tokens for
                # all serving backends instead of looking at len(output.itl)
                # since multiple output tokens may be bundled together
                # Note: this may inflate the output token count slightly
                output_len = len(
                    tokenizer(output.generated_text,
                              add_special_tokens=False).input_ids)
                actual_output_lens.append(output_len)
                # For follow-up turns prompt_len covers the entire history.
                total_input += output.prompt_len
                if output_len > 1:
                    tpots.append(
                        (output.latency - output.ttft) / (output_len - 1))
                itls += output.itl
                ttfts.append(output.ttft)
                completed += 1
                latencies.append(output.latency * 1000)
                sts.append(output.start_time)
                if output_len > 1:
                    print('CONV: {} TURN: {} TTFT: {} TPOT: {} INPUT: {} OUTPUT: {} LATENCY: {} START: {}'.format(
                        conv.conversation_id, turn_idx, ttfts[-1], tpots[-1],
                        output.prompt_len, output_len, latencies[-1], sts[-1]))
            else:
                actual_output_lens.append(0)

    # for latency in latencies:
    #     print('[LAT]: {} ms'.format(latency))

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        completed_conversations=len(conv_latencies),
        mean_conv_latency_ms=np.mean(conv_latencies or 0),
        median_conv_latency_ms=np.median(conv_latencies or 0),
        p99_conv_latency_ms=np.percentile(conv_latencies or 0, 99),
    )

    return metrics, actual_output_lens


async def run_conversation(
    conversation_id: int,
    conversation: Conversation,
    request_func,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_executor: ThreadPoolExecutor,
    model_id: str,
    api_url: str,
    best_of: int,
    use_beam_search: bool,
    pbar: Optional[tqdm] = None,
) -> ConversationResult:
    """Replay one conversation as a chain of dependent chat requests."""
    loop = asyncio.get_running_loop()
    result = ConversationResult(
        conversation_id=conversation_id,
        planned_turns=len(conversation.turns),
    )
    messages: List[Dict[str, str]] = []

    start_time = time.perf_counter()

    for turn_idx, turn in enumerate(conversation.turns):
        messages.append({
            "role": "user",
            "content": turn.prompt,
        })

        # Tokenize the complete chat history using the model's chat template.
        prompt_len = await loop.run_in_executor(
            tokenizer_executor,
            lambda: len(tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )),
        )

        request_func_input = RequestFuncInput(
            model=model_id,
            prompt="",  # Unused by the chat completions backend
            messages=list(messages),
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=turn.output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )

        output = await request_func(
            request_func_input=request_func_input,
            pbar=pbar,
        )

        result.outputs.append(output)

        if not output.success:
            break

        messages.append({
            "role": "assistant",
            "content": output.generated_text,
        })

    result.latency = time.perf_counter() - start_time

    # Turns that were never sent still count towards the progress bar total.
    skipped_turns = result.planned_turns - len(result.outputs)
    if pbar is not None and skipped_turns:
        pbar.update(skipped_turns)

    return result


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Conversation],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    disable_tqdm: bool,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run..., url: {}".format(api_url))
    test_turn = input_requests[0].turns[0]
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_turn.prompt,
        messages=[{"role": "user", "content": test_turn.prompt}],
        api_url=api_url,
        prompt_len=test_turn.prompt_len,
        output_len=test_turn.output_len,
        best_of=best_of,
        use_beam_search=use_beam_search,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")
    print(f"Traffic request rate: {request_rate}")

    total_requests = sum(len(conv.turns) for conv in input_requests)
    multi_turn = any(len(conv.turns) > 1 for conv in input_requests)
    pbar = None if disable_tqdm else tqdm(total=total_requests)

    tokenizer_executor = ThreadPoolExecutor(max_workers=1)
    benchmark_start_time = time.perf_counter()
    tasks = []
    async for conversation in get_request(input_requests, request_rate):
        tasks.append(
            asyncio.create_task(
                run_conversation(
                    conversation_id=len(tasks),
                    conversation=conversation,
                    request_func=request_func,
                    tokenizer=tokenizer,
                    tokenizer_executor=tokenizer_executor,
                    model_id=model_id,
                    api_url=api_url,
                    best_of=best_of,
                    use_beam_search=use_beam_search,
                    pbar=pbar,
                )))
    conversation_results: List[ConversationResult] = await asyncio.gather(
        *tasks)

    if not disable_tqdm:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time
    tokenizer_executor.shutdown()

    metrics, actual_output_lens = calculate_metrics(
        conversation_results=conversation_results,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )
    # Flattened per-request view, in the same order as `actual_output_lens`.
    outputs: List[RequestFuncOutput] = [
        output for conv in conversation_results for output in conv.outputs
    ]

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    if multi_turn:
        print("{:<40} {:<10}".format(
            "Successful conversations:",
            f"{metrics.completed_conversations}/{len(conversation_results)}"))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):",
                                    metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                    metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)',
                               n=50,
                               c='-'))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                    metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s='Inter-token Latency', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    if multi_turn:
        print("{s:{c}^{n}}".format(s='End-to-end Conversation Latency',
                                   n=50,
                                   c='-'))
        print("{:<40} {:<10.2f}".format("Mean conversation latency (ms):",
                                        metrics.mean_conv_latency_ms))
        print("{:<40} {:<10.2f}".format("Median conversation latency (ms):",
                                        metrics.median_conv_latency_ms))
        print("{:<40} {:<10.2f}".format("P99 conversation latency (ms):",
                                        metrics.p99_conv_latency_ms))
    print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "completed_conversations": metrics.completed_conversations,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "mean_conv_latency_ms": metrics.mean_conv_latency_ms,
        "median_conv_latency_ms": metrics.median_conv_latency_ms,
        "p99_conv_latency_ms": metrics.p99_conv_latency_ms,
        # Per-request lists (one entry per request sent, in conversation order
        # then turn order). `conversation_ids` / `turn_indices` identify which
        # conversation and turn each entry belongs to.
        "conversation_ids": [
            conv.conversation_id for conv in conversation_results
            for _ in conv.outputs
        ],
        "turn_indices": [
            turn_idx for conv in conversation_results
            for turn_idx in range(len(conv.outputs))
        ],
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        # Per-conversation lists.
        "conversation_turns": [
            conv.planned_turns for conv in conversation_results
        ],
        "conversation_latencies": [
            conv.latency for conv in conversation_results
        ],
        "conversation_success": [
            conv.success for conv in conversation_results
        ],
    }
    return result


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)

    if args.sharegpt_max_turns < 0:
        raise ValueError("--sharegpt-max-turns must be >= 0")
    # 0 means "use every turn in the conversation".
    sharegpt_max_turns = (args.sharegpt_max_turns
                          if args.sharegpt_max_turns > 0 else None)

    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next "
            "release. Please use '--dataset-name' and "
            "'--dataset-path' in the future runs.",
            stacklevel=2)
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            max_turns=sharegpt_max_turns,
        )

    elif args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            max_turns=sharegpt_max_turns,
        )

    elif args.dataset_name == "sonnet":
        # Do not format the prompt, pass to message directly
        if args.backend == "openai-chat":
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt, prompt_len, output_len)
                              for prompt, prompt_formatted, prompt_len,
                              output_len in input_requests]
            input_requests = to_single_turn_conversations(input_requests)
        else:
            assert (
                tokenizer.chat_template or tokenizer.default_chat_template
            ), "Tokenizer/model must have chat template for sonnet dataset."
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt_formatted, prompt_len, output_len)
                              for prompt, prompt_formatted, prompt_len,
                              output_len in input_requests]
            input_requests = to_single_turn_conversations(input_requests)


    elif args.dataset_name == 'dummy':
        input_requests = to_single_turn_conversations(
            sample_dummy_dataset(args.num_prompts, tokenizer,
                                 args.long_prompts, args.long_prompt_len))

    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
        ))

    # Save config and results to json
    if args.save_result:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts
        result_json["sharegpt_max_turns"] = args.sharegpt_max_turns

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  #noqa
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="openai-chat",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in the "
        "next release.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "sonnet", "dummy"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts (conversations) to process.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request (every turn, for multi-turn "
        "runs). Overrides the output length from the ShareGPT dataset.")
    parser.add_argument(
        "--sharegpt-max-turns",
        type=int,
        default=1,
        help="Maximum number of (user, assistant) exchanges to replay per "
        "ShareGPT conversation, sent as a chain of dependent requests where "
        "each prompt includes the previous generated responses. 1 (default) "
        "sends a single prompt per conversation; 0 uses every turn. "
        "Conversations that run out of context early are cut short.")
    parser.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second (conversations per second for "
        "multi-turn runs; follow-up turns are sent as soon as the previous "
        "response finishes). If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument("--long-prompts", type=int, default=0)
    parser.add_argument("--long-prompt-len",
                        type=int,
                        default=32000,
                        help="Long prompt length for testing")
    args = parser.parse_args()
    main(args)
