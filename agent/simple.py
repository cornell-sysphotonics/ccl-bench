import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
)

logger = logging.getLogger("trace_agent")

KB_DIR = Path(__file__).parent / "knowledge"
KB_NOTES = KB_DIR / "notes.md"
KB_SCRIPTS = KB_DIR / "scripts"

SYSTEM_PROMPT = """\
You are a GPU trace analysis agent. You will be given a Kineto/PyTorch profiler \
trace file (JSON) and a description of metrics to compute. Your job is to write \
and execute Python code to parse the trace file and compute as many of the \
requested metrics as possible, then write the results to a JSON output file.

Important guidelines:
- The trace file can be very large (tens of MB). Always use streaming/incremental \
JSON parsing (ijson) or load it once and work in memory — never read it multiple times.
- Kernel events have cat="kernel". NCCL communication kernels have "nccl" in the name.
- Each event has "ts" (timestamp in microseconds) and "dur" (duration in microseconds).
- The distributedInfo section contains rank, world_size, and process group configs.
- Focus on metrics that can actually be computed from a single-rank Kineto trace.
- For metrics that require multi-rank data or external info not in the trace, \
set the value to null with a note explaining why.
- Write the final results as a JSON file with metric names as keys.

## Knowledge Base

You have a persistent knowledge base at: {kb_dir}

Structure:
  {kb_dir}/notes.md     — your running notes: learnings, gotchas, trace format details
  {kb_dir}/scripts/     — reusable Python scripts for trace analysis

At the START of each run:
- Read {kb_dir}/notes.md if it exists to recall what you learned before.
- Check {kb_dir}/scripts/ for existing analysis scripts you can reuse or adapt.

At the END of each run (after writing the output JSON):
- Update {kb_dir}/notes.md with anything new you learned (trace quirks, metric \
  formulas that worked, edge cases, etc.). Append to existing content — don't overwrite \
  previous notes.
- Save any reusable analysis scripts to {kb_dir}/scripts/ with descriptive names \
  (e.g., compute_nccl_metrics.py, parse_kernel_events.py). If a script already exists \
  and you improved it, update it in place.

This lets you get smarter across runs. Always check the knowledge base first before \
writing new code from scratch.
"""


def init_knowledge_base() -> str:
    """Initialize the knowledge base directory and return a summary of its contents."""
    KB_DIR.mkdir(exist_ok=True)
    KB_SCRIPTS.mkdir(exist_ok=True)

    summary_parts = []

    if KB_NOTES.exists():
        notes = KB_NOTES.read_text().strip()
        if notes:
            summary_parts.append(f"## Existing notes ({KB_NOTES}):\n{notes}")

    scripts = sorted(KB_SCRIPTS.glob("*.py"))
    if scripts:
        listing = "\n".join(f"  - {s.name} ({s.stat().st_size} bytes)" for s in scripts)
        summary_parts.append(f"## Existing scripts in {KB_SCRIPTS}:\n{listing}")

    if summary_parts:
        return (
            "The knowledge base has content from previous runs:\n\n"
            + "\n\n".join(summary_parts)
            + "\n\nRead the relevant files to reuse what you've already learned."
        )
    return "The knowledge base is empty — this appears to be the first run."


def setup_logging(log_file: str | None, verbose: bool) -> None:
    """Configure logging to both console and optional log file."""
    console_level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)


def log_tool_use(block: ToolUseBlock) -> None:
    """Log a tool use with its input (truncated for large inputs)."""
    input_str = json.dumps(block.input, indent=2)
    if len(input_str) > 2000:
        input_str = input_str[:2000] + "\n... (truncated)"
    logger.info("TOOL CALL: %s (id=%s)", block.name, block.id)
    logger.debug("TOOL INPUT:\n%s", input_str)


def log_tool_result(block: ToolResultBlock) -> None:
    """Log a tool result (truncated for large outputs)."""
    if isinstance(block.content, str):
        content = block.content
    else:
        content = json.dumps(block.content, indent=2) if block.content else "(empty)"
    if len(content) > 3000:
        content = content[:3000] + "\n... (truncated)"
    status = "ERROR" if block.is_error else "OK"
    logger.info("TOOL RESULT [%s]: tool_use_id=%s", status, block.tool_use_id)
    logger.debug("TOOL OUTPUT:\n%s", content)


async def run_agent(trace_path: str, metrics_path: str, output_path: str) -> None:
    kb_summary = init_knowledge_base()
    system_prompt = SYSTEM_PROMPT.replace("{kb_dir}", str(KB_DIR))

    prompt = f"""\
Analyze the GPU profiler trace file and compute metrics.

Trace file: {trace_path}
Metrics description: {metrics_path}
Output file: {output_path}
Knowledge base: {KB_DIR}

{kb_summary}

Steps:
1. Check the knowledge base for notes and reusable scripts.
2. Read the metrics description file to understand what metrics to compute.
3. Write/reuse a Python script to load the trace JSON and compute each metric.
4. Execute the script to produce the results.
5. Write the results as a JSON file to the output path.
6. Update the knowledge base with what you learned and save useful scripts.

Start by checking the knowledge base, then read the metrics file.
"""

    logger.info("=" * 60)
    logger.info("AGENT SESSION START")
    logger.info("Trace:   %s", trace_path)
    logger.info("Metrics: %s", metrics_path)
    logger.info("Output:  %s", output_path)
    logger.info("KB dir:  %s", KB_DIR)
    logger.info("KB status: %s", "has content" if kb_summary.startswith("The knowledge base has") else "empty")
    logger.info("=" * 60)

    turn = 0

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
            permission_mode="bypassPermissions",
            cwd=str(Path(trace_path).parent),
            max_turns=30,
        ),
    ):
        if isinstance(message, AssistantMessage):
            turn += 1
            logger.info("-" * 40)
            logger.info("ASSISTANT TURN %d (model=%s)", turn, message.model or "default")
            for block in message.content:
                if isinstance(block, ThinkingBlock):
                    thinking = block.thinking or ""
                    if len(thinking) > 1000:
                        thinking = thinking[:1000] + "... (truncated)"
                    logger.debug("THINKING:\n%s", thinking)
                elif isinstance(block, TextBlock):
                    logger.info("TEXT:\n%s", block.text)
                elif isinstance(block, ToolUseBlock):
                    log_tool_use(block)
                elif isinstance(block, ToolResultBlock):
                    log_tool_result(block)

        elif isinstance(message, UserMessage):
            logger.debug("USER MESSAGE (uuid=%s)", message.uuid)
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    log_tool_result(block)
                elif isinstance(block, TextBlock):
                    logger.debug("USER TEXT: %s", block.text)

        elif isinstance(message, SystemMessage):
            logger.info("SYSTEM [%s]: %s", message.subtype, message.data)

        elif isinstance(message, ResultMessage):
            logger.info("=" * 60)
            logger.info("AGENT SESSION END")
            logger.info("  Turns:      %d", message.num_turns)
            logger.info("  Duration:   %.1fs", (message.duration_ms or 0) / 1000)
            logger.info("  API time:   %.1fs", (message.duration_api_ms or 0) / 1000)
            logger.info("  Cost:       $%.4f", message.total_cost_usd or 0)
            logger.info("  Session ID: %s", message.session_id)
            if message.usage:
                logger.info("  Usage:      %s", message.usage)
            if message.is_error:
                logger.warning("  ERROR: %s", message.result)
            elif message.result:
                logger.info("  Result:     %s", message.result)
            logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute metrics from a GPU profiler trace using a Claude agent"
    )
    parser.add_argument("trace", help="Path to the Kineto trace JSON file")
    parser.add_argument(
        "--metrics",
        default=str(Path(__file__).parent / "metrics.md"),
        help="Path to metrics description (default: metrics.md)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <trace_dir>/metrics_output.json)",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Path to write detailed log file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show debug-level output on console",
    )
    args = parser.parse_args()

    trace_path = str(Path(args.trace).resolve())
    metrics_path = str(Path(args.metrics).resolve())
    output_path = args.output or str(
        Path(trace_path).parent / "metrics_output.json"
    )
    log_file = args.log or str(
        Path(output_path).with_suffix(".log")
    )

    setup_logging(log_file, args.verbose)
    logger.info("Log file: %s", log_file)

    asyncio.run(run_agent(trace_path, metrics_path, output_path))


if __name__ == "__main__":
    main()
