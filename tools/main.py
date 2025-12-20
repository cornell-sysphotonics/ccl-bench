#!/usr/bin/env python3
r"""CCL-Bench Metrics Analysis Tool.

Runs all metric tools on profile traces and generates a comprehensive HTML report.

Usage:
    python main.py --workload-dir <path> [--output <output_dir>]

Example:
    python main.py \\
        --workload-dir /pscratch/sd/g/gb555/final_proj/ccl-torchtitan-train/trace_collection/deepseek-v2-lite-torchtitan-fsdp-perlmutter-16

The tool will automatically find:
    - Traces in: <workload-dir>/traces/
    - Workload card: <workload-dir>/workload_card.yaml
    - Output: <workload-dir>/analysis_output/ (default)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
from typing import Any


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
_LOGGER = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result from a single metric tool."""

    tool_name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class AnalysisReport:
    """Complete analysis report with all metrics."""

    profile_dir: str
    workload_card: str
    metrics: list[MetricResult] = field(default_factory=list)
    iterations: list[str] = field(default_factory=list)


def discover_tools(tools_dir: Path) -> list[str]:
    """Discover all metric tools in the tools directory.

    Looks for subdirectories containing a metric.py file with a metric_cal function.

    Returns:
        List of tool names (directory names)
    """
    tools = []

    for item in tools_dir.iterdir():
        if not item.is_dir():
            continue

        # Skip common directories and hidden directories
        if item.name.startswith("_") or item.name in ["common", "__pycache__"]:
            continue

        # Check if it has a metric.py file
        metric_file = item / "metric.py"
        if metric_file.exists():
            tools.append(item.name)
            _LOGGER.info(f"Discovered tool: {item.name}")

    return sorted(tools)


def find_iterations(traces_dir: Path) -> list[tuple[str, Path]]:
    """Find all iteration directories in the traces directory.

    Looks for:
    - profile_traces/iteration_*/
    - Direct iteration_* subdirectories

    Returns:
        List of (iteration_name, iteration_path) tuples
    """
    iterations = []

    # Check for profile_traces/iteration_* structure
    profile_traces_dir = traces_dir / "profile_traces"
    if profile_traces_dir.exists():
        for item in sorted(profile_traces_dir.iterdir()):
            if item.is_dir() and item.name.startswith("iteration_"):
                iterations.append((item.name, item))

    # Also check for direct iteration_* directories in traces_dir
    for item in sorted(traces_dir.iterdir()):
        if item.is_dir() and item.name.startswith("iteration_"):
            if (item.name, item) not in iterations:
                iterations.append((item.name, item))

    if not iterations:
        # If no iterations found, use the traces_dir itself
        _LOGGER.warning(f"No iteration directories found, using traces_dir: {traces_dir}")
        iterations.append(("all", traces_dir))

    return iterations


def run_tool(
    tool_name: str,
    iteration_path: Path,
    workload_card_path: Path,
    tools_dir: Path,
) -> MetricResult:
    """Run a single metric tool on an iteration.

    Args:
        tool_name: Name of the tool (directory name)
        iteration_path: Path to iteration directory
        workload_card_path: Path to workload card YAML
        tools_dir: Base tools directory

    Returns:
        MetricResult with tool output
    """
    try:
        # Import the tool module
        module_path = f"tools.{tool_name}.metric"
        spec = importlib.util.spec_from_file_location(
            module_path, tools_dir / tool_name / "metric.py"
        )

        if spec is None or spec.loader is None:
            return MetricResult(
                tool_name=tool_name,
                success=False,
                error=f"Could not load module for {tool_name}",
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the metric_cal function
        if not hasattr(module, "metric_cal"):
            return MetricResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool {tool_name} does not have metric_cal function",
            )

        metric_func = module.metric_cal

        # Call the metric function
        # Check function signature to see if it accepts workload_card_path
        import inspect

        sig = inspect.signature(metric_func)

        if "workload_card_path" in sig.parameters:
            result = metric_func(
                str(iteration_path),
                workload_card_path=str(workload_card_path),
            )
        else:
            result = metric_func(str(iteration_path))

        return MetricResult(
            tool_name=tool_name,
            success=True,
            data=result if isinstance(result, dict) else {"value": result},
        )

    except Exception as e:
        _LOGGER.exception(f"Error running tool {tool_name}: {e}")
        return MetricResult(
            tool_name=tool_name,
            success=False,
            error=str(e),
        )


def generate_html_report(
    report: AnalysisReport,
    output_dir: Path,
) -> Path:
    """Generate an HTML report with all metrics, charts, and visualizations.

    Args:
        report: AnalysisReport with all metrics
        output_dir: Output directory for the report

    Returns:
        Path to generated HTML file
    """
    html_path = output_dir / "report.html"

    # Group metrics by iteration
    metrics_by_iteration: dict[str, list[MetricResult]] = {}
    for metric in report.metrics:
        # Extract iteration from metric data if available
        iteration = metric.data.get("iteration", "all")
        if iteration not in metrics_by_iteration:
            metrics_by_iteration[iteration] = []
        metrics_by_iteration[iteration].append(metric)

    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CCL-Bench Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            min-height: 100vh;
            color: #1a1a2e;
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 3rem 2rem;
            margin-bottom: 2rem;
            border-radius: 1rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}

        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }}

        header .subtitle {{
            opacity: 0.8;
            font-size: 1.1rem;
        }}

        section {{
            background: white;
            border-radius: 1rem;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        section h2 {{
            color: #1a1a2e;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 3px solid #2E86AB;
            font-size: 1.5rem;
        }}

        .metric-card {{
            background: #f8f9fa;
            border-left: 4px solid #2E86AB;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
        }}

        .metric-card h3 {{
            color: #2E86AB;
            margin-bottom: 0.5rem;
            font-size: 1.2rem;
        }}

        .metric-card.error {{
            border-left-color: #C73E1D;
            background: #fff5f5;
        }}

        .metric-card.error h3 {{
            color: #C73E1D;
        }}

        .metric-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #1a1a2e;
            margin: 0.5rem 0;
        }}

        .metric-details {{
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #dee2e6;
        }}

        .metric-details pre {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            font-size: 0.9rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .stat-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            text-align: center;
        }}

        .stat-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: #2E86AB;
        }}

        .stat-card .label {{
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: #6c757d;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 CCL-Bench Analysis Report</h1>
            <p class="subtitle">Traces Directory: {report.profile_dir}</p>
            <p class="subtitle">Workload Card: {report.workload_card}</p>
        </header>

        <div class="summary-grid">
            <div class="stat-card">
                <div class="value">{len(report.metrics)}</div>
                <div class="label">Total Metrics</div>
            </div>
            <div class="stat-card">
                <div class="value">{sum(1 for m in report.metrics if m.success)}</div>
                <div class="label">Successful</div>
            </div>
            <div class="stat-card">
                <div class="value">{sum(1 for m in report.metrics if not m.success)}</div>
                <div class="label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="value">{len(report.iterations)}</div>
                <div class="label">Iterations</div>
            </div>
        </div>
"""

    # Add metrics section
    html_content += """
        <section>
            <h2>📈 Metrics</h2>
"""

    for metric in report.metrics:
        if metric.success:
            # Format metric data
            data_str = json.dumps(metric.data, indent=2)
            html_content += f"""
            <div class="metric-card">
                <h3>{metric.tool_name}</h3>
                <div class="metric-details">
                    <pre>{data_str}</pre>
                </div>
            </div>
"""
        else:
            html_content += f"""
            <div class="metric-card error">
                <h3>{metric.tool_name}</h3>
                <p><strong>Error:</strong> {metric.error}</p>
            </div>
"""

    html_content += """
        </section>

        <footer>
            <p>Generated by CCL-Bench Metrics Analysis Tool</p>
        </footer>
    </div>
</body>
</html>
"""

    # Write HTML file
    with open(html_path, "w") as f:
        f.write(html_content)

    _LOGGER.info(f"Generated HTML report: {html_path}")
    return html_path


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CCL-Bench Metrics Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workload-dir",
        type=str,
        required=True,
        help="Path to workload directory containing traces/ and workload_card.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for reports (default: <workload-dir>/analysis_output)",
    )
    parser.add_argument(
        "--tools-dir",
        type=str,
        default=None,
        help="Path to tools directory (default: same directory as main.py)",
    )
    parser.add_argument(
        "--tools",
        type=str,
        nargs="+",
        default=None,
        help="Specific tools to run (default: all discovered tools)",
    )

    args = parser.parse_args()

    # Resolve workload directory
    workload_dir = Path(args.workload_dir).resolve()

    # Auto-detect traces directory and workload card
    traces_dir = workload_dir / "traces"
    workload_card = workload_dir / "workload_card.yaml"

    # Set output directory (default to workload_dir/analysis_output)
    output_dir = Path(args.output).resolve() if args.output else workload_dir / "analysis_output"

    if args.tools_dir:
        tools_dir = Path(args.tools_dir).resolve()
    else:
        # Assume tools directory is parent of this script
        tools_dir = Path(__file__).parent.resolve()

    # Validate inputs
    if not workload_dir.exists():
        _LOGGER.error(f"Workload directory does not exist: {workload_dir}")
        return 1

    if not traces_dir.exists():
        _LOGGER.error(f"Traces directory does not exist: {traces_dir}")
        _LOGGER.error(f"Expected traces in: {traces_dir}")
        return 1

    if not workload_card.exists():
        _LOGGER.error(f"Workload card does not exist: {workload_card}")
        _LOGGER.error(f"Expected workload card at: {workload_card}")
        return 1

    if not tools_dir.exists():
        _LOGGER.error(f"Tools directory does not exist: {tools_dir}")
        return 1

    _LOGGER.info(f"Workload directory: {workload_dir}")
    _LOGGER.info(f"Traces directory: {traces_dir}")
    _LOGGER.info(f"Workload card: {workload_card}")
    _LOGGER.info(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover tools
    if args.tools:
        tools = args.tools
        _LOGGER.info(f"Using specified tools: {tools}")
    else:
        tools = discover_tools(tools_dir)
        _LOGGER.info(f"Discovered {len(tools)} tools: {tools}")

    if not tools:
        _LOGGER.error("No tools found to run")
        return 1

    # Find all iterations
    all_iterations = find_iterations(traces_dir)
    _LOGGER.info(f"Found {len(all_iterations)} iterations: {[it[0] for it in all_iterations]}")

    # Select the middle iteration
    if not all_iterations:
        _LOGGER.error("No iterations found to analyze")
        return 1

    # Get middle iteration (if even number, take the one before middle)
    middle_idx = (len(all_iterations) - 1) // 2
    iterations = [all_iterations[middle_idx]]
    iter_name, iter_path = iterations[0]

    _LOGGER.info(
        f"Analyzing middle iteration: {iter_name} (out of {len(all_iterations)} iterations)"
    )

    # Initialize report
    report = AnalysisReport(
        profile_dir=str(traces_dir),
        workload_card=str(workload_card),
        iterations=[it[0] for it in all_iterations],  # Keep all for reference
    )

    # Run tools on the middle iteration only
    _LOGGER.info("Running metrics analysis...")
    _LOGGER.info(f"Processing iteration: {iter_name}")

    for tool_name in tools:
        _LOGGER.info(f"  Running tool: {tool_name}")
        result = run_tool(tool_name, iter_path, workload_card, tools_dir)
        result.data["iteration"] = iter_name
        report.metrics.append(result)

    # Generate HTML report
    _LOGGER.info("Generating HTML report...")
    html_path = generate_html_report(report, output_dir)

    # Also save JSON report
    json_path = output_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "profile_dir": report.profile_dir,
                "workload_card": report.workload_card,
                "iterations": report.iterations,
                "metrics": [
                    {
                        "tool_name": m.tool_name,
                        "success": m.success,
                        "data": m.data,
                        "error": m.error,
                    }
                    for m in report.metrics
                ],
            },
            f,
            indent=2,
        )

    _LOGGER.info("Analysis complete!")
    _LOGGER.info(f"  HTML report: {html_path}")
    _LOGGER.info(f"  JSON report: {json_path}")

    # Flush output to ensure all logs are written
    sys.stdout.flush()
    sys.stderr.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
