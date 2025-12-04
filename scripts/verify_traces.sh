#!/bin/bash
# =============================================================================
# Verify Traces Helper Script
# =============================================================================
# Verifies that all three types of traces (Nsys, Torch ET, Kineto CPU) are
# collected after a training run.
#
# Usage:
#   ./scripts/verify_traces.sh <workload_name>
#   ./scripts/verify_traces.sh llama3_8b_tp
#
# Or verify all workloads:
#   ./scripts/verify_traces.sh --all
# =============================================================================

set -euo pipefail

# Source common configuration for TRACE_BASE
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CCL_BENCH_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default trace base (same as common.sh)
TRACE_BASE="${SCRATCH:-$CCL_BENCH_HOME}/ccl-bench-traces"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
}

print_ok() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

verify_workload() {
    local workload_name="$1"
    local trace_dir="${TRACE_BASE}/${workload_name}"
    local nsys_found=0
    local torch_profiler_found=0
    local all_good=true

    print_header "Verifying traces for: ${workload_name}"
    echo "Trace directory: ${trace_dir}"
    echo ""

    if [[ ! -d ${trace_dir} ]]; then
        print_error "Trace directory does not exist!"
        return 1
    fi

    # ---------------------------------------------------------------------
    # 1. Check for Nsys traces
    # ---------------------------------------------------------------------
    echo "--- Nsys Traces (GPU kernels, NVTX markers) ---"

    local nsys_qdrep
    nsys_qdrep=$(find "${trace_dir}" -maxdepth 1 -name "*.qdrep" -type f 2>/dev/null | head -5)
    local nsys_rep
    nsys_rep=$(find "${trace_dir}" -maxdepth 1 -name "*.nsys-rep" -type f 2>/dev/null | head -5)

    if [[ -n ${nsys_qdrep} ]] || [[ -n ${nsys_rep} ]]; then
        nsys_found=1
        if [[ -n ${nsys_qdrep} ]]; then
            echo "${nsys_qdrep}" | while read -r f; do
                local size
                size=$(du -h "$f" 2>/dev/null | cut -f1)
                print_ok "Found: $(basename "$f") (${size})"
            done
        fi
        if [[ -n ${nsys_rep} ]]; then
            echo "${nsys_rep}" | while read -r f; do
                local size
                size=$(du -h "$f" 2>/dev/null | cut -f1)
                print_ok "Found: $(basename "$f") (${size})"
            done
        fi
    else
        print_warning "No Nsys traces found (.qdrep or .nsys-rep)"
        all_good=false
    fi
    echo ""

    # ---------------------------------------------------------------------
    # 2. Check for Torch Profiler traces (Torch ET + Kineto CPU)
    # ---------------------------------------------------------------------
    echo "--- Torch Profiler Traces (Torch ET + Kineto CPU) ---"

    local profile_trace_dir="${trace_dir}/profile_trace"

    if [[ -d ${profile_trace_dir} ]]; then
        # Find all iteration directories
        local iterations
        iterations=$(find "${profile_trace_dir}" -maxdepth 1 -type d -name "iteration_*" 2>/dev/null | sort)

        if [[ -n ${iterations} ]]; then
            local iter_count
            iter_count=$(echo "${iterations}" | wc -l)
            print_ok "Found ${iter_count} iteration(s) with traces"

            # Check each iteration for rank traces
            echo "${iterations}" | while read -r iter_dir; do
                local iter_name
                iter_name=$(basename "$iter_dir")
                local rank_traces
                rank_traces=$(find "${iter_dir}" -maxdepth 1 -name "rank*_trace.json" -type f 2>/dev/null | wc -l)

                if [[ ${rank_traces} -gt 0 ]]; then
                    # Get size of first trace
                    local first_trace
                    first_trace=$(find "${iter_dir}" -maxdepth 1 -name "rank*_trace.json" -type f 2>/dev/null | head -1)
                    local size
                    size=$(du -h "$first_trace" 2>/dev/null | cut -f1)
                    print_ok "  ${iter_name}: ${rank_traces} rank trace(s) (~${size} each)"
                    torch_profiler_found=1
                else
                    print_warning "  ${iter_name}: No rank traces found"
                fi
            done
        else
            print_warning "No iteration directories found in profile_trace/"
            all_good=false
        fi
    else
        print_warning "profile_trace/ directory does not exist"
        all_good=false
    fi
    echo ""

    # ---------------------------------------------------------------------
    # 3. Summary
    # ---------------------------------------------------------------------
    echo "--- Summary ---"

    if [[ ${nsys_found} -eq 1 ]]; then
        print_ok "Nsys traces: FOUND"
    else
        print_error "Nsys traces: MISSING"
        echo "    → Check if PROFILE_MODE was set to 'torch' (disables Nsys)"
    fi

    if [[ ${torch_profiler_found} -eq 1 ]]; then
        print_ok "Torch Profiler traces (Torch ET + Kineto): FOUND"
    else
        print_error "Torch Profiler traces: MISSING"
        echo "    → Check if PROFILE_MODE was set to 'nsys' (disables Torch Profiler)"
        echo "    → Verify enable_profiling = true in TOML config"
    fi

    echo ""

    if [[ ${nsys_found} -eq 1 ]] && [[ ${torch_profiler_found} -eq 1 ]]; then
        print_ok "All three trace types collected successfully!"
        return 0
    else
        print_warning "Some traces are missing. See details above."
        return 1
    fi
}

verify_all_workloads() {
    print_header "Verifying all workloads in ${TRACE_BASE}"

    if [[ ! -d ${TRACE_BASE} ]]; then
        print_error "Trace base directory does not exist: ${TRACE_BASE}"
        exit 1
    fi

    local workloads
    workloads=$(find "${TRACE_BASE}" -maxdepth 1 -type d -name "*" ! -name "$(basename "${TRACE_BASE}")" 2>/dev/null | sort)

    if [[ -z ${workloads} ]]; then
        print_warning "No workload directories found in ${TRACE_BASE}"
        exit 0
    fi

    local success_count=0
    local fail_count=0

    echo "${workloads}" | while read -r workload_dir; do
        local workload_name
        workload_name=$(basename "$workload_dir")

        if verify_workload "${workload_name}"; then
            ((success_count++)) || true
        else
            ((fail_count++)) || true
        fi
    done

    print_header "Overall Summary"
    echo "Workloads verified: $((success_count + fail_count))"
    echo "  Passed: ${success_count}"
    echo "  Failed: ${fail_count}"
}

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

usage() {
    echo "Usage: $0 <workload_name>"
    echo "       $0 --all"
    echo ""
    echo "Examples:"
    echo "  $0 llama3_8b_tp"
    echo "  $0 qwen3_32b_3d"
    echo "  $0 --all"
    echo ""
    echo "Environment:"
    echo "  TRACE_BASE: ${TRACE_BASE}"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

case "$1" in
    --all|-a)
        verify_all_workloads
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    *)
        verify_workload "$1"
        ;;
esac
