#!/bin/bash
# =============================================================================
# Register DeepSeek-V2-Lite TrainSpec
# =============================================================================
# This script registers the DeepSeek-V2-Lite TrainSpec before training.
# Source this script in your run.sbatch before launching training.
#
# Usage:
#   source register_deepseek_v2_lite.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add trace_collection directory to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Register the TrainSpec
python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
try:
    import register_deepseek_v2_lite
    print('✓ DeepSeek-V2-Lite TrainSpec registered successfully')
except Exception as e:
    print(f'✗ Failed to register TrainSpec: {e}')
    sys.exit(1)
" || {
    echo "ERROR: Failed to register DeepSeek-V2-Lite TrainSpec"
    exit 1
}

