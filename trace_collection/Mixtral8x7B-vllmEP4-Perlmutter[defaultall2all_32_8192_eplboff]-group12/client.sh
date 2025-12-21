#the default benchmark script for varying basic configs (batch_size, sequence length, etc)
#change config in lauch scripts 
#u can run this in a seprate terminal (ssh to the same node)
#check with curl http://localhost:8000/health

DATASET=${1:-../ShareGPT_V3_unfiltered_cleaned_split.json}
INPUT_SEQUENCE_LEN=${2:-128}
PORT=${3:-8000}
NUM_PROMPTS=${4:-50}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
HF_CACHE_DIR="${SCRIPT_DIR}/../../hf_cache"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"

echo "checking server..."
#check server
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    echo "Waiting for server... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Server failed to start!"
    exit 1
fi

echo "========================================="
echo "Running benchmark with $NUM_PROMPTS prompts"
echo "========================================="


python3 benchmark.py \
    --backend vllm \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset-name sharegpt \
    --dataset-path ${DATASET} \
    --num-prompts ${NUM_PROMPTS} \
    --save-result \
    --result-dir results_json/

