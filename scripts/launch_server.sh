#!/bin/bash

# Configuration
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/FAPO-GenRM-4B"}
BASE_PORT=31000          # First worker port (will increment for each worker)
NUM_WORKERS=8            # Number of workers (should match your GPU count)
ROUTER_PORT=30000        # Router port

MODEL_NAME=$(basename "$MODEL_PATH")

# Launch workers
declare -a WORKER_URLS
for ((i=0; i<NUM_WORKERS; i++)); do
    PORT=$((BASE_PORT + i))
    WORKER_URLS+=("http://localhost:$PORT")

    echo "Launching worker $i on GPU $i at port $PORT"
    CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
        --model-path $MODEL_PATH --served-model-name $MODEL_NAME --host localhost --port $PORT > /dev/null 2>&1 &
done

# Launch router
echo "Launching router at port $ROUTER_PORT with worker URLs:"
printf '  %s\n' "${WORKER_URLS[@]}"
python -m sglang_router.launch_router --host 0.0.0.0 --port $ROUTER_PORT --balance-abs-threshold 4 --worker-urls "${WORKER_URLS[@]}" > router.log 2>&1 &

# Cleanup on exit
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
