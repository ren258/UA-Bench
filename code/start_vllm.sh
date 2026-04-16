MODEL_PATH="Qwen/Qwen3-8B"

BASE_PORT=11546
LOG_DIR="logs"
PID_DIR="${LOG_DIR}/pids"
MODEL_NAME="my_model"

mkdir -p ${LOG_DIR}
mkdir -p ${PID_DIR}

rm -f ${PID_DIR}/*.pid

echo "Starting 8 vLLM instances..."

for GPU_ID in {0..7}; do
  PORT=$((BASE_PORT + GPU_ID))
  LOG_FILE="${LOG_DIR}/vllm_gpu${GPU_ID}.log"
  PID_FILE="${PID_DIR}/gpu${GPU_ID}.pid"

  echo "Launching GPU ${GPU_ID} on port ${PORT} ..."

  CUDA_VISIBLE_DEVICES=${GPU_ID} \
  python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --served-model-name ${MODEL_NAME}-gpu${GPU_ID} \
    > ${LOG_FILE} 2>&1 &

  echo $! > ${PID_FILE}
done

echo "All 8 vLLM instances launched."
echo "Logs directory: ${LOG_DIR}"
echo "PID directory: ${PID_DIR}"
