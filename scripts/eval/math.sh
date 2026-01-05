eval "$(conda shell.bash hook)"
conda activate lm-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5

PROJECT_DIR="$(pwd)"
SAVE_PATH="${PROJECT_DIR}/data"
export HF_HOME=$SAVE_PATH
export VLLM_CACHE_ROOT=$SAVE_PATH/vllm_cache
export NCCL_P2P_DISABLE=1

MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
TP_SIZE=1
DP_SIZE=1
MAX_GEN_TOKENS=1024
ARGS="pretrained=$MODEL_NAME,\
tensor_parallel_size=$TP_SIZE,\
dtype=auto,\
gpu_memory_utilization=0.8,\
data_parallel_size=$DP_SIZE \
max_gen_toks=$MAX_GEN_TOKENS"

TASKS="gsm8k,hendrycks_math"
TASKS="hendrycks_math"
TASKS="leaderboard_math_hard"

LM_EVAL_ARGS="--model vllm \
              --model_args $ARGS \
              --tasks $TASKS \
              --batch_size auto \
              --apply_chat_template"

echo Running lm-eval with args:
echo $LM_EVAL_ARGS

python -m lm_eval $LM_EVAL_ARGS