MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
GPUS_PER_MODEL=1
MODEL_REPLICAS=1
TASK=minerva_math
MAX_MODEL_LEN=2048
OUTPUT_PATH=./outputs
GPU=0
GPU_MEMORY_UTILIZATION=0.2

CMD="CUDA_VISIBLE_DEVICES=${GPU} python -m lm_eval --model vllm \
    --model_args pretrained=${MODEL_NAME},tensor_parallel_size=${GPUS_PER_MODEL},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},data_parallel_size=${MODEL_REPLICAS},max_model_len=${MAX_MODEL_LEN} \
    --batch_size auto \
    --tasks ${TASK} \
    --output_path ${OUTPUT_PATH} --log_samples --predict_only \
    --seed 42,42,42,42"
echo ${CMD}

# lm_eval --tasks list > tasks.txt