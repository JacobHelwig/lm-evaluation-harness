#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate lm_eval
export PATH=$CONDA_PREFIX/bin:$PATH
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,3,4,5
export DATA_PATH=$PWD/../verlData
export HF_HOME=$DATA_PATH
export VLLM_CACHE_DIR=$DATA_PATH/vllm_cache
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

set -xeuo pipefail

############################ Quick Config ############################

MODEL="Qwen/Qwen3-0.6B"
TASKS="humaneval,mbpp"
NUM_FEWSHOT=0
MAX_GEN_TOKS=1024
BACKEND="vllm"   # or "hf"

# Multi-GPU notes:
#   - TP (tensor parallel): only needed when the model doesn't fit on one GPU; adds
#     per-layer all-reduce overhead, so it SLOWS small models down.
#   - DP (data parallel, replicate+shard): would be the throughput win for small models,
#     but vLLM 0.19 disallows offline DP for dense models. For now stick to 1 GPU.
TP_SIZE=1
DP_SIZE=4

OUTPUT_DIR="$PWD/eval_outputs/$(echo "$MODEL" | tr '/' '_')_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

############################ Launch ############################
# --log_samples writes per-example model inputs + outputs to
#   $OUTPUT_DIR/<model>/samples_<task>_*.jsonl
# Each record has a "arguments" / "resps" field containing the exact prompt
# sent to the model and the generated completion.

if [[ "$BACKEND" == "vllm" ]]; then
    MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=8192,tensor_parallel_size=${TP_SIZE},data_parallel_size=${DP_SIZE},enforce_eager=True"
else
    MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16"
fi

lm-eval run \
    --model "$BACKEND" \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --num_fewshot "$NUM_FEWSHOT" \
    --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}" \
    --batch_size auto \
    --output_path "$OUTPUT_DIR" \
    --log_samples \
    --write_out \
    --confirm_run_unsafe_code \
    --limit 16
    "$@"

echo "Done. Per-sample inputs/outputs saved under: $OUTPUT_DIR"

############################ Show first few prompts + responses ############################
N_PREVIEW=${N_PREVIEW:-2}
PREVIEW_FILE="$OUTPUT_DIR/preview.txt"
python - "$OUTPUT_DIR" "$N_PREVIEW" <<'PY' 2>&1 | tee "$PREVIEW_FILE"
import glob, json, sys, os, textwrap

out_dir, n = sys.argv[1], int(sys.argv[2])
for task_file in sorted(glob.glob(f"{out_dir}/**/samples_*.jsonl", recursive=True)):
    task = os.path.basename(task_file).split("_")[1]
    print("\n" + "=" * 80)
    print(f"TASK: {task}   (showing first {n} examples from {task_file})")
    print("=" * 80)
    with open(task_file) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            d = json.loads(line)
            prompt = d["arguments"]["gen_args_0"]["arg_0"]
            resp = d["resps"][0][0] if d.get("resps") else ""
            metric = {k: v for k, v in d.get("metrics", {}).items()} if isinstance(d.get("metrics"), dict) else d.get("metrics")
            score = d.get("pass_at_1", d.get("pass@1"))
            print(f"\n--- doc_id={d.get('doc_id')}  score={score} ---")
            print("PROMPT:")
            print(textwrap.indent(prompt.rstrip(), "  | "))
            print("RESPONSE:")
            print(textwrap.indent(resp.rstrip(), "  > "))
PY
