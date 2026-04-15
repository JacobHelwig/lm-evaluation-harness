#!/usr/bin/env bash
# One-time environment setup for running eval_code.sh.
# Creates the `lm_eval` conda env if missing, then installs deps + patches.
#
# Run once:  bash env.sh
set -euo pipefail

ENV_NAME="lm_eval"
PY_VERSION="3.12"

eval "$(conda shell.bash hook)"

# Create env if it doesn't exist.
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=${PY_VERSION}" pip
fi

conda activate "$ENV_NAME"

# --- lm-eval: install editable from this repo so upstream pulls take effect immediately.
# --- vLLM pinned to 0.10.2: later versions (0.19+) removed `swap_space` from EngineArgs,
#     and lm_eval still passes it. Pinning avoids a runtime patch.
uv pip install -e ".[vllm]" "vllm==0.10.2"
uv pip install transformers==4.57.3
# ray is required by vLLM's data_parallel_size > 1 path.
uv pip install "ray[default]"

echo "env.sh: done"
