#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/infer.sh <audio_or_directory_or_list> <output_dir> [gpu_id] [extra flags...]
if [ "$#" -lt 2 ]; then
    echo "Usage: bash scripts/infer.sh <audio_or_directory_or_list> <output_dir> [gpu_id] [extra flags...]" >&2
    exit 2
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Keep the caller's working directory so relative paths retain their meaning.
msvae_input="$1"
msvae_output_dir="$2"
shift 2

# Consume an optional GPU ID, leaving command-line flags for argparse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [ "$#" -gt 0 ] && [[ "$1" != -* || "$1" == "-1" ]]; then
    export CUDA_VISIBLE_DEVICES="${1:-${CUDA_VISIBLE_DEVICES}}"
    shift
fi

exec "${PYTHON_BIN:-python3}" -m src.infer.infer \
    --model-config "${MODEL_CONFIG:-${PROJECT_ROOT}/src/config/vae/config.yaml}" \
    --ckpt-path "${CKPT_PATH:-${PROJECT_ROOT}/model.pt}" \
    --input "${msvae_input}" \
    --output-dir "${msvae_output_dir}" \
    "$@"
