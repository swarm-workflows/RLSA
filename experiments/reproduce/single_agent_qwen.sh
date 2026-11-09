#!/usr/bin/env bash
# Single-agent gating environment with real Qwen-2.5 bidders (one GPU).
#   fig_hf_training (3B), fig_model_sweep (3B / 7B / 14B)
#
# Needs: pip install -e ".[hf]" and access to the Qwen/Qwen2.5-*-Instruct
# checkpoints on the HuggingFace Hub (set HF_HOME to choose the cache location;
# HF_HUB_OFFLINE=1 if the models are already cached). The 14B model needs about
# 30 GB of GPU memory in bf16.
#
#   bash experiments/reproduce/single_agent_qwen.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
OUT="${OUT:-runs/single_agent}"

$PYTHON -m experiments.single_agent.train --bidder hf \
    --models Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct \
    --prompt oneshot --seeds 3 --timesteps 4000 --rollout 256 --pretrain 1000 --log-every 4 \
    --out-dir "$OUT/qwen_model_sweep"
