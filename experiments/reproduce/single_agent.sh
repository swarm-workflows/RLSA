#!/usr/bin/env bash
# Single-agent gating environment with the mock oracle bidder (CPU only).
#   fig_training_curves, fig_cost_sweep, Table II (ablation)
#
#   bash experiments/reproduce/single_agent.sh            # writes to runs/single_agent
#   OUT=results/single_agent bash experiments/reproduce/single_agent.sh   # overwrite released results
#
# Runtime on one CPU node: roughly 1-2 hours in total.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
OUT="${OUT:-runs/single_agent}"

# Training curves: 5 seeds, 20k PPO steps after 3k imitation-pretraining steps.
$PYTHON -m experiments.single_agent.train --seeds 5 --timesteps 20000 --rollout 512 \
    --pretrain 3000 --out-dir "$OUT/training"

# Cost-weight sweep: 7 betas x 3 seeds.
$PYTHON -m experiments.single_agent.sweep_cost_beta --betas 0,0.25,0.5,1,2,4,8 --seeds 3 \
    --timesteps 12000 --pretrain 2000 --out "$OUT/cost_sweep/sweep.csv"

# Ablation: 4 arms x 3 seeds.
$PYTHON -m experiments.single_agent.ablation --seeds 3 --timesteps 12000 --pretrain 2000 \
    --out "$OUT/ablation/ablation.csv"
