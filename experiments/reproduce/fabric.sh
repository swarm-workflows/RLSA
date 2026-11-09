#!/usr/bin/env bash
# Real wide-area deployment on the FABRIC testbed (paper: 17 VMs over 3 sites).
#   fig_fabric_robustness, fig_fabric_scaling, Table III
#
# Needs a FABRIC account (https://portal.fabric-testbed.net), configured fablib
# credentials, and: pip install -e ".[fabric]"
#
# The slice CONSUMES allocation while it exists; the last step deletes it.
# Run the steps one by one the first time:
#
#   STEPS=check bash experiments/reproduce/fabric.sh       # local TCP dry run, no FABRIC
#   STEPS=slice bash experiments/reproduce/fabric.sh       # provision 1 + 16 VMs
#   STEPS="robustness scaling" bash experiments/reproduce/fabric.sh
#   STEPS=teardown bash experiments/reproduce/fabric.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
OUT="${OUT:-runs/fabric}"
STEPS="${STEPS:-check}"
# Frozen policy evaluated by every agent (the paper used headline seed 0).
CKPT="${CKPT:-results/auction/headline_mock/policies/rlsa_seed0.pt}"

for step in $STEPS; do
case $step in

check)  # Coordinator + 4 agents over 127.0.0.1 with the real wire protocol.
  $PYTHON -m rlsa.fabric.local --algo rlsa --n-agents 4 --n-jobs 50 --policy-path "$CKPT"
  ;;

slice)
  $PYTHON -m experiments.fabric.setup_slice --n-agents 16 --n-sites 3 --yes
  $PYTHON -m experiments.fabric.setup_slice --pack
  ;;

robustness)  # N=4, tc-netem loss in {0,5,10,20}%, 3 seeds. --provision installs code + checkpoint.
  $PYTHON -m experiments.fabric.run_fabric --mode robustness --provision --ckpt "$CKPT" \
      --algos rlsa always_llm --ns 4 --netem-loss 0 5 10 20 --seeds 3 --n-jobs 400 \
      --out-dir "$OUT/robustness"
  ;;

scaling)  # N in {4,8,16}, no injected loss, 3 seeds.
  $PYTHON -m experiments.fabric.run_fabric --mode scaling --ckpt "$CKPT" \
      --algos rlsa always_llm --ns 4 8 16 --seeds 3 --n-jobs 400 \
      --out-dir "$OUT/scaling"
  ;;

teardown)
  $PYTHON -m experiments.fabric.teardown
  ;;

*) echo "unknown step: $step" >&2; exit 1 ;;
esac
done
