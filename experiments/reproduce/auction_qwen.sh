#!/usr/bin/env bash
# Threaded auction with real Qwen-2.5 bidders (one GPU; all agents share the
# loaded model, the heterogeneous run loads 3B + 7B + 14B together, ~50 GB).
#   Table I (Qwen-7B column), fig_distributed_main (right half), fig_hetero_llm
#
# Needs: pip install -e ".[hf]" and the Qwen/Qwen2.5-*-Instruct checkpoints
# (set HF_HOME; HF_HUB_OFFLINE=1 if already cached).
#
#   bash experiments/reproduce/auction_qwen.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
OUT="${OUT:-runs/auction}"
STEPS="${STEPS:-headline hetero}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
RUN="$PYTHON -m experiments.auction.run_auction"
QWEN7B=Qwen/Qwen2.5-7B-Instruct

for step in $STEPS; do
case $step in

headline)  # N=4, 200 jobs, 5 seeds, 200 ms bid deadline.
  $RUN --algos rlsa --ns 4 --seeds 5 --n-jobs 200 --bidder hf --hf-model $QWEN7B \
      --pretrain-steps 40 --deadline-ms 200 \
      --save-policy-path "$OUT/headline_qwen7b/policies/rlsa_hf_seed{seed}.pt" \
      --dump-residual-data "$OUT/headline_qwen7b/policies/residdata_hf_seed{seed}.npz" \
      --out-dir "$OUT/headline_qwen7b"
  $RUN --algos always_llm,oracle --ns 4 --seeds 5 --n-jobs 200 --bidder hf --hf-model $QWEN7B \
      --pretrain-steps 40 --deadline-ms 200 --out-dir "$OUT/headline_qwen7b"
  ;;

hetero)  # Agents 0-3 = Qwen 3B / 7B / 14B / mock; 100 jobs; three repeats of seed 0.
  for algo in always_llm oracle rlsa; do
    for rep in 0 1 2; do
      $RUN --algos $algo --ns 4 --seeds 1 --n-jobs 100 --bidder hf \
          --bidder-kinds hf,hf,hf,mock \
          --hf-models-per-agent Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct,none \
          --pretrain-steps 20 --deadline-ms 200 \
          --prefix "trueHetero_seed${rep}_" --out-dir "$OUT/hetero_llm"
    done
  done
  ;;

*) echo "unknown step: $step" >&2; exit 1 ;;
esac
done
