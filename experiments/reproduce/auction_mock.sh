#!/usr/bin/env bash
# Threaded asynchronous auction with the mock bidder (CPU only).
#   Table I (mock column + confidence gate), fig_distributed_main (mock),
#   fig_scale_n, fig_robustness_drop, fig_gamma_ablation, long-horizon numbers.
#
#   bash experiments/reproduce/auction_mock.sh             # writes to runs/auction
#   STEPS="headline confgate" bash experiments/reproduce/auction_mock.sh
#
# Runtime on one CPU node: a few hours in total; the headline step alone is
# about one hour. Threads sleep for the simulated network delay, so wall-clock
# time does not shrink with more cores.
#
# Note: the auction runs agents as threads that share one policy, and PPO samples
# from torch's global RNG, so a rerun matches the paper statistically (means and
# confidence intervals) rather than bit for bit.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
OUT="${OUT:-runs/auction}"
STEPS="${STEPS:-headline confgate gamma scale drop_retrain drop_frozen long}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
RUN="$PYTHON -m experiments.auction.run_auction"
ALL=random,capacity,always_surrogate,always_llm,oracle,rlsa

for step in $STEPS; do
case $step in

headline)  # Table I mock column: N=4, 400 jobs, 10 seeds. Saves checkpoints for confgate/FABRIC.
  $RUN --algos $ALL --ns 4 --seeds 10 --n-jobs 400 --pretrain-steps 80 \
      --save-policy-path "$OUT/headline_mock/policies/rlsa_seed{seed}.pt" \
      --dump-residual-data "$OUT/headline_mock/policies/residdata_seed{seed}.npz" \
      --out-dir "$OUT/headline_mock"
  ;;

confgate)  # Confidence gate on the frozen headline policies; tau sweep (tau=10 matches RLSA's budget).
  for S in $(seq 0 9); do
    $PYTHON -m experiments.auction.train_residual_head \
        --data "$OUT/headline_mock/policies/residdata_seed${S}.npz" \
        --policy "$OUT/headline_mock/policies/rlsa_seed${S}.pt" \
        --out "$OUT/headline_mock/policies/residual_seed${S}.pt"
  done
  for TAU in 2 4 6 8 10 12 16 20; do
    $RUN --algos confidence_gate --ns 4 --seeds 10 --n-jobs 400 --pretrain-steps 0 \
        --load-policy-path "$OUT/headline_mock/policies/rlsa_seed{seed}.pt" \
        --residual-path "$OUT/headline_mock/policies/residual_seed{seed}.pt" \
        --confidence-tau $TAU --prefix "tau${TAU}_" --out-dir "$OUT/confidence_gate"
  done
  ;;

gamma)  # Discount factor in the coupled auction.
  for G in 0.0 0.95 0.99; do
    $RUN --algos rlsa --ns 4 --seeds 10 --n-jobs 400 --pretrain-steps 80 \
        --gamma $G --gae-lambda 0.95 --prefix "g${G}_" --out-dir "$OUT/gamma"
  done
  ;;

scale)  # N in {1,2,4,8,16}, 300 jobs, 3 seeds.
  $RUN --algos rlsa,always_llm,always_surrogate --ns 1,2,4,8,16 --seeds 3 --n-jobs 300 \
      --pretrain-steps 60 --prefix scale_ --out-dir "$OUT/scale_n"
  ;;

drop_retrain)  # Message drop, RLSA retrained at each level. Slower, noisier network (5 +- 2 ms).
  for D in 0.0 0.05 0.10 0.20; do
    $RUN --algos rlsa,always_llm --ns 4 --seeds 3 --n-jobs 200 --pretrain-steps 40 \
        --net-mean-ms 5.0 --net-std-ms 2.0 --net-drop $D --prefix "drop${D}_" \
        --out-dir "$OUT/drop_retrain"
  done
  ;;

drop_frozen)  # Train at drop=0, freeze, evaluate at each drop level.
  $RUN --algos rlsa --ns 4 --seeds 3 --n-jobs 200 --pretrain-steps 40 \
      --net-mean-ms 5.0 --net-std-ms 2.0 --net-drop 0.0 \
      --save-policy-path "$OUT/drop_frozen/policies/rlsa_seed{seed}.pt" \
      --prefix train_p0_ --out-dir "$OUT/drop_frozen"
  for D in 0.0 0.05 0.10 0.20; do
    $RUN --algos rlsa --ns 4 --seeds 3 --n-jobs 200 --pretrain-steps 0 \
        --net-mean-ms 5.0 --net-std-ms 2.0 --net-drop $D \
        --load-policy-path "$OUT/drop_frozen/policies/rlsa_seed{seed}.pt" \
        --prefix "frozen_drop${D}_" --out-dir "$OUT/drop_frozen"
    $RUN --algos always_llm --ns 4 --seeds 3 --n-jobs 200 --pretrain-steps 0 \
        --net-mean-ms 5.0 --net-std-ms 2.0 --net-drop $D --tiebreak random \
        --prefix "allLLMrand_drop${D}_" --out-dir "$OUT/drop_frozen"
  done
  ;;

long)  # Longer horizon: 800 jobs.
  $RUN --algos rlsa,always_llm,oracle --ns 4 --seeds 3 --n-jobs 800 --pretrain-steps 80 \
      --prefix long_ --out-dir "$OUT/long_horizon"
  ;;

*) echo "unknown step: $step" >&2; exit 1 ;;
esac
done
