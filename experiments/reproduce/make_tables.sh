#!/usr/bin/env bash
# Print every table and in-text number of the paper from results/.
#
#   bash experiments/reproduce/make_tables.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
R=results/auction

echo "== Table I (mock): distributed auction, N=4, 400 jobs, 10 seeds"
$PYTHON -m experiments.auction.summarize --run-dir $R/headline_mock \
    --algos random,capacity,always_surrogate,always_llm,oracle,rlsa
echo "-- confidence gate at the matched budget (tau=10)"
$PYTHON -m experiments.auction.summarize --run-dir $R/confidence_gate --algos confidence_gate --prefix tau10_
echo "-- paired tests"
$PYTHON -m experiments.auction.paired_stats --run-dir $R/headline_mock --ref rlsa --others always_llm
$PYTHON -m experiments.auction.paired_stats --run-dir $R/headline_mock --ref rlsa \
    --others confidence_gate --other-dir $R/confidence_gate --other-prefix tau10_

echo
echo "== Table I (real Qwen-2.5-7B): N=4, 200 jobs, 5 seeds"
$PYTHON -m experiments.auction.summarize --run-dir $R/headline_qwen7b --algos always_llm,oracle,rlsa

echo
echo "== Table II: ablation in the gating environment"
$PYTHON -m experiments.single_agent.ablation --summarize --out results/single_agent/ablation/ablation.csv

echo
echo "== Long horizon (N=4, 800 jobs)"
$PYTHON -m experiments.auction.summarize --run-dir $R/long_horizon --prefix long_ --algos always_llm,oracle,rlsa

echo
echo "== Discount factor in the coupled auction"
for G in 0.0 0.95 0.99; do
  echo "-- gamma=$G"
  $PYTHON -m experiments.auction.summarize --run-dir $R/gamma --prefix "g${G}_" --algos rlsa
done

echo
echo "== Message drop: retrained at each level (left) and frozen p=0 policy (right)"
for D in 0.0 0.05 0.10 0.20; do
  echo "-- drop=$D retrained"
  $PYTHON -m experiments.auction.summarize --run-dir $R/drop_retrain --prefix "drop${D}_" --algos rlsa,always_llm
  echo "-- drop=$D frozen"
  $PYTHON -m experiments.auction.summarize --run-dir $R/drop_frozen --prefix "frozen_drop${D}_" --algos rlsa
done

echo
echo "== Table III: FABRIC scaling"
$PYTHON -m experiments.plots.plot_fabric --table-only --scaling-dir results/fabric/scaling
