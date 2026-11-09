#!/usr/bin/env bash
# Regenerate every paper figure and table from the results in results/.
# No training, no GPU: takes about a minute.
#
#   bash experiments/reproduce/make_figures.sh            # writes figures/*.pdf
#   OUT=/some/dir bash experiments/reproduce/make_figures.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
OUT="${OUT:-figures}"
R=results
mkdir -p "$OUT"

echo "== Single-agent gating environment"
$PYTHON -m experiments.plots.plot_training $R/single_agent/training/seed*.csv -o "$OUT/fig_training_curves.pdf"
$PYTHON -m experiments.plots.plot_training $R/single_agent/qwen_model_sweep/qwen2.5-3b/seed*.csv -o "$OUT/fig_hf_training.pdf"
$PYTHON -m experiments.plots.plot_model_sweep --sweep-dir $R/single_agent/qwen_model_sweep -o "$OUT/fig_model_sweep.pdf"
$PYTHON -m experiments.plots.plot_cost_sweep $R/single_agent/cost_sweep/sweep.csv -o "$OUT/fig_cost_sweep.pdf"

echo "== Multi-agent auction"
$PYTHON -m experiments.plots.plot_auction --results $R/auction --out "$OUT"
$PYTHON -m experiments.plots.plot_gamma --results $R/auction/gamma --out "$OUT"
$PYTHON -m experiments.plots.plot_hetero --results $R/auction/hetero_llm --out "$OUT"

echo "== FABRIC"
$PYTHON -m experiments.plots.plot_fabric_topology --out "$OUT"
$PYTHON -m experiments.plots.plot_fabric --robustness-dir $R/fabric/robustness \
    --scaling-dir $R/fabric/scaling --fig-dir "$OUT"

echo
bash experiments/reproduce/make_tables.sh

# Refresh the PNGs used by the documentation site.
if [ "${DOCS:-0}" = "1" ]; then
  mkdir -p docs/assets/figures
  cp "$OUT"/*.png docs/assets/figures/
  echo "copied PNGs to docs/assets/figures/"
fi
