# Reproducing the paper

There are two levels:

1. **Regenerate every figure and table from the released results** (one minute,
   CPU only). This checks that the numbers in the paper follow from the data.
2. **Rerun the experiments** that produced the data.

## 1. Figures and tables from `results/`

```bash
pip install -e ".[plot]"
bash experiments/reproduce/make_figures.sh     # writes figures/*.pdf (+ .png)
bash experiments/reproduce/make_tables.sh      # Table I, II, III and in-text numbers
```

`tests/test_paper_results.py` asserts the key numbers, so `pytest` fails if the
released data and the paper ever diverge.

Figures use LaTeX text rendering when a `latex` binary is available (as in the
paper) and matplotlib's built-in math text otherwise. Force either with
`RLSA_USETEX=1` or `RLSA_USETEX=0`.

## 2. Rerunning the experiments

Each recipe in `experiments/reproduce/` contains the exact paper configuration.
Outputs go to `runs/` unless `OUT=...` is set; plot them by pointing the plotting
scripts at the new directory (or set `OUT=results/...` to replace the released
data and rerun `make_figures.sh`).

| Recipe | Paper items | Hardware |
|---|---|---|
| `single_agent.sh` | training curves, cost sweep, Table II | CPU |
| `single_agent_qwen.sh` | Qwen-3B training curves, model-size sweep | 1 GPU, Qwen-2.5 3B/7B/14B |
| `auction_mock.sh` | Table I (mock + confidence gate), long horizon, N scaling, message drop, discount factor | CPU |
| `auction_qwen.sh` | Table I (Qwen-7B), heterogeneous bidders | 1 GPU, ~50 GB for the heterogeneous run |
| `fabric.sh` | FABRIC robustness, scaling, Table III | FABRIC account |

The auction recipes are split into steps; run a subset with
`STEPS="headline confgate" bash experiments/reproduce/auction_mock.sh`. The
`confgate` step needs the checkpoints written by `headline`.

### What "reproduce" means here

* The gating environment, the job streams and the simulated network are seeded.
  Baselines without a learned policy are reproduced exactly.
* PPO draws gate actions from PyTorch's global RNG, and in the auction the agent
  threads share one policy, so the order of those draws depends on thread
  scheduling. A rerun gives the same means and confidence intervals as the paper,
  not the same per-seed numbers.
* Real-LLM runs also depend on the GPU, the kernels and the model revision.

### Mapping

| Paper item | Data | Command |
|---|---|---|
| `fig_training_curves` | `single_agent/training/` | `python -m experiments.plots.plot_training results/single_agent/training/seed*.csv` |
| `fig_hf_training` | `single_agent/qwen_model_sweep/qwen2.5-3b/` | `python -m experiments.plots.plot_training .../qwen2.5-3b/seed*.csv -o figures/fig_hf_training.pdf` |
| `fig_model_sweep` | `single_agent/qwen_model_sweep/` | `python -m experiments.plots.plot_model_sweep` |
| `fig_cost_sweep` | `single_agent/cost_sweep/` | `python -m experiments.plots.plot_cost_sweep` |
| Table II | `single_agent/ablation/` | `python -m experiments.single_agent.ablation --summarize` |
| Table I, `fig_distributed_main` | `auction/headline_mock/`, `auction/headline_qwen7b/`, `auction/confidence_gate/` | `python -m experiments.auction.summarize --run-dir ...`, `python -m experiments.plots.plot_auction` |
| paired p-values | `auction/headline_mock/` | `python -m experiments.auction.paired_stats --run-dir results/auction/headline_mock` |
| `fig_scale_n`, `fig_robustness_drop` | `auction/scale_n/`, `auction/drop_retrain/` | `python -m experiments.plots.plot_auction` |
| frozen-policy drop numbers | `auction/drop_frozen/` | `python -m experiments.auction.summarize --run-dir results/auction/drop_frozen --prefix frozen_drop0.20_ --algos rlsa` |
| long-horizon numbers | `auction/long_horizon/` | `python -m experiments.auction.summarize --run-dir results/auction/long_horizon --prefix long_` |
| `fig_gamma_ablation` | `auction/gamma/` | `python -m experiments.plots.plot_gamma` |
| `fig_hetero_llm` | `auction/hetero_llm/` | `python -m experiments.plots.plot_hetero` |
| `fig_fabric_topology` | `data/us-states.geojson` | `python -m experiments.plots.plot_fabric_topology` |
| `fig_fabric_robustness`, `fig_fabric_scaling`, Table III | `fabric/` | `python -m experiments.plots.plot_fabric` |

All data paths are relative to `results/`. A description of the file formats is in
`results/README.md`.
