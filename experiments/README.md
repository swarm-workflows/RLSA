# Experiments

Everything here is run from the repository root as a module, e.g.
`python -m experiments.auction.run_auction --help`.

```
experiments/
  single_agent/   train.py, sweep_cost_beta.py, ablation.py      (controlled gating env)
  auction/        run_auction.py, train_residual_head.py,         (threaded multi-agent auction)
                  summarize.py, paired_stats.py
  fabric/         setup_slice.py, run_fabric.py, teardown.py      (FABRIC testbed)
  plots/          one script per paper figure + shared style.py
  reproduce/      shell recipes with the exact paper configurations
```

## Paper item -> data -> command

Figures are regenerated from `results/` with `bash experiments/reproduce/make_figures.sh`
and tables with `bash experiments/reproduce/make_tables.sh`. To rerun an experiment,
use the recipe in the last column (outputs go to `runs/` by default; set `OUT=results/...`
to overwrite the released data).

| Paper item | Released data in `results/` | Plot / table command | Experiment recipe |
|---|---|---|---|
| Training curves, mock bidder (`fig_training_curves`) | `single_agent/training/` | `plots.plot_training` | `single_agent.sh` |
| Training with Qwen-2.5-3B (`fig_hf_training`) | `single_agent/qwen_model_sweep/qwen2.5-3b/` | `plots.plot_training` | `single_agent_qwen.sh` |
| LLM size sweep (`fig_model_sweep`) | `single_agent/qwen_model_sweep/` | `plots.plot_model_sweep` | `single_agent_qwen.sh` |
| Cost-weight sweep (`fig_cost_sweep`) | `single_agent/cost_sweep/sweep.csv` | `plots.plot_cost_sweep` | `single_agent.sh` |
| Table II, ablation | `single_agent/ablation/ablation.csv` | `single_agent.ablation --summarize` | `single_agent.sh` |
| Table I, mock column (`fig_distributed_main`) | `auction/headline_mock/` | `auction.summarize`, `plots.plot_auction` | `auction_mock.sh` step `headline` |
| Table I, confidence gate | `auction/confidence_gate/` (tau=10 row) | `auction.summarize --prefix tau10_` | `auction_mock.sh` step `confgate` |
| Paired tests (p = 0.072 etc.) | `auction/headline_mock/` | `auction.paired_stats` | - |
| Table I, Qwen-7B column | `auction/headline_qwen7b/` | `auction.summarize` | `auction_qwen.sh` step `headline` |
| Long horizon (800 jobs, 1.03 LLM/job) | `auction/long_horizon/` | `auction.summarize --prefix long_` | `auction_mock.sh` step `long` |
| Scaling N (`fig_scale_n`) | `auction/scale_n/` | `plots.plot_auction` | `auction_mock.sh` step `scale` |
| Message drop, retrained (`fig_robustness_drop`) | `auction/drop_retrain/` | `plots.plot_auction` | `auction_mock.sh` step `drop_retrain` |
| Message drop, frozen policy (2.45 -> 1.60 LLM/job) | `auction/drop_frozen/` | `auction.summarize --prefix frozen_drop<p>_` | `auction_mock.sh` step `drop_frozen` |
| Discount factor (`fig_gamma_ablation`) | `auction/gamma/` | `plots.plot_gamma` | `auction_mock.sh` step `gamma` |
| Heterogeneous LLMs (`fig_hetero_llm`) | `auction/hetero_llm/` | `plots.plot_hetero` | `auction_qwen.sh` step `hetero` |
| FABRIC topology (`fig_fabric_topology`) | `data/us-states.geojson` | `plots.plot_fabric_topology` | - |
| FABRIC robustness (`fig_fabric_robustness`) | `fabric/robustness/` | `plots.plot_fabric` | `fabric.sh` step `robustness` |
| FABRIC scaling + Table III (`fig_fabric_scaling`) | `fabric/scaling/` | `plots.plot_fabric` | `fabric.sh` step `scaling` |

## Determinism

* The gating environment, job streams and network jitter are fully seeded.
* PPO samples gate actions from PyTorch's global RNG, and in the multi-agent
  auction several agent threads share one policy, so the order of those draws
  depends on thread scheduling. Rerunning an auction experiment therefore
  reproduces the paper's means and confidence intervals, not the exact per-seed
  numbers. Baselines without a learned policy (random, capacity,
  always-surrogate, always-LLM, oracle) are exactly reproducible.
* Real-LLM runs depend on GPU kernels and model versions as well.

## Hardware used for the paper

CPU experiments ran on NERSC Perlmutter CPU nodes, the Qwen-2.5 runs on a
single NVIDIA A100 (80 GB), and the wide-area study on a 17-VM FABRIC slice over
the STAR, MICH and MASS sites (VM size: the defaults of `fabric/setup_slice.py`).
