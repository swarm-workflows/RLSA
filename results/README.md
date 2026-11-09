# Released results

Raw outputs behind every number and figure in the paper. Each auction JSON is one
`(algorithm, N, seed)` cell from `experiments.auction.run_auction`; each FABRIC JSON
is one `(mode, algorithm, N, loss, seed)` cell from `experiments.fabric.run_fabric`.
See `experiments/README.md` for which paper item uses which directory.

```
single_agent/
  training/            seed{0..4}.csv           mock bidder, per-PPO-update metrics
  qwen_model_sweep/    qwen2.5-{3b,7b,14b}/seed{0..2}.csv
  cost_sweep/          sweep.csv                one row per (beta, seed)
  ablation/            ablation.csv             one row per (arm, seed)
auction/
  headline_mock/       <algo>_N4_seed{0..9}.json
    policies/          rlsa_seed*.pt            trained RLSA policies (state_dict)
                       residdata_seed*.npz      residual-head training data
                       residual_seed*.pt        confidence-gate residual heads
  headline_qwen7b/     <algo>_N4_seed{0..4}.json, policies/ (same layout, Qwen-7B)
  confidence_gate/     tau<T>_confidence_gate_N4_seed*.json
  gamma/               g<gamma>_rlsa_N4_seed*.json
  scale_n/             scale_<algo>_N<n>_seed*.json
  drop_retrain/        drop<p>_<algo>_N4_seed*.json
  drop_frozen/         train_p0_*, frozen_drop<p>_*, allLLMrand_drop<p>_*, policies/
  long_horizon/        long_<algo>_N4_seed*.json
  hetero_llm/          trueHetero_seed<rep>_<algo>_N4_seed0.json   (3 repeats of seed 0)
fabric/
  robustness/          robustness_<algo>_n4_loss<p>_s<seed>.json
  scaling/             scaling_<algo>_n<N>_loss0_s<seed>.json
```

Key fields of an auction JSON: `makespan`, `llm_calls_per_job`, `jain_fairness`,
`wins_per_agent`, `deadline_misses_per_agent`, `stale_bids`, `dropped_msgs`,
`per_model_analysis`, `distributed_config`. Some files record the relative path a
checkpoint had when it was written (`saved_policy_path`, `load_policy_path`); those
paths refer to the original run directory, the files themselves are in `policies/`.

The FABRIC study evaluated `auction/headline_mock/policies/rlsa_seed0.pt` frozen on
every agent.
