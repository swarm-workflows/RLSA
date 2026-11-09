# Getting started

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/swarm-workflows/RLSA.git
cd RLSA
pip install -e ".[plot]"
```

The core dependencies are `numpy`, `torch` and `gymnasium`. A CPU-only PyTorch
build is enough for every mock-bidder experiment:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

| Extra | Adds | Needed for |
|---|---|---|
| `plot` | matplotlib, pandas, scipy, seaborn | figures, tables, paired tests |
| `hf` | transformers, accelerate | real Qwen-2.5 bidders (GPU) |
| `fabric` | fabrictestbed-extensions | FABRIC testbed runs |
| `docs` | mkdocs-material, mkdocstrings | this site |
| `dev` | pytest, ruff | tests and lint |

Check the install:

```bash
pip install -e ".[plot,dev]"
pytest
```

## Train a gate in the controlled environment

```python
from rlsa.envs.gating_env import GatingEnv, GatingEnvConfig
from rlsa.ppo import PPOConfig, RLSATrainer
from rlsa.rewards import RewardWeights

env_cfg = GatingEnvConfig(reward_weights=RewardWeights(cost_beta=1.0))
trainer = RLSATrainer(
    env_fn=lambda: GatingEnv(cfg=env_cfg, seed=0),
    cfg=PPOConfig(obs_dim=env_cfg.obs_dim, rollout_length=512,
                  gamma=0.0, gae_lambda=0.0,          # bandit credit assignment
                  entropy_coef=0.02, value_coef=0.5),
)
trainer.train(total_timesteps=20_000, pretrain_steps=3_000, log_every=5,
              csv_log_path="runs/demo/seed0.csv")
```

The log prints the gate rate on easy (`g_easy`) and hard (`g_hard`) jobs; the
first should fall toward 0% and the second rise toward 100%. To use the trained
policy:

```python
import torch

env = GatingEnv(cfg=env_cfg, seed=1)
obs, _ = env.reset()
out = trainer.policy.act(torch.as_tensor(obs).unsqueeze(0), deterministic=True)
print(out["gate"].item(), out["bid"].item())
```

The environment follows the Gymnasium API: the action is
`{"bid": float in [0, 100], "gate": 0 or 1}` and `info` reports `is_hard_job`,
`score_error` and the reward breakdown.

## Run the multi-agent auction

```python
from rlsa.envs.auction import AuctionConfig
from rlsa.envs.distributed import DistributedConfig, run_distributed

cfg = DistributedConfig(
    base=AuctionConfig(n_agents=4, n_jobs=400),
    bid_deadline_ms=50, net_mean_ms=2.0, net_std_ms=0.5,
    pretrain_steps=80, sharing_mode="ctde",
)
for algo in ["always_llm", "rlsa"]:
    m = run_distributed(algo, cfg, seed=0)
    print(algo, round(m["makespan"]), m["llm_calls_per_job"], round(m["jain_fairness"], 3))
```

or from the command line, one JSON per cell:

```bash
python -m experiments.auction.run_auction --algos always_llm,rlsa --ns 4 --seeds 3 \
    --n-jobs 400 --out-dir runs/demo
python -m experiments.auction.summarize --run-dir runs/demo --algos always_llm,rlsa
```

## Use a real LLM bidder

```bash
pip install -e ".[hf]"
```

```python
from rlsa.bidders.hf_local import HFLocalBidder
from rlsa.envs.gating_env import GatingEnv, GatingEnvConfig

bidder = HFLocalBidder(model_name="Qwen/Qwen2.5-3B-Instruct", prompt_name="oneshot")
env_cfg = GatingEnvConfig(bidder=bidder)   # gate=1 now calls the model
```

In the auction, pass `bidder_kind="hf"` and `hf_model_name=...` to
`DistributedConfig` (CLI: `--bidder hf --hf-model ...`). All agents share one loaded
model per name. Any object with a `bid(agent_state, job, true_bid=None) -> BidResult`
method can serve as a bidder; see [`rlsa.bidders`](api.md#bidders).
