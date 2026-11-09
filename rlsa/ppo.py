"""PPO trainer for the single-agent gating environment.

The gate is the RL-controlled Bernoulli action; the bid comes from the surrogate
head. Whenever the gate calls the LLM, the returned bid is also used as a
supervision target for the surrogate (auxiliary teacher loss), and an optional
imitation pretraining stage warm-starts the surrogate before PPO.

Example::

    from rlsa.envs.gating_env import GatingEnv, GatingEnvConfig
    from rlsa.ppo import PPOConfig, RLSATrainer

    env_cfg = GatingEnvConfig()
    trainer = RLSATrainer(
        env_fn=lambda: GatingEnv(cfg=env_cfg, seed=0),
        cfg=PPOConfig(obs_dim=env_cfg.obs_dim, gamma=0.0, gae_lambda=0.0),
    )
    trainer.train(total_timesteps=20_000, pretrain_steps=3_000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rlsa.gating.policy import GatingConfig, GatingPolicy


@dataclass
class PPOConfig:
    obs_dim: int = 20
    hidden_dim: int = 128
    rollout_length: int = 256
    update_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    # Auxiliary teacher loss: when the LLM is invoked its bid supervises the
    # surrogate head (masked MSE). Without it the surrogate gets no bid signal.
    surrogate_imitation_coef: float = 1.0
    max_grad_norm: float = 0.5
    device: str = "cpu"


@dataclass
class RolloutBuffer:
    obs: List[np.ndarray] = field(default_factory=list)
    actions_gate: List[int] = field(default_factory=list)
    actions_bid: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    # Teacher targets for the surrogate bidder: when gate=1, the LLM's bid
    # acts as a label for an MSE auxiliary loss. When gate=0, we have no
    # ground truth, so we mask the loss out (teacher_mask=0).
    teacher_bid: List[float] = field(default_factory=list)
    teacher_mask: List[float] = field(default_factory=list)

    def clear(self) -> None:
        for f in (
            self.obs,
            self.actions_gate,
            self.actions_bid,
            self.log_probs,
            self.rewards,
            self.values,
            self.dones,
            self.teacher_bid,
            self.teacher_mask,
        ):
            f.clear()

    def __len__(self) -> int:
        return len(self.obs)


class RLSATrainer:
    """A compact PPO trainer specialized for the RLSA gating policy."""

    def __init__(
        self,
        env_fn: Callable[[], Any],
        cfg: Optional[PPOConfig] = None,
    ):
        self.cfg = cfg or PPOConfig()
        self.env = env_fn()
        self.policy = GatingPolicy(
            GatingConfig(obs_dim=self.cfg.obs_dim, hidden_dim=self.cfg.hidden_dim)
        ).to(self.cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate)
        self.buffer = RolloutBuffer()
        self.global_step = 0
        self.episode_returns: List[float] = []
        self._ep_return = 0.0

    # ---------------------------------------------------------------- rollout

    def collect_rollout(self) -> Dict[str, float]:
        obs, _ = self.env.reset()
        rollout_return = 0.0
        n_episodes = 0
        # Per-bucket counters so the caller can plot easy/hard gate-rate
        # evolution without rerunning evaluation between updates.
        n_easy, n_hard, g_easy, g_hard = 0, 0, 0, 0
        err_easy_sum, err_hard_sum = 0.0, 0.0
        # Reward-component accumulators so the caller can plot the actual
        # makespan / utilization / LLM-cost contributions as training
        # progresses (rather than the opaque PPO loss numbers).
        r_make_sum, r_util_sum, r_cost_sum = 0.0, 0.0, 0.0
        r_breakdown_n = 0

        for _ in range(self.cfg.rollout_length):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
            with torch.no_grad():
                bid, p_gate, value, _ = self.policy(obs_t)
                gate_dist = torch.distributions.Bernoulli(p_gate)
                gate = gate_dist.sample()
                log_prob = gate_dist.log_prob(gate)

            action = {"bid": float(bid.item()), "gate": int(gate.item())}
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)

            # Capture the teacher signal: when the LLM was invoked, its bid
            # (returned by the env as info["effective_bid"]) is a free label
            # for the surrogate head's imitation loss.
            if action["gate"] == 1:
                teacher = float(info.get("effective_bid", 0.0))
                mask = 1.0
            else:
                teacher = 0.0
                mask = 0.0

            self.buffer.obs.append(obs.copy())
            self.buffer.actions_gate.append(int(gate.item()))
            self.buffer.actions_bid.append(float(bid.item()))
            self.buffer.log_probs.append(float(log_prob.item()))
            self.buffer.rewards.append(float(reward))
            self.buffer.values.append(float(value.item()))
            self.buffer.dones.append(done)

            # Bucket stats — only meaningful if env exposes is_hard_job
            if isinstance(info, dict) and "is_hard_job" in info:
                if info["is_hard_job"]:
                    n_hard += 1
                    g_hard += int(gate.item())
                    err_hard_sum += float(info.get("score_error", 0.0))
                else:
                    n_easy += 1
                    g_easy += int(gate.item())
                    err_easy_sum += float(info.get("score_error", 0.0))

            # Reward decomposition (per-step contributions)
            br = info.get("reward_breakdown") if isinstance(info, dict) else None
            if br is not None:
                r_make_sum += float(br.get("r_make", 0.0))
                r_util_sum += float(br.get("r_util", 0.0))
                r_cost_sum += float(br.get("r_cost", 0.0))
                r_breakdown_n += 1
            self.buffer.teacher_bid.append(teacher)
            self.buffer.teacher_mask.append(mask)

            rollout_return += float(reward)
            self._ep_return += float(reward)
            self.global_step += 1

            obs = next_obs
            if done:
                self.episode_returns.append(self._ep_return)
                self._ep_return = 0.0
                n_episodes += 1
                obs, _ = self.env.reset()

        # bootstrap value for the last state
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
            _, _, last_value, _ = self.policy(obs_t)
            self._last_value = float(last_value.item())

        # Mean per-step reward contributions (raw R_*, before lambda weighting)
        n_br = max(1, r_breakdown_n)
        return {
            "rollout_return": rollout_return,
            "rollout_episodes": n_episodes,
            "n_easy": n_easy,
            "n_hard": n_hard,
            "gate_easy": float(g_easy / n_easy) if n_easy else 0.0,
            "gate_hard": float(g_hard / n_hard) if n_hard else 0.0,
            "gate_overall": float((g_easy + g_hard) / max(1, n_easy + n_hard)),
            "score_err_easy": float(err_easy_sum / n_easy) if n_easy else 0.0,
            "score_err_hard": float(err_hard_sum / n_hard) if n_hard else 0.0,
            "r_make_mean": r_make_sum / n_br,
            "r_util_mean": r_util_sum / n_br,
            "r_cost_mean": r_cost_sum / n_br,
        }

    # ----------------------------------------------------------------- update

    def _gae(self) -> np.ndarray:
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        values = np.asarray(self.buffer.values, dtype=np.float32)
        dones = np.asarray(self.buffer.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        next_value = self._last_value
        next_non_terminal = 1.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * next_value * next_non_terminal - values[t]
            last_gae = (
                delta
                + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_gae
            )
            advantages[t] = last_gae
            next_value = values[t]
            next_non_terminal = 1.0 - dones[t]
        return advantages

    def update(self) -> Dict[str, float]:
        advantages = self._gae()
        returns = advantages + np.asarray(self.buffer.values, dtype=np.float32)

        # Normalize advantages
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.cfg.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        obs_t = torch.as_tensor(np.asarray(self.buffer.obs), dtype=torch.float32, device=self.cfg.device)
        gate_t = torch.as_tensor(self.buffer.actions_gate, dtype=torch.float32, device=self.cfg.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs, dtype=torch.float32, device=self.cfg.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.cfg.device)
        teacher_bid_t = torch.as_tensor(self.buffer.teacher_bid, dtype=torch.float32, device=self.cfg.device)
        teacher_mask_t = torch.as_tensor(self.buffer.teacher_mask, dtype=torch.float32, device=self.cfg.device)

        n = obs_t.shape[0]
        indices = np.arange(n)
        loss_log = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "imitation": 0.0}

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb = indices[start:end]
                if len(mb) == 0:
                    continue

                # Full forward: we need both gate log-prob and surrogate bid.
                bid_pred, p_gate, value, _ = self.policy(obs_t[mb])
                eps = 1e-8
                log_p = (
                    gate_t[mb] * torch.log(p_gate + eps)
                    + (1.0 - gate_t[mb]) * torch.log(1.0 - p_gate + eps)
                )
                entropy = -(
                    p_gate * torch.log(p_gate + eps)
                    + (1.0 - p_gate) * torch.log(1.0 - p_gate + eps)
                )

                ratio = torch.exp(log_p - old_log_probs[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon)
                    * adv_t[mb]
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (value - returns_t[mb]).pow(2).mean()
                entropy_bonus = entropy.mean()

                # Surrogate imitation: MSE against the LLM teacher, masked to
                # the gate=1 samples. Normalize by bid scale (100) so the loss
                # magnitude is comparable to the policy / value terms.
                mask = teacher_mask_t[mb]
                if mask.sum() > 0:
                    sq_err = ((bid_pred - teacher_bid_t[mb]) / 100.0) ** 2
                    imitation_loss = (sq_err * mask).sum() / (mask.sum() + 1e-8)
                else:
                    imitation_loss = torch.tensor(0.0, device=self.cfg.device)

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy_bonus
                    + self.cfg.surrogate_imitation_coef * imitation_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                loss_log["policy"] += float(policy_loss.item())
                loss_log["value"] += float(value_loss.item())
                loss_log["entropy"] += float(entropy_bonus.item())
                loss_log["imitation"] += float(imitation_loss.item())

        self.buffer.clear()
        return loss_log

    # ----------------------------------------------------- imitation pretrain

    def pretrain_surrogate(
        self,
        n_steps: int = 2000,
        lr: float = 1e-3,
    ) -> None:
        """Run with gate=1 always, collect (obs, llm_bid) pairs, and train the
        surrogate head + encoder to imitate the LLM. This bootstraps a useful
        surrogate before PPO has to decide whether to trust it. Without this
        step PPO sees "gate=0 always gives a bad bid" and never explores the
        gate=0 path.
        """
        pre_opt = optim.Adam(self.policy.parameters(), lr=lr)
        obs, _ = self.env.reset()
        obs_buf, target_buf = [], []
        # Collect rollouts using gate=1 only
        for _ in range(n_steps):
            action = {"bid": 50.0, "gate": 1}  # bid doesn't matter when gated
            next_obs, _, term, trunc, info = self.env.step(action)
            obs_buf.append(obs.copy())
            target_buf.append(float(info["effective_bid"]))
            obs = next_obs
            if term or trunc:
                obs, _ = self.env.reset()
        obs_t = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=self.cfg.device)
        tgt_t = torch.as_tensor(target_buf, dtype=torch.float32, device=self.cfg.device)
        # 100 epochs of supervised regression
        for epoch in range(100):
            idx = np.random.permutation(len(obs_buf))
            losses = []
            for s in range(0, len(idx), 128):
                mb = idx[s : s + 128]
                bid_pred, _, _, _ = self.policy(obs_t[mb])
                loss = ((bid_pred - tgt_t[mb]) / 100.0).pow(2).mean()
                pre_opt.zero_grad()
                loss.backward()
                pre_opt.step()
                losses.append(loss.item())
            if (epoch + 1) % 25 == 0:
                print(f"  [pretrain ep {epoch+1:3d}] MSE={np.mean(losses):.5f}")
        print("  Pretrain done.")

    # ---------------------------------------------------------------- driver

    def train(
        self,
        total_timesteps: int = 10_000,
        log_every: int = 1,
        pretrain_steps: int = 0,
        csv_log_path: Optional[str] = None,
    ) -> None:
        """Run training. If `csv_log_path` is given, write one row per PPO
        update with rollout stats, loss components, and per-bucket gate
        rates (see ``experiments/plots/plot_training.py``).
        """
        if pretrain_steps > 0:
            print(f"--- Imitation pretrain: {pretrain_steps} env steps ---")
            self.pretrain_surrogate(n_steps=pretrain_steps)
            print(f"--- PPO training: {total_timesteps} env steps ---")

        csv_file = None
        csv_writer = None
        if csv_log_path:
            import csv as _csv
            from pathlib import Path as _Path
            _Path(csv_log_path).parent.mkdir(parents=True, exist_ok=True)
            csv_file = open(csv_log_path, "w", newline="")
            fieldnames = [
                "update", "global_step",
                "rollout_return", "ep_return_10",
                "gate_overall", "gate_easy", "gate_hard",
                "score_err_easy", "score_err_hard",
                "r_make_mean", "r_util_mean", "r_cost_mean",
                "loss_policy", "loss_value", "entropy", "loss_imitation",
            ]
            csv_writer = _csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

        n_updates = total_timesteps // self.cfg.rollout_length
        for i in range(n_updates):
            rollout_stats = self.collect_rollout()
            loss_stats = self.update()
            recent = (
                float(np.mean(self.episode_returns[-10:]))
                if self.episode_returns
                else float("nan")
            )
            if csv_writer is not None:
                csv_writer.writerow({
                    "update": i + 1,
                    "global_step": self.global_step,
                    "rollout_return": rollout_stats["rollout_return"],
                    "ep_return_10": recent,
                    "gate_overall": rollout_stats["gate_overall"],
                    "gate_easy": rollout_stats["gate_easy"],
                    "gate_hard": rollout_stats["gate_hard"],
                    "score_err_easy": rollout_stats["score_err_easy"],
                    "score_err_hard": rollout_stats["score_err_hard"],
                    "r_make_mean": rollout_stats["r_make_mean"],
                    "r_util_mean": rollout_stats["r_util_mean"],
                    "r_cost_mean": rollout_stats["r_cost_mean"],
                    "loss_policy": loss_stats["policy"],
                    "loss_value": loss_stats["value"],
                    "entropy": loss_stats["entropy"],
                    "loss_imitation": loss_stats["imitation"],
                })
                csv_file.flush()
            if (i + 1) % log_every == 0:
                print(
                    f"[step {self.global_step:>7}] "
                    f"ep_R10={recent:+8.3f} "
                    f"roll_R={rollout_stats['rollout_return']:+8.3f} "
                    f"g_easy={100*rollout_stats['gate_easy']:5.1f}% "
                    f"g_hard={100*rollout_stats['gate_hard']:5.1f}% "
                    f"pol={loss_stats['policy']:+6.3f} "
                    f"val={loss_stats['value']:+7.3f} "
                    f"imit={loss_stats['imitation']:+6.4f}"
                )

        if csv_file is not None:
            csv_file.close()
            print(f"[csv] wrote {csv_log_path}")
