"""Controlled single-agent gating environment (paper Sec. V-A).

Every step is one job. The agent emits a dual action ``{bid, gate}``:

* ``gate=0``: the surrogate ``bid`` is used, corrupted by noise whose scale grows
  with the (hidden) job difficulty, so the surrogate is accurate on easy jobs and
  unreliable on hard ones;
* ``gate=1``: the LLM bidder answers instead. By default the built-in oracle
  returns the true bid after a Gaussian latency; a real :class:`~rlsa.bidders.Bidder`
  (for example :class:`~rlsa.bidders.hf_local.HFLocalBidder`) can be attached via
  ``GatingEnvConfig.bidder``.

A bid error delays the job quadratically, which drives ``R_make``; each LLM call
pays ``R_cost``. The policy has to learn to gate only when difficulty is high.

Observation layout (``obs_dim`` floats in [0, 1])::

    obs[0:4]  agent utilization (cpu, gpu, ram, disk)
    obs[4:8]  job resource request
    obs[8]    noisy difficulty signal
    obs[9:]   filler features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlsa.bidders.base import AgentState, Bidder, JobSpec
from rlsa.rewards import RewardWeights, composite_reward


@dataclass
class GatingEnvConfig:
    obs_dim: int = 20
    bid_min: float = 0.0
    bid_max: float = 100.0
    max_steps_per_episode: int = 100

    # Built-in oracle LLM latency.
    llm_latency_seconds_mean: float = 2.0
    llm_latency_seconds_std: float = 0.5

    # Job difficulty mixture.
    hard_job_prob: float = 0.3
    easy_difficulty: float = 0.1
    hard_difficulty: float = 0.9

    # Surrogate noise std (bid units) interpolates from easy to hard with difficulty.
    surrogate_noise_easy: float = 2.0
    surrogate_noise_hard: float = 55.0

    # A bid error E (bid units) costs error_scale * (E / 100) ** error_exponent
    # extra seconds on a 60 s job. Tuned so that on hard jobs the expected
    # surrogate penalty clearly exceeds the LLM cost, and on easy jobs it does not.
    error_scale: float = 800.0
    error_exponent: float = 2.0

    reward_weights: Optional[RewardWeights] = None

    # Optional real bidder for gate=1 calls; ``None`` uses the built-in oracle.
    bidder: Optional[Bidder] = None

    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = RewardWeights()


class GatingEnv(gym.Env):
    """Gymnasium environment for learning when to call the LLM."""

    metadata = {"render_modes": []}

    _SLOT_UTIL = slice(0, 4)
    _SLOT_REQ = slice(4, 8)
    _SLOT_DIFFICULTY = 8

    def __init__(self, cfg: Optional[GatingEnvConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or GatingEnvConfig()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.cfg.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Dict(
            {
                "bid": spaces.Box(
                    low=self.cfg.bid_min, high=self.cfg.bid_max, shape=(), dtype=np.float32
                ),
                "gate": spaces.Discrete(2),
            }
        )
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._obs: Optional[np.ndarray] = None
        self._difficulty: float = 0.0
        self._true_bid: float = 0.0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._obs = self._sample_obs()
        return self._obs, {}

    # -- job sampling ---------------------------------------------------------

    def _sample_obs(self) -> np.ndarray:
        obs = self._rng.uniform(0.0, 1.0, size=(self.cfg.obs_dim,)).astype(np.float32)
        is_hard = self._rng.random() < self.cfg.hard_job_prob
        self._difficulty = self.cfg.hard_difficulty if is_hard else self.cfg.easy_difficulty
        # The optimal bid is high when the job asks for little.
        req_mean = float(obs[self._SLOT_REQ].mean())
        self._true_bid = (1.0 - req_mean) * 100.0
        # Difficulty is observable, but only through a noisy channel.
        noisy_difficulty = self._difficulty + self._rng.normal(0, 0.1)
        obs[self._SLOT_DIFFICULTY] = float(np.clip(noisy_difficulty, 0.0, 1.0))
        return obs

    # -- bid sources ----------------------------------------------------------

    def _surrogate_score(self, base_bid: float, difficulty: float) -> float:
        sigma = self.cfg.surrogate_noise_easy + (
            self.cfg.surrogate_noise_hard - self.cfg.surrogate_noise_easy
        ) * difficulty
        corrupted = base_bid + self._rng.normal(0.0, sigma)
        return float(np.clip(corrupted, self.cfg.bid_min, self.cfg.bid_max))

    def _llm_score(self) -> Tuple[float, float]:
        if self.cfg.bidder is not None:
            result = self.cfg.bidder.bid(
                self._current_agent_state(),
                self._current_job_spec(),
                true_bid=float(self._true_bid),
            )
            return float(result.score), float(result.latency_seconds)
        latency = max(
            0.01,
            self._rng.normal(self.cfg.llm_latency_seconds_mean, self.cfg.llm_latency_seconds_std),
        )
        return float(self._true_bid), float(latency)

    def _current_agent_state(self) -> AgentState:
        util = self._obs[self._SLOT_UTIL]
        return AgentState(
            cpu_util=float(util[0]),
            gpu_util=float(util[1]),
            ram_util=float(util[2]),
            disk_util=float(util[3]),
            queue_length=int(self._step % 16),
            neighbor_load=float(np.mean(util)),
            historical_success_rate=1.0,
        )

    def _current_job_spec(self) -> JobSpec:
        req = self._obs[self._SLOT_REQ]
        return JobSpec(
            cpus=int(round(64 * float(req[0]))),
            gpus=int(round(4 * float(req[1]))),
            ram_gb=float(128.0 * float(req[2])),
            disk_gb=float(500.0 * float(req[3])),
            estimated_seconds=60.0,
            user="agent",
        )

    # -- step -----------------------------------------------------------------

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        bid_score = float(action["bid"])
        gate = int(action["gate"])

        if gate:
            effective_bid, llm_latency = self._llm_score()
        else:
            effective_bid = self._surrogate_score(bid_score, self._difficulty)
            llm_latency = 0.0

        score_error_norm = abs(effective_bid - self._true_bid) / 100.0
        extra_seconds = self.cfg.error_scale * (score_error_norm**self.cfg.error_exponent)
        estimated_seconds = 60.0
        actual_seconds = estimated_seconds + extra_seconds

        breakdown = composite_reward(
            actual_seconds=actual_seconds,
            estimated_seconds=estimated_seconds,
            utilization=np.asarray(self._obs[self._SLOT_UTIL], dtype=np.float32),
            gate=gate,
            llm_latency_seconds=llm_latency,
            weights=self.cfg.reward_weights,
        )

        self._step += 1
        truncated = self._step >= self.cfg.max_steps_per_episode
        info = {
            "difficulty": self._difficulty,
            "is_hard_job": self._difficulty > 0.5,
            "true_bid": self._true_bid,
            "effective_bid": effective_bid,
            "score_error": score_error_norm,
            "llm_latency": llm_latency,
            "reward_breakdown": breakdown,
        }
        self._obs = self._sample_obs()
        return self._obs, breakdown["total"], False, truncated, info
