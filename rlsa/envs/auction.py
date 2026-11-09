"""Shared-cluster auction environment.

* one shared cluster with finite CPU / GPU / RAM / disk;
* a pre-generated job stream (identical for every algorithm under a given seed,
  so seeds are valid units of paired comparison);
* every agent bids on every job, the highest bid wins (ties: lowest agent id);
* the winner holds the job's resources for its whole (possibly bid-error
  inflated) duration;
* the winner receives the composite reward, losers receive 0.

This module also provides the :class:`Cluster` and :class:`MetricsTracker` used
by the threaded distributed auction (:mod:`rlsa.envs.distributed`) and the
FABRIC coordinator (:mod:`rlsa.fabric.coordinator`).

The auction is coordinated centrally (argmax); a decentralized consensus
protocol is out of scope.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rlsa.bidders.base import AgentState, JobSpec
from rlsa.rewards import RewardWeights, composite_reward

# ----------------------------------------------------------------------
# Cluster: simple shared resource pool
# ----------------------------------------------------------------------

@dataclass
class _RunningJob:
    job_id: int
    agent_id: int
    cpus: int
    gpus: int
    ram_gb: float
    disk_gb: float
    arrival_t: float
    start_t: float
    end_t: float
    is_hard: bool
    true_bid: float


@dataclass
class Cluster:
    """Shared HPC resource pool advanced by explicit, synchronous ``tick()`` calls.

    Simulated time is decoupled from wall-clock time, so results do not depend
    on host speed.
    """

    cpus: int = 64
    gpus: int = 8
    ram_gb: float = 256.0
    disk_gb: float = 4096.0

    _t: float = 0.0
    _free_cpus: int = field(init=False)
    _free_gpus: int = field(init=False)
    _free_ram: float = field(init=False)
    _free_disk: float = field(init=False)
    _running: List[_RunningJob] = field(default_factory=list)
    _completed: List[_RunningJob] = field(default_factory=list)

    def __post_init__(self):
        self._free_cpus = self.cpus
        self._free_gpus = self.gpus
        self._free_ram = self.ram_gb
        self._free_disk = self.disk_gb

    def can_run(self, job: JobSpec) -> bool:
        return (
            job.cpus <= self._free_cpus
            and job.gpus <= self._free_gpus
            and job.ram_gb <= self._free_ram
            and job.disk_gb <= self._free_disk
        )

    def util(self) -> np.ndarray:
        return np.array([
            1.0 - self._free_cpus / max(1, self.cpus),
            1.0 - self._free_gpus / max(1, self.gpus),
            1.0 - self._free_ram / max(1.0, self.ram_gb),
            1.0 - self._free_disk / max(1.0, self.disk_gb),
        ], dtype=np.float32)

    def assign(self, job: JobSpec, job_id: int, agent_id: int,
               arrival_t: float, duration: float,
               is_hard: bool, true_bid: float) -> _RunningJob:
        self._free_cpus -= job.cpus
        self._free_gpus -= job.gpus
        self._free_ram -= job.ram_gb
        self._free_disk -= job.disk_gb
        rj = _RunningJob(
            job_id=job_id, agent_id=agent_id,
            cpus=job.cpus, gpus=job.gpus,
            ram_gb=job.ram_gb, disk_gb=job.disk_gb,
            arrival_t=arrival_t, start_t=self._t, end_t=self._t + duration,
            is_hard=is_hard, true_bid=true_bid,
        )
        self._running.append(rj)
        return rj

    def tick(self, dt: float = 1.0) -> List[_RunningJob]:
        """Advance time, release finished jobs. Returns the ones that finished."""
        self._t += dt
        finished = [rj for rj in self._running if rj.end_t <= self._t]
        for rj in finished:
            self._free_cpus += rj.cpus
            self._free_gpus += rj.gpus
            self._free_ram += rj.ram_gb
            self._free_disk += rj.disk_gb
            self._completed.append(rj)
        self._running = [rj for rj in self._running if rj.end_t > self._t]
        return finished

    def drain(self, max_t: float = 1e9) -> None:
        """Advance until all running jobs done."""
        while self._running and self._t < max_t:
            next_end = min(rj.end_t for rj in self._running)
            self.tick(dt=max(1.0, next_end - self._t))


# ----------------------------------------------------------------------
# Metrics tracker
# ----------------------------------------------------------------------

@dataclass
class MetricsTracker:
    n_agents: int
    n_jobs: int = 0
    n_llm_calls: int = 0
    n_llm_calls_per_agent: Dict[int, int] = field(default_factory=dict)
    bids_per_agent: Dict[int, int] = field(default_factory=dict)
    wins_per_agent: Dict[int, int] = field(default_factory=dict)
    # difficulty -> P(LLM call) sample per agent
    diff_route_log: Dict[int, List[Tuple[float, int]]] = field(default_factory=dict)
    rejection_per_agent: Dict[int, int] = field(default_factory=dict)
    queue_wait_per_job: List[float] = field(default_factory=list)
    start_wallclock: float = field(default_factory=time.time)

    def record_bid(self, agent_id: int, gate: int, difficulty: float):
        self.bids_per_agent[agent_id] = self.bids_per_agent.get(agent_id, 0) + 1
        if gate:
            self.n_llm_calls += 1
            self.n_llm_calls_per_agent[agent_id] = self.n_llm_calls_per_agent.get(agent_id, 0) + 1
        self.diff_route_log.setdefault(agent_id, []).append((difficulty, gate))

    def record_win(self, agent_id: int):
        self.wins_per_agent[agent_id] = self.wins_per_agent.get(agent_id, 0) + 1

    def record_rejection(self, agent_id: int):
        self.rejection_per_agent[agent_id] = self.rejection_per_agent.get(agent_id, 0) + 1

    def finalize(self, cluster: Cluster) -> Dict[str, Any]:
        all_jobs = cluster._completed
        if not all_jobs:
            return {"error": "no completed jobs"}
        makespan = max(rj.end_t for rj in all_jobs)
        flow = [rj.end_t - rj.arrival_t for rj in all_jobs]
        # utilization = sum(nodes * runtime) / (max possible)
        node_seconds_used = sum(
            (rj.cpus / cluster.cpus) * (rj.end_t - rj.start_t)
            for rj in all_jobs
        )
        # Effective parallelism over makespan window:
        utilization = node_seconds_used / max(1e-9, makespan)
        # Jain fairness over wins
        wins = [self.wins_per_agent.get(i, 0) for i in range(self.n_agents)]
        s1 = sum(wins)
        s2 = sum(w * w for w in wins)
        jain = (s1 * s1) / max(1e-9, self.n_agents * s2) if s2 > 0 else 0.0
        # Per-agent gating selectivity (Spearman ρ between difficulty & route)
        selectivity = {}
        for ai, log in self.diff_route_log.items():
            if len(log) < 5:
                selectivity[ai] = float("nan")
                continue
            diffs = np.array([d for d, _ in log])
            routes = np.array([r for _, r in log], dtype=np.float64)
            # Spearman: rank correlation
            d_rank = np.argsort(np.argsort(diffs))
            r_rank = np.argsort(np.argsort(routes))
            if r_rank.std() == 0:
                selectivity[ai] = float("nan")
            else:
                rho = float(np.corrcoef(d_rank, r_rank)[0, 1])
                selectivity[ai] = rho
        bid_win_rate = {
            ai: self.wins_per_agent.get(ai, 0) / max(1, self.bids_per_agent.get(ai, 0))
            for ai in range(self.n_agents)
        }
        return {
            "n_jobs": self.n_jobs,
            "n_agents": self.n_agents,
            "makespan": float(makespan),
            "flow_time_mean": float(np.mean(flow)),
            "flow_time_std": float(np.std(flow)),
            "utilization": float(utilization),
            "n_llm_calls_total": self.n_llm_calls,
            "llm_calls_per_job": self.n_llm_calls / max(1, self.n_jobs),
            "n_llm_calls_per_agent": dict(self.n_llm_calls_per_agent),
            "wins_per_agent": dict(self.wins_per_agent),
            "bids_per_agent": dict(self.bids_per_agent),
            "bid_win_rate_per_agent": bid_win_rate,
            "rejection_per_agent": dict(self.rejection_per_agent),
            "jain_fairness": float(jain),
            "gating_selectivity_per_agent": selectivity,
            "wallclock_seconds": time.time() - self.start_wallclock,
        }


# ----------------------------------------------------------------------
# Multi-agent env
# ----------------------------------------------------------------------

@dataclass
class AuctionConfig:
    n_agents: int = 4
    n_jobs: int = 200
    cluster_cpus: int = 64
    cluster_gpus: int = 8
    cluster_ram_gb: float = 256.0
    cluster_disk_gb: float = 4096.0
    arrival_rate: float = 0.5  # jobs per time unit (avg)
    hard_job_prob: float = 0.3
    obs_dim: int = 20
    bid_min: float = 0.0
    bid_max: float = 100.0
    reward_weights: Optional[RewardWeights] = None

    # Difficulty model (same constants as GatingEnv)
    surrogate_noise_easy: float = 2.0
    surrogate_noise_hard: float = 55.0
    error_scale: float = 800.0
    error_exponent: float = 2.0
    llm_latency_seconds_mean: float = 2.0
    llm_latency_seconds_std: float = 0.5

    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = RewardWeights()


class AuctionEnv:
    """N agents bid on a shared cluster; one wins per job.

    Synchronous usage::

        env = AuctionEnv(cfg, seed=0)
        env.reset()
        while not env.done():
            env.next_job()
            for agent_id in range(cfg.n_agents):
                obs, state = env.get_obs(agent_id)
                ...  # compute (bid, gate, llm_latency) from obs/state
                env.submit_bid(agent_id, bid, gate, llm_latency)
            rewards = env.resolve_round()  # picks winner, assigns the job
        metrics = env.finalize()
    """

    def __init__(self, cfg: Optional[AuctionConfig] = None,
                 seed: Optional[int] = None):
        self.cfg = cfg or AuctionConfig()
        self.rng = np.random.default_rng(seed)
        self.cluster = Cluster(
            cpus=self.cfg.cluster_cpus, gpus=self.cfg.cluster_gpus,
            ram_gb=self.cfg.cluster_ram_gb, disk_gb=self.cfg.cluster_disk_gb,
        )
        self.metrics = MetricsTracker(n_agents=self.cfg.n_agents)
        self._next_job_id = 0
        self._jobs_emitted = 0
        # Pre-generated job stream so all algos see the same stream per seed
        self._stream = self._generate_stream()
        # Round-state
        self._round_bids: Dict[int, Tuple[float, int, float, float]] = {}  # agent_id -> (bid, gate, llm_latency, true_bid)
        self._round_obs: Dict[int, np.ndarray] = {}  # agent_id -> obs (for record)
        self._round_current_job: Optional[Tuple[JobSpec, float, bool, float, float]] = None
        # (job, arrival_t, is_hard, true_bid, duration)
        self._round_blacklist: set = set()

    # -------- stream

    def _generate_stream(self) -> List[Tuple[JobSpec, float, bool, float, float]]:
        out = []
        t = 0.0
        for jid in range(self.cfg.n_jobs):
            # Sample job
            cpus = int(self.rng.integers(1, 32))
            gpus = int(self.rng.integers(0, 4))
            ram = float(self.rng.uniform(2, 64))
            disk = float(self.rng.uniform(5, 200))
            # difficulty
            is_hard = bool(self.rng.random() < self.cfg.hard_job_prob)
            # true bid: inverse of the job's mean resource burden
            burden = (cpus / 32 + gpus / 4 + ram / 64 + disk / 200) / 4.0
            true_bid = float(np.clip((1.0 - burden) * 100.0, 0.0, 100.0))
            # duration: hard jobs longer
            base_dur = 60.0 + 60.0 * burden
            duration = base_dur * (1.5 if is_hard else 1.0)
            # arrival time: poisson-ish
            t += float(self.rng.exponential(1.0 / max(1e-3, self.cfg.arrival_rate)))
            job = JobSpec(
                cpus=cpus, gpus=gpus, ram_gb=ram, disk_gb=disk,
                estimated_seconds=duration, user=f"u{jid % 5}", job_id=str(jid),
            )
            out.append((job, t, is_hard, true_bid, duration))
        return out

    # -------- public API

    def reset(self) -> None:
        self.cluster = Cluster(
            cpus=self.cfg.cluster_cpus, gpus=self.cfg.cluster_gpus,
            ram_gb=self.cfg.cluster_ram_gb, disk_gb=self.cfg.cluster_disk_gb,
        )
        self.metrics = MetricsTracker(n_agents=self.cfg.n_agents)
        self._next_job_id = 0
        self._jobs_emitted = 0
        self._round_bids = {}
        self._round_obs = {}
        self._round_current_job = None
        self._round_blacklist = set()

    def done(self) -> bool:
        return self._jobs_emitted >= len(self._stream)

    def next_job(self) -> Tuple[JobSpec, float, bool, float, float]:
        """Pop the next job from the stream and advance cluster time to its arrival."""
        if self.done():
            raise RuntimeError("stream exhausted")
        item = self._stream[self._jobs_emitted]
        job, arrival_t, is_hard, true_bid, duration = item
        # Advance cluster time to arrival
        while self.cluster._t < arrival_t and self.cluster._running:
            next_end = min(rj.end_t for rj in self.cluster._running)
            if next_end > arrival_t:
                self.cluster.tick(dt=arrival_t - self.cluster._t)
                break
            self.cluster.tick(dt=next_end - self.cluster._t)
        if self.cluster._t < arrival_t:
            self.cluster._t = arrival_t
        # Reset round state
        self._round_bids = {}
        self._round_obs = {}
        self._round_blacklist = set()
        self._round_current_job = item
        self._jobs_emitted += 1
        self._next_job_id += 1
        self.metrics.n_jobs += 1
        return item

    def get_obs(self, agent_id: int) -> Tuple[np.ndarray, AgentState]:
        """Build the observation for one agent. Job is the round's current job.

        Each agent has a slightly different "local view" via a per-agent noise
        on the difficulty signal slot. Resources are shared, so utilization is
        identical across agents (this is the centralized auction property).
        """
        if self._round_current_job is None:
            raise RuntimeError("call next_job() before get_obs()")
        job, arrival_t, is_hard, true_bid, _ = self._round_current_job
        util = self.cluster.util()
        obs = self.rng.uniform(0.0, 1.0, size=self.cfg.obs_dim).astype(np.float32)
        obs[0:4] = util
        # req slot: normalized to [0, 1]
        obs[4] = min(1.0, job.cpus / 32.0)
        obs[5] = min(1.0, job.gpus / 4.0)
        obs[6] = min(1.0, job.ram_gb / 64.0)
        obs[7] = min(1.0, job.disk_gb / 200.0)
        # difficulty signal slot (noisy)
        difficulty = 0.9 if is_hard else 0.1
        obs[8] = float(np.clip(difficulty + self.rng.normal(0, 0.1), 0.0, 1.0))
        state = AgentState(
            cpu_util=float(util[0]), gpu_util=float(util[1]),
            ram_util=float(util[2]), disk_util=float(util[3]),
            queue_length=len(self.cluster._running),
            neighbor_load=float(np.mean(util)),
            historical_success_rate=1.0,
        )
        self._round_obs[agent_id] = obs.copy()
        return obs, state

    def get_job_spec(self) -> JobSpec:
        return self._round_current_job[0]

    def get_true_bid(self) -> float:
        return self._round_current_job[3]

    def get_is_hard(self) -> bool:
        return self._round_current_job[2]

    def submit_bid(self, agent_id: int, bid: float, gate: int,
                   llm_latency: float, effective_bid: Optional[float] = None) -> None:
        """Record an agent's bid. effective_bid is the bid that will be compared
        (overrides bid if LLM was called and produced its own value)."""
        eff = float(effective_bid if effective_bid is not None else bid)
        true_bid = self._round_current_job[3]
        is_hard = self._round_current_job[2]
        difficulty = 0.9 if is_hard else 0.1
        self._round_bids[agent_id] = (eff, int(gate), float(llm_latency), true_bid)
        self.metrics.record_bid(agent_id, int(gate), difficulty)

    def resolve_round(self) -> Dict[int, float]:
        """Pick winner, assign job, return per-agent rewards (winner=composite, losers=0)."""
        if not self._round_bids:
            raise RuntimeError("no bids submitted")
        job, arrival_t, is_hard, true_bid, duration = self._round_current_job
        rewards = {ai: 0.0 for ai in range(self.cfg.n_agents)}

        # Try winners in descending bid order until one fits or we exhaust
        eligible = [
            (ai, b) for ai, (b, _, _, _) in self._round_bids.items()
            if ai not in self._round_blacklist
        ]
        winner = None
        while eligible:
            eligible.sort(key=lambda x: (-x[1], x[0]))  # desc bid, asc agent_id
            cand_id = eligible[0][0]
            if self.cluster.can_run(job):
                winner = cand_id
                break
            else:
                self._round_blacklist.add(cand_id)
                self.metrics.record_rejection(cand_id)
                eligible.pop(0)
        if winner is None:
            # Nobody fits; job has to wait. Advance cluster until something finishes.
            if self.cluster._running:
                next_end = min(rj.end_t for rj in self.cluster._running)
                self.cluster.tick(dt=next_end - self.cluster._t)
                # Re-try without blacklist
                self._round_blacklist = set()
                return self.resolve_round()
            else:
                # Pathological: nobody can ever run this job. Skip.
                return rewards

        # Winner gets the reward
        eff_bid, gate, llm_lat, _ = self._round_bids[winner]
        score_err = abs(eff_bid - true_bid) / 100.0
        extra_sec = self.cfg.error_scale * (score_err ** self.cfg.error_exponent)
        actual_sec = duration + extra_sec
        breakdown = composite_reward(
            actual_seconds=actual_sec,
            estimated_seconds=duration,
            utilization=self.cluster.util(),
            gate=gate,
            llm_latency_seconds=llm_lat,
            weights=self.cfg.reward_weights,
        )
        rewards[winner] = breakdown["total"]

        # Assign job to cluster
        self.cluster.assign(
            job=job, job_id=self._next_job_id, agent_id=winner,
            arrival_t=arrival_t, duration=actual_sec,
            is_hard=is_hard, true_bid=true_bid,
        )
        self.metrics.record_win(winner)
        return rewards

    def finalize(self) -> Dict[str, Any]:
        # Drain remaining jobs
        self.cluster.drain()
        return self.metrics.finalize(self.cluster)

    def write_metrics(self, path: str) -> None:
        m = self.finalize()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(m, f, indent=2)
