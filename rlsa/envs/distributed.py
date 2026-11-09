"""Threaded asynchronous auction (paper Sec. V-B onward).

Each round (one job):

    coordinator (main thread)
      - builds every agent's local observation
      - starts one worker thread per agent
      - waits for bids; bids that arrive after ``bid_deadline_ms`` are flagged
        stale (lightly penalized, still counted); dropped messages are lost
      - the highest bid wins (deterministic or random tiebreak) and the winner
        holds the job's resources until completion

    agent worker (one thread per agent per round)
      - inbound network delay / drop -> compute the bid (surrogate, LLM, or the
        RLSA gate deciding between them) -> outbound delay / drop

RLSA agents learn online with PPO. In CTDE mode all agents share one policy
(pooled updates); in DTDE mode every agent owns its policy and optimizer.

Scope: threads in one process with a *simulated* network (Gaussian delay plus
optional drop). Real processes over a real WAN are covered by
:mod:`rlsa.fabric`.

Supported algorithms: ``random``, ``capacity``, ``always_surrogate``,
``always_llm``, ``oracle`` (privileged: calls the LLM exactly on hard jobs),
``confidence_gate`` (non-RL baseline over a frozen RLSA surrogate) and ``rlsa``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from rlsa.bidders.base import AgentState, JobSpec, capacity_bid
from rlsa.envs.auction import AuctionConfig, AuctionEnv

# ----------------------------------------------------------------------
# Network latency simulator
# ----------------------------------------------------------------------

@dataclass
class NetworkSim:
    """Gaussian per-message delay (ms) plus Bernoulli message drop."""

    mean_ms: float = 2.0
    std_ms: float = 0.5
    drop_prob: float = 0.0
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._lock = threading.Lock()

    def draw_delay(self) -> float:
        with self._lock:
            d = float(self._rng.normal(self.mean_ms, self.std_ms))
        return max(0.0, d)

    def send_ok(self) -> bool:
        if self.drop_prob <= 0.0:
            return True
        with self._lock:
            return float(self._rng.random()) >= self.drop_prob


# ----------------------------------------------------------------------
# Distributed config
# ----------------------------------------------------------------------

@dataclass
class DistributedConfig:
    base: AuctionConfig = field(default_factory=AuctionConfig)

    # Async auction
    bid_deadline_ms: float = 50.0  # how long coordinator waits before stale

    # Network simulation
    net_mean_ms: float = 2.0
    net_std_ms: float = 0.5
    net_drop_prob: float = 0.0
    # Optional per-agent overrides (length n_agents) of the net_* fields.
    net_mean_ms_per_agent: Optional[List[float]] = None
    net_std_ms_per_agent: Optional[List[float]] = None
    net_drop_prob_per_agent: Optional[List[float]] = None

    # Bidder per agent: "mock" or "hf". ``bidder_kinds`` (length n_agents)
    # overrides ``bidder_kind``; ``hf_model_names_per_agent`` overrides
    # ``hf_model_name`` (heterogeneous-LLM experiment).
    bidder_kinds: Optional[List[str]] = None
    bidder_kind: str = "mock"
    hf_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    hf_model_names_per_agent: Optional[List[str]] = None

    # Training (RLSA only)
    sharing_mode: str = "ctde"  # "ctde" or "dtde"
    pretrain_steps: int = 200
    update_every_rounds: int = 8

    # Discount for the PPO update. gamma=0 is the bandit-style advantage
    # (adv = reward - value); gamma>0 runs GAE over each agent's round-ordered
    # trajectory (discount-factor ablation).
    gamma: float = 0.0
    gae_lambda: float = 0.95

    # Confidence-gate baseline: invoke the LLM when the residual head predicts
    # a surrogate error above ``confidence_tau``. Needs ``load_policy_path``
    # (frozen RLSA policy) and ``residual_path`` (trained ResidualHead).
    confidence_tau: float = 0.0
    residual_path: str = ""
    # RLSA only: dump (obs, |LLM bid - surrogate bid|) on gated steps to an
    # .npz file, the training data for the residual head.
    dump_residual_data: str = ""

    # Tiebreak among equal top bids: "deterministic" (lowest agent id) or "random".
    tiebreak: str = "deterministic"
    tiebreak_tol: float = 1e-9

    # Save the trained (CTDE) policy, or load one and freeze it (no PPO updates,
    # no pretraining) for deployment-time evaluation.
    save_policy_path: str = ""
    load_policy_path: str = ""


# ----------------------------------------------------------------------
# Per-agent state (held by each thread)
# ----------------------------------------------------------------------

@dataclass
class AgentRuntime:
    agent_id: int
    algo: str
    bidder: Any  # MockLLMBidder | HFLocalBidder | None
    bidder_kind: str
    policy: Any = None  # GatingPolicy or None
    optimizer: Any = None
    policy_lock: Any = None  # threading.Lock; for CTDE this is shared across agents
    rng: Any = None
    pretrain_collected: int = 0
    in_pretrain: bool = True
    buf_obs: list = field(default_factory=list)
    buf_gate: list = field(default_factory=list)
    buf_logp: list = field(default_factory=list)
    buf_reward: list = field(default_factory=list)
    buf_value: list = field(default_factory=list)
    buf_teacher: list = field(default_factory=list)
    buf_mask: list = field(default_factory=list)
    rounds_since_update: int = 0
    pretrain_steps_total: int = 0
    conf_gate: Any = None  # ConfidenceGate (confidence_gate baseline only)


_HF_CACHE: Dict[str, Any] = {}  # model_name -> HFLocalBidder
_HF_CACHE_LOCK = threading.Lock()


def _build_bidder(kind: str, hf_model: str, seed: int, agent_id: int):
    """Build a per-agent bidder. HF bidders are *shared* across agents to
    avoid loading the same multi-GB model N times into the same GPU.
    Mock bidders are per-agent (so seeds differ for variance)."""
    from rlsa.bidders.mock import MockLLMBidder
    if kind == "mock":
        return MockLLMBidder(latency_mean=2.0, seed=seed + 7 * agent_id)
    elif kind == "hf":
        from rlsa.bidders.hf_local import HFLocalBidder
        with _HF_CACHE_LOCK:
            if hf_model not in _HF_CACHE:
                _HF_CACHE[hf_model] = HFLocalBidder(
                    model_name=hf_model, prompt_name="baseline")
            return _HF_CACHE[hf_model]
    else:
        return MockLLMBidder(latency_mean=2.0, seed=seed + 7 * agent_id)


def clear_hf_cache():
    """Drop cached HF bidders (e.g. between experiment cells) to free GPU memory."""
    with _HF_CACHE_LOCK:
        _HF_CACHE.clear()


def _compute_gae(rewards, values, gamma: float, lam: float):
    """GAE over a single round-ordered trajectory. gamma=0 reduces exactly to the
    original bandit advantage (adv = reward - value, returns = reward)."""
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    n = len(r)
    adv = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in range(n - 1, -1, -1):
        next_v = v[t + 1] if t + 1 < n else 0.0
        delta = r[t] + gamma * next_v - v[t]
        last = delta + gamma * lam * last
        adv[t] = last
    ret = adv + v
    return adv, ret


def _local_ppo_update(policy, opt, obs, gate, logp_old, rewards, values,
                      teacher_bid, mask, gamma=0.0, gae_lambda=0.95,
                      clip=0.2, value_coef=0.5,
                      ent_coef=0.02, imit_coef=1.0, epochs=4):
    import torch
    obs_t = torch.tensor(np.array(obs), dtype=torch.float32)
    gate_t = torch.tensor(gate, dtype=torch.float32)
    logp_old_t = torch.tensor(logp_old, dtype=torch.float32)
    tgt_t = torch.tensor(teacher_bid, dtype=torch.float32)
    mask_t = torch.tensor(mask, dtype=torch.float32)
    adv_np, ret_np = _compute_gae(rewards, values, gamma, gae_lambda)
    adv = torch.tensor(adv_np, dtype=torch.float32)
    if adv.std().item() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    returns = torch.tensor(ret_np, dtype=torch.float32)
    for _ in range(epochs):
        bid_pred, p_gate, value, _ = policy(obs_t)
        eps = 1e-8
        lp = (gate_t * torch.log(p_gate + eps)
              + (1.0 - gate_t) * torch.log(1.0 - p_gate + eps))
        ent = -(p_gate * torch.log(p_gate + eps)
                + (1.0 - p_gate) * torch.log(1.0 - p_gate + eps))
        ratio = torch.exp(lp - logp_old_t)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (value - returns).pow(2).mean()
        ent_bonus = ent.mean()
        if mask_t.sum().item() > 0:
            imit = ((bid_pred - tgt_t) / 100.0).pow(2)
            imit = (imit * mask_t).sum() / (mask_t.sum() + 1e-8)
        else:
            imit = torch.tensor(0.0)
        loss = policy_loss + value_coef * value_loss - ent_coef * ent_bonus + imit_coef * imit
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()


# ----------------------------------------------------------------------
# Single-agent bidder thread function
# ----------------------------------------------------------------------

def _agent_bid_fn(art: AgentRuntime, obs: np.ndarray, state: AgentState,
                  job: JobSpec, true_bid: float, is_hard: bool, net: NetworkSim,
                  out_box: dict, send_ok_in: bool, send_delay_in_ms: float) -> None:
    """Run on a worker thread per round per agent.

    out_box is a per-agent dict the coordinator inspects after wait_for_bids.
    Fills: {bid, gate, llm_latency, sur_bid, teacher_bid, mask, logp, value,
            obs, arrival_ts, recv_ok_out}
    """
    # Simulate inbound network: drop or delay
    if not send_ok_in:
        out_box["dropped_in"] = True
        return
    time.sleep(send_delay_in_ms / 1000.0)
    t_after_in = time.time()

    bid_val = 0.0
    gate_val = 0
    llm_lat = 0.0
    sur_bid = 0.0
    teacher_bid = 0.0
    mask = 0.0
    logp = 0.0
    value = 0.0
    bid_ok = True  # whether the LLM produced a valid bid (only meaningful when gate=1)

    algo = art.algo
    bidder = art.bidder

    if algo == "always_llm":
        r = bidder.bid(state, job, true_bid=true_bid)
        bid_val = float(r.score); gate_val = 1
        llm_lat = float(r.latency_seconds)
        bid_ok = bool(getattr(r, "succeeded", True))
    elif algo in ("always_surrogate", "capacity"):
        bid_val = capacity_bid(state.cpu_util, state.gpu_util,
                               state.ram_util, state.disk_util)
        gate_val = 0
    elif algo == "random":
        bid_val = float(art.rng.uniform(0, 100)); gate_val = 0
    elif algo == "oracle":
        gate_val = 1 if is_hard else 0
        if gate_val:
            r = bidder.bid(state, job, true_bid=true_bid)
            bid_val = float(r.score)
            llm_lat = float(r.latency_seconds)
            bid_ok = bool(getattr(r, "succeeded", True))
        else:
            bid_val = float(true_bid)
    elif algo == "confidence_gate":
        import torch
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        gate_val, sur_bid, err_hat = art.conf_gate.decide(obs_t)
        if gate_val == 1:
            r = bidder.bid(state, job, true_bid=true_bid)
            bid_val = float(r.score)
            llm_lat = float(r.latency_seconds)
            bid_ok = bool(getattr(r, "succeeded", True))
        else:
            bid_val = float(sur_bid)
    elif algo == "rlsa":
        import torch
        if art.in_pretrain and art.pretrain_collected < art.pretrain_steps_total:
            r = bidder.bid(state, job, true_bid=true_bid)
            bid_val = float(r.score); gate_val = 1
            llm_lat = float(r.latency_seconds)
            teacher_bid = bid_val; mask = 1.0
            art.pretrain_collected += 1
        else:
            # All accesses to art.policy / art.optimizer must hold the lock
            # since (for CTDE) multiple agent threads share one policy.
            lock = art.policy_lock
            with lock:
                if art.in_pretrain:
                    # Just exited pretrain — run imitation pass
                    if art.buf_obs:
                        ot = torch.tensor(np.array(art.buf_obs), dtype=torch.float32)
                        tt = torch.tensor(art.buf_teacher, dtype=torch.float32)
                        for _ in range(30):
                            bp, _, _, _ = art.policy(ot)
                            il = ((bp - tt) / 100.0).pow(2).mean()
                            art.optimizer.zero_grad(); il.backward(); art.optimizer.step()
                    art.in_pretrain = False
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    bid_pred, p_gate, val_pred, _ = art.policy(obs_t)
                    gate_dist = torch.distributions.Bernoulli(p_gate)
                    g_sample = gate_dist.sample()
                    lp = gate_dist.log_prob(g_sample)
            sur_bid = float(bid_pred.item())
            value = float(val_pred.item())
            logp = float(lp.item())
            gate_val = int(g_sample.item())
            if gate_val == 1:
                r = bidder.bid(state, job, true_bid=true_bid)
                bid_val = float(r.score)
                llm_lat = float(r.latency_seconds)
                teacher_bid = bid_val; mask = 1.0
            else:
                bid_val = sur_bid
                teacher_bid = 0.0; mask = 0.0
    else:
        bid_val = 50.0

    # Simulate outbound network
    if not net.send_ok():
        out_box["dropped_out"] = True
        return
    out_delay = net.draw_delay()
    time.sleep(out_delay / 1000.0)

    arrival_ts = time.time()
    out_box.update({
        "bid": float(bid_val), "gate": int(gate_val),
        "llm_latency": float(llm_lat),
        "sur_bid": float(sur_bid), "teacher_bid": float(teacher_bid),
        "mask": float(mask), "logp": float(logp), "value": float(value),
        "obs": obs, "arrival_ts": arrival_ts,
        "send_delay_in_ms": send_delay_in_ms, "send_delay_out_ms": out_delay,
        "compute_time_ms": (arrival_ts - t_after_in) * 1000.0,
        "bid_ok": bool(bid_ok),
    })


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def run_distributed(algo: str, cfg: DistributedConfig, seed: int,
                    verbose: bool = False) -> Dict[str, Any]:
    """Run one thread-per-agent asynchronous auction episode.

    Returns the :class:`~rlsa.envs.auction.MetricsTracker` metrics (makespan,
    ``llm_calls_per_job``, ``jain_fairness``, per-agent wins, ...) plus
    distributed fields: ``stale_bids``, ``dropped_msgs``,
    ``bid_latency_ms_{mean,p95,p99}``, ``deadline_misses_per_agent``,
    ``per_model_analysis`` and ``distributed_config``.
    """
    n_agents = cfg.base.n_agents
    bidder_kinds = cfg.bidder_kinds or [cfg.bidder_kind] * n_agents
    assert len(bidder_kinds) == n_agents

    per_agent_models = cfg.hf_model_names_per_agent or [cfg.hf_model_name] * n_agents

    # One NetworkSim per agent (independent RNG streams).
    means = cfg.net_mean_ms_per_agent or [cfg.net_mean_ms] * n_agents
    stds = cfg.net_std_ms_per_agent or [cfg.net_std_ms] * n_agents
    drops = cfg.net_drop_prob_per_agent or [cfg.net_drop_prob] * n_agents
    assert len(means) == n_agents and len(stds) == n_agents and len(drops) == n_agents
    nets = [NetworkSim(mean_ms=means[ai], std_ms=stds[ai], drop_prob=drops[ai],
                        seed=seed + 13 * ai) for ai in range(n_agents)]
    # Separate RNG for random tiebreaks so they do not consume network jitter draws.
    auction_rng = np.random.default_rng(seed + 31337)

    # The auction env provides the job stream, cluster, observations and metrics.
    env = AuctionEnv(cfg.base, seed=seed)
    env.reset()
    cluster = env.cluster
    metrics = env.metrics

    # ---- Build per-agent runtimes ----
    needs_torch = (algo in ("rlsa", "confidence_gate"))
    central_policy = None
    central_opt = None
    conf_gate = None  # shared ConfidenceGate (frozen surrogate + residual head)
    # A loaded policy is evaluated frozen: no pretraining and no PPO updates.
    freeze_policy = bool(cfg.load_policy_path)
    if needs_torch:
        import torch

        from rlsa.gating.policy import GatingConfig, GatingPolicy
        if algo == "confidence_gate":
            from rlsa.gating.confidence_gate import ConfidenceGate, ResidualHead
            frozen = GatingPolicy(GatingConfig(obs_dim=cfg.base.obs_dim))
            sd = torch.load(cfg.load_policy_path, map_location="cpu", weights_only=True)
            frozen.load_state_dict(sd)
            rh = ResidualHead(hidden_dim=GatingConfig(obs_dim=cfg.base.obs_dim).hidden_dim)
            rsd = torch.load(cfg.residual_path, map_location="cpu", weights_only=True)
            rh.load_state_dict(rsd)
            conf_gate = ConfidenceGate(frozen, rh, tau=cfg.confidence_tau)
            print(f"  confidence_gate: policy={cfg.load_policy_path} "
                  f"residual={cfg.residual_path} tau={cfg.confidence_tau}", flush=True)
        elif cfg.sharing_mode == "ctde":
            central_policy = GatingPolicy(GatingConfig(obs_dim=cfg.base.obs_dim))
            central_opt = torch.optim.Adam(central_policy.parameters(), lr=3e-4)
            if cfg.load_policy_path:
                sd = torch.load(cfg.load_policy_path, map_location="cpu",
                                 weights_only=True)
                central_policy.load_state_dict(sd)
                print(f"  loaded policy from {cfg.load_policy_path}", flush=True)

    runtimes: List[AgentRuntime] = []
    needs_bidder = algo in ("always_llm", "oracle", "rlsa", "confidence_gate")
    # Warm up HF bidders before the deadline-controlled loop so model loading
    # does not make the first bids stale.
    if needs_bidder:
        dummy_state = AgentState(0.5, 0.5, 0.5, 0.5, 0, 0.5)
        dummy_job = JobSpec(cpus=4, gpus=0, ram_gb=8.0, disk_gb=20.0,
                             estimated_seconds=60.0, user="warmup", job_id="0")
    for ai in range(n_agents):
        bidder = None
        if needs_bidder:
            bidder = _build_bidder(bidder_kinds[ai], per_agent_models[ai],
                                    seed=seed, agent_id=ai)
            if bidder_kinds[ai] == "hf" and hasattr(bidder, "_ensure_loaded"):
                try:
                    bidder._ensure_loaded()
                    # one warmup call so CUDA graph/cache is settled
                    bidder.bid(dummy_state, dummy_job, true_bid=50.0)
                except Exception as e:
                    print(f"  warn: HF warmup failed for agent {ai}: {e}",
                          flush=True)
        art = AgentRuntime(
            agent_id=ai, algo=algo, bidder=bidder,
            bidder_kind=bidder_kinds[ai],
            rng=np.random.default_rng(seed + 91 * ai),
        )
        if algo == "confidence_gate":
            # No RL policy/optimizer; all agents share the frozen gate.
            art.conf_gate = conf_gate
            art.in_pretrain = False
            art.pretrain_steps_total = 0
        elif needs_torch:
            if cfg.sharing_mode == "ctde":
                # All agents share the central policy via reference (threads)
                art.policy = central_policy
                art.optimizer = central_opt
                # Will be set to the shared lock just below
            else:
                # DTDE: each agent has its own
                art.policy = GatingPolicy(GatingConfig(obs_dim=cfg.base.obs_dim))
                art.optimizer = torch.optim.Adam(art.policy.parameters(), lr=3e-4)
                art.policy_lock = threading.Lock()
        if algo != "confidence_gate":
            art.pretrain_steps_total = cfg.pretrain_steps
            # A frozen (loaded) policy skips pretraining.
            if freeze_policy:
                art.in_pretrain = False
                art.pretrain_steps_total = 0
        runtimes.append(art)

    # CTDE: all agents share one policy lock (set after runtimes are built)
    if needs_torch and cfg.sharing_mode == "ctde":
        shared_policy_lock = threading.Lock()
        for art in runtimes:
            art.policy_lock = shared_policy_lock

    # Metrics
    stale_bids = 0
    dropped_msgs = 0
    bid_arrival_ms: List[float] = []
    deadline_misses_per_agent: Dict[int, int] = {ai: 0 for ai in range(n_agents)}

    # Residual-head training data (gated steps only): (obs, |teacher - surrogate|)
    _resid_obs: List = []
    _resid_err: List[float] = []
    # Per-model LLM statistics (latency, invalid outputs, bid error)
    per_model_lat: Dict[str, list] = {}
    per_model_invalid: Dict[str, list] = {}
    per_model_abserr: Dict[str, list] = {}

    # ---- Main async-auction loop ----
    while not env.done():
        env.next_job()
        true_bid = env.get_true_bid()
        is_hard = env.get_is_hard()
        job = env.get_job_spec()
        per_agent_obs = [env.get_obs(ai) for ai in range(n_agents)]

        # Per-round comm setup: each agent has its own network sim
        in_send_ok = [nets[ai].send_ok() for ai in range(n_agents)]
        in_delay_ms = [nets[ai].draw_delay() for ai in range(n_agents)]

        boxes: List[dict] = [{"dropped_in": False, "dropped_out": False}
                              for _ in range(n_agents)]

        t_round_start = time.time()
        threads: List[threading.Thread] = []
        for ai in range(n_agents):
            t = threading.Thread(
                target=_agent_bid_fn,
                args=(runtimes[ai], per_agent_obs[ai][0], per_agent_obs[ai][1],
                      job, true_bid, is_hard, nets[ai], boxes[ai],
                      in_send_ok[ai], in_delay_ms[ai]),
                daemon=True,
            )
            t.start()
            threads.append(t)

        # Bids after the deadline still count but are flagged stale; threads
        # are only abandoned after a generous tail (a real fault).
        deadline_window_s = cfg.bid_deadline_ms / 1000.0
        max_wait_s = max(deadline_window_s + 30.0, 40.0)
        for t in threads:
            remaining = max_wait_s - (time.time() - t_round_start)
            if remaining > 0:
                t.join(timeout=remaining)

        # Collect arrived (and in-deadline) bids
        bids: Dict[int, dict] = {}
        for ai in range(n_agents):
            box = boxes[ai]
            if box.get("dropped_in") or box.get("dropped_out"):
                dropped_msgs += 1
                deadline_misses_per_agent[ai] += 1
                continue
            if "bid" not in box:
                # thread didn't finish in time
                dropped_msgs += 1
                deadline_misses_per_agent[ai] += 1
                continue
            arrival_s = box["arrival_ts"] - t_round_start
            box["is_stale"] = arrival_s > deadline_window_s
            if box["is_stale"]:
                stale_bids += 1
                deadline_misses_per_agent[ai] += 1
            bid_arrival_ms.append(arrival_s * 1000.0)
            bids[ai] = box

        if not bids:
            continue  # all dropped/missed — skip this job

        # Argmax over arrived (even stale) bids.
        # Tiebreak: deterministic = lowest agent_id; random = uniform among ties.
        items = [(ai, b["bid"]) for ai, b in bids.items()]
        top_bid = max(b for _, b in items)
        tied = [ai for ai, b in items if (top_bid - b) <= cfg.tiebreak_tol]
        if cfg.tiebreak == "random" and len(tied) > 1:
            winner_cand = int(auction_rng.choice(tied))
        else:
            winner_cand = min(tied)  # deterministic by lowest id
        # Resource-feasibility fallthrough: if winner can't run, advance time
        winner = None
        if cluster.can_run(job):
            winner = winner_cand
        elif cluster._running:
            next_end = min(rj.end_t for rj in cluster._running)
            cluster.tick(dt=next_end - cluster._t)
            winner = winner_cand
        if winner is None:
            continue

        # Reward for the winner
        from rlsa.rewards import composite_reward
        wb = bids[winner]
        eff_bid = wb["bid"]; gate = wb["gate"]; llm_lat = wb["llm_latency"]
        score_err = abs(eff_bid - true_bid) / 100.0
        item = env._round_current_job
        _, arrival_t, _, _, duration = item
        extra_sec = cfg.base.error_scale * (score_err ** cfg.base.error_exponent)
        actual_sec = duration + extra_sec
        breakdown = composite_reward(
            actual_seconds=actual_sec,
            estimated_seconds=duration,
            utilization=cluster.util(),
            gate=gate, llm_latency_seconds=llm_lat,
            weights=cfg.base.reward_weights,
        )
        winner_reward = float(breakdown["total"])

        per_agent_reward = {ai: 0.0 for ai in range(n_agents)}
        per_agent_reward[winner] = winner_reward
        for ai, b in bids.items():
            if b.get("is_stale"):
                per_agent_reward[ai] -= 0.05  # stale penalty

        # Record bids
        for ai, b in bids.items():
            difficulty = 0.9 if is_hard else 0.1
            metrics.record_bid(ai, b["gate"], difficulty)
            # Per-model LLM statistics (only when the LLM was actually called)
            if b.get("gate", 0) == 1:
                mdl = per_agent_models[ai] if bidder_kinds[ai] == "hf" else "mock"
                per_model_lat.setdefault(mdl, []).append(b.get("llm_latency", 0.0) * 1000.0)
                inv = per_model_invalid.setdefault(mdl, [0, 0]); inv[1] += 1
                if not b.get("bid_ok", True):
                    inv[0] += 1
                per_model_abserr.setdefault(mdl, []).append(abs(b.get("bid", 0.0) - true_bid))
        metrics.record_win(winner)

        cluster.assign(
            job=job, job_id=env._next_job_id, agent_id=winner,
            arrival_t=arrival_t, duration=actual_sec,
            is_hard=is_hard, true_bid=true_bid,
        )

        # ---- Commit experience to each agent's buffer ----
        if algo == "rlsa":
            for ai in range(n_agents):
                if ai not in bids:
                    continue
                b = bids[ai]
                art = runtimes[ai]
                art.buf_obs.append(b["obs"])
                art.buf_gate.append(b["gate"])
                art.buf_logp.append(b["logp"])
                art.buf_reward.append(per_agent_reward[ai])
                art.buf_value.append(b["value"])
                art.buf_teacher.append(b["teacher_bid"])
                art.buf_mask.append(b["mask"])
                art.rounds_since_update += 1
                # Residual-head data: gated post-pretrain steps only (during
                # pretraining the surrogate is not evaluated and sur_bid == 0).
                if (cfg.dump_residual_data and b.get("mask", 0.0) == 1.0
                        and b.get("sur_bid", 0.0) != 0.0):
                    _resid_obs.append(b["obs"])
                    _resid_err.append(abs(b["teacher_bid"] - b["sur_bid"]))

            # ---- PPO updates (main thread, under the policy lock) ----
            if freeze_policy:
                for art in runtimes:
                    art.buf_obs.clear(); art.buf_gate.clear(); art.buf_logp.clear()
                    art.buf_reward.clear(); art.buf_value.clear()
                    art.buf_teacher.clear(); art.buf_mask.clear()
                    art.rounds_since_update = 0
            elif cfg.sharing_mode == "dtde":
                for art in runtimes:
                    if (art.rounds_since_update >= cfg.update_every_rounds
                            and len(art.buf_obs) >= 8):
                        with art.policy_lock:
                            _local_ppo_update(art.policy, art.optimizer,
                                              art.buf_obs, art.buf_gate, art.buf_logp,
                                              art.buf_reward, art.buf_value,
                                              art.buf_teacher, art.buf_mask,
                                              gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
                        art.buf_obs.clear(); art.buf_gate.clear(); art.buf_logp.clear()
                        art.buf_reward.clear(); art.buf_value.clear()
                        art.buf_teacher.clear(); art.buf_mask.clear()
                        art.rounds_since_update = 0
            else:
                # CTDE: pool buffers across agents once every K rounds
                total_rounds = sum(a.rounds_since_update for a in runtimes)
                if total_rounds >= cfg.update_every_rounds * max(1, n_agents):
                    pooled_obs = []; pooled_g = []; pooled_lp = []
                    pooled_r = []; pooled_v = []; pooled_t = []; pooled_m = []
                    for art in runtimes:
                        pooled_obs.extend(art.buf_obs)
                        pooled_g.extend(art.buf_gate)
                        pooled_lp.extend(art.buf_logp)
                        pooled_r.extend(art.buf_reward)
                        pooled_v.extend(art.buf_value)
                        pooled_t.extend(art.buf_teacher)
                        pooled_m.extend(art.buf_mask)
                    if len(pooled_obs) >= 16:
                        with runtimes[0].policy_lock:
                            _local_ppo_update(central_policy, central_opt,
                                              pooled_obs, pooled_g, pooled_lp,
                                              pooled_r, pooled_v, pooled_t, pooled_m,
                                              gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
                    for art in runtimes:
                        art.buf_obs.clear(); art.buf_gate.clear(); art.buf_logp.clear()
                        art.buf_reward.clear(); art.buf_value.clear()
                        art.buf_teacher.clear(); art.buf_mask.clear()
                        art.rounds_since_update = 0

    cluster.drain()
    out = metrics.finalize(cluster)
    out["stale_bids"] = int(stale_bids)
    out["dropped_msgs"] = int(dropped_msgs)
    out["bid_latency_ms_mean"] = float(np.mean(bid_arrival_ms)) if bid_arrival_ms else 0.0
    out["bid_latency_ms_p95"] = float(np.percentile(bid_arrival_ms, 95)) if bid_arrival_ms else 0.0
    out["bid_latency_ms_p99"] = float(np.percentile(bid_arrival_ms, 99)) if bid_arrival_ms else 0.0
    out["deadline_misses_per_agent"] = {str(k): v for k, v in deadline_misses_per_agent.items()}
    out["distributed_config"] = {
        "n_agents": n_agents,
        "bidder_kinds": bidder_kinds,
        "per_agent_hf_models": per_agent_models if cfg.hf_model_names_per_agent else None,
        "bid_deadline_ms": cfg.bid_deadline_ms,
        "net_mean_ms": cfg.net_mean_ms,
        "net_std_ms": cfg.net_std_ms,
        "net_drop_prob": cfg.net_drop_prob,
        "net_mean_ms_per_agent": means,
        "net_std_ms_per_agent": stds,
        "net_drop_prob_per_agent": drops,
        "sharing_mode": cfg.sharing_mode,
        "tiebreak": cfg.tiebreak,
        "freeze_policy": freeze_policy,
        "load_policy_path": cfg.load_policy_path,
        "gamma": cfg.gamma,
        "confidence_tau": cfg.confidence_tau if algo == "confidence_gate" else None,
        "cost_beta": cfg.base.reward_weights.cost_beta,
    }

    # Per-model LLM statistics (latency percentiles, invalid rate, bid error)
    def _pct(xs, q):
        return float(np.percentile(xs, q)) if xs else 0.0
    _models = set(list(per_model_lat) + list(per_model_invalid) + list(per_model_abserr))
    out["per_model_analysis"] = {
        mdl: {
            "llm_latency_ms_p50": _pct(per_model_lat.get(mdl, []), 50),
            "llm_latency_ms_p95": _pct(per_model_lat.get(mdl, []), 95),
            "llm_latency_ms_p99": _pct(per_model_lat.get(mdl, []), 99),
            "invalid_rate": (per_model_invalid[mdl][0] / per_model_invalid[mdl][1])
                              if per_model_invalid.get(mdl, [0, 0])[1] else 0.0,
            "bid_abserr_mean": float(np.mean(per_model_abserr[mdl])) if per_model_abserr.get(mdl) else 0.0,
            "n_calls": per_model_invalid.get(mdl, [0, 0])[1],
        }
        for mdl in _models
    }

    # Residual-head training data
    if cfg.dump_residual_data and _resid_obs:
        import os as _os
        _os.makedirs(_os.path.dirname(cfg.dump_residual_data) or ".", exist_ok=True)
        np.savez(cfg.dump_residual_data,
                 obs=np.array(_resid_obs, dtype=np.float32),
                 err=np.array(_resid_err, dtype=np.float32))
        out["residual_data_path"] = cfg.dump_residual_data
        out["residual_data_n"] = int(len(_resid_err))

    # Save the shared (CTDE) policy if requested
    if needs_torch and cfg.save_policy_path and central_policy is not None:
        import os as _os

        import torch as _torch
        _os.makedirs(_os.path.dirname(cfg.save_policy_path) or ".", exist_ok=True)
        _torch.save(central_policy.state_dict(), cfg.save_policy_path)
        out["saved_policy_path"] = cfg.save_policy_path
        print(f"  saved policy to {cfg.save_policy_path}", flush=True)
    return out
