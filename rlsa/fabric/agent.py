"""Bidding agent for the real-network (FABRIC) deployment.

Connects to :mod:`rlsa.fabric.coordinator` over TCP and answers one bid per
round with a frozen policy (no training happens on the agents).

Protocol::

    agent -> coord   {"type": "hello", "agent_id"}
    coord -> agent   {"type": "job", "round_id", "obs", "state_tuple", "job_kw", "true_bid"}
    agent -> coord   {"type": "bid", "round_id", "agent_id", "bid", "gate", "llm_latency", "compute_s"}
    coord -> agent   {"type": "stop"}                     (or the socket closes)

Run on an agent VM::

    python -m rlsa.fabric.agent --host <coord-ip> --port 9009 --agent-id 0 \
        --obs-dim 20 --algo rlsa --policy-path policy.pt
"""
from __future__ import annotations

import argparse
import socket
import time

import numpy as np

from rlsa.bidders.base import AgentState, JobSpec, capacity_bid
from rlsa.fabric.wire import recv_msg, send_msg


def _make_policy(obs_dim: int, policy_path: str):
    import torch

    from rlsa.gating.policy import GatingConfig, GatingPolicy
    policy = GatingPolicy(GatingConfig(obs_dim=obs_dim))
    policy.load_state_dict(torch.load(policy_path, map_location="cpu", weights_only=True))
    policy.eval()
    return policy


def compute_bid(algo, policy, bidder, msg, seed, agent_id):
    """Return ``(bid, gate, llm_latency)`` for one round message."""
    import torch

    round_id = msg["round_id"]
    obs = msg["obs"]
    state_tuple = msg["state_tuple"]
    job_kw = msg["job_kw"]
    true_bid = msg["true_bid"]

    gate = 0
    bid = 50.0
    llm_lat = 0.0
    if algo == "always_llm":
        st = AgentState(*state_tuple); js = JobSpec(**job_kw)
        r = bidder.bid(st, js, true_bid=true_bid)
        bid = float(r.score); gate = 1; llm_lat = float(r.latency_seconds)
    elif algo in ("always_surrogate", "capacity"):
        bid = capacity_bid(*state_tuple[:4])
    elif algo == "random":
        bid = float(np.random.default_rng(seed + round_id * 131 + agent_id).uniform(0, 100))
    elif algo == "oracle":
        bid = float(true_bid)
    elif algo == "rlsa":
        ot = torch.tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            bid_pred, p_gate, _, _ = policy(ot)
            g_sample = torch.distributions.Bernoulli(p_gate).sample()
        gate = int(g_sample.item())
        if gate == 1:
            st = AgentState(*state_tuple); js = JobSpec(**job_kw)
            r = bidder.bid(st, js, true_bid=true_bid)
            bid = float(r.score); llm_lat = float(r.latency_seconds)
        else:
            bid = float(bid_pred.item())
    return bid, gate, llm_lat


def serve(host, port, agent_id, obs_dim, algo, seed, policy_path,
          connect_timeout_s=120.0):
    from rlsa.bidders.mock import MockLLMBidder

    policy = None
    if algo in ("rlsa", "confidence_gate") and policy_path:
        policy = _make_policy(obs_dim, policy_path)
    bidder = MockLLMBidder(latency_mean=2.0, seed=seed + 7 * agent_id)

    # The coordinator may not be listening yet (it is started after the agents),
    # so retry the connect until it comes up. Tolerate a slow/late coordinator
    # rather than depending on a fragile start order.
    sock = None
    deadline = time.time() + connect_timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=5.0)
            break
        except OSError as e:
            last_err = e
            time.sleep(1.0)
    if sock is None:
        raise OSError(f"agent {agent_id} could not reach coordinator "
                      f"{host}:{port} within {connect_timeout_s}s: {last_err}")
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    send_msg(sock, {"type": "hello", "agent_id": agent_id})

    try:
        while True:
            try:
                msg = recv_msg(sock)
            except EOFError:
                break
            if msg is None or msg.get("type") == "stop":
                break
            if msg.get("type") != "job":
                continue
            t0 = time.time()
            try:
                bid, gate, llm_lat = compute_bid(algo, policy, bidder, msg, seed, agent_id)
                send_msg(sock, {
                    "type": "bid", "round_id": msg["round_id"], "agent_id": agent_id,
                    "bid": bid, "gate": gate, "llm_latency": llm_lat,
                    "compute_s": time.time() - t0,
                })
            except Exception as exc:
                print(f"[agent {agent_id}] compute_bid/send error: {exc}", flush=True)
                raise
    finally:
        sock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--agent-id", type=int, required=True)
    ap.add_argument("--obs-dim", type=int, required=True)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy-path", default="")
    ap.add_argument("--connect-timeout-s", type=float, default=120.0)
    a = ap.parse_args()
    serve(a.host, a.port, a.agent_id, a.obs_dim, a.algo, a.seed, a.policy_path,
          connect_timeout_s=a.connect_timeout_s)


if __name__ == "__main__":
    main()
