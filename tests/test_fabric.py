"""The real-network transport, exercised over 127.0.0.1 TCP sockets."""

import socket
import threading
import time

from rlsa.bidders import capacity_bid
from rlsa.envs.auction import AuctionConfig
from rlsa.fabric.agent import serve
from rlsa.fabric.coordinator import accept_agents, run_coordinator
from rlsa.fabric.local import run_local
from rlsa.fabric.wire import recv_msg, send_msg


def test_wire_roundtrip():
    a, b = socket.socketpair()
    try:
        msg = {"type": "bid", "round_id": 3, "bid": 42.5, "obs": list(range(100))}
        send_msg(a, msg)
        assert recv_msg(b) == msg
        a.close()
        assert recv_msg(b) is None  # clean close
    finally:
        b.close()


def test_local_e2e_all_agents_participate():
    out = run_local("always_surrogate", n_agents=3, n_jobs=30, seed=0)
    assert out["n_rounds"] == 30
    assert all(out["bids_per_agent"].get(ai, 0) > 0 for ai in range(3))
    assert out["makespan"] > 0 and out["wan_rtt_ms_p50"] >= 0


def test_local_rlsa_with_checkpoint(tmp_path):
    import torch

    from rlsa.gating import GatingConfig, GatingPolicy

    path = tmp_path / "policy.pt"
    torch.save(GatingPolicy(GatingConfig(obs_dim=20)).state_dict(), path)
    out = run_local("rlsa", n_agents=2, n_jobs=20, seed=0, policy_path=str(path))
    assert out["n_rounds"] == 20 and 0.0 <= out["llm_calls_per_job"] <= 2.0


def _slow_agent(host, port, agent_id, deadline_s):
    """Capacity bidder that misses round 1's deadline, then behaves."""
    sock = socket.create_connection((host, port))
    send_msg(sock, {"type": "hello", "agent_id": agent_id})
    try:
        while True:
            msg = recv_msg(sock)
            if msg is None or msg.get("type") == "stop":
                break
            if msg["round_id"] == 1:
                time.sleep(deadline_s + 0.5)
            send_msg(sock, {"type": "bid", "round_id": msg["round_id"], "agent_id": agent_id,
                            "bid": capacity_bid(*msg["state_tuple"][:4]), "gate": 0,
                            "llm_latency": 0.0, "compute_s": 0.0})
    except (EOFError, OSError):
        pass
    finally:
        sock.close()


def test_slow_agent_rejoins_after_missed_deadline():
    n_agents, n_jobs, deadline_s = 3, 15, 1.0
    cfg = AuctionConfig(n_agents=n_agents, n_jobs=n_jobs)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(n_agents)
    port = server.getsockname()[1]
    threads = [threading.Thread(target=_slow_agent, args=("127.0.0.1", port, 0, deadline_s), daemon=True)]
    threads += [threading.Thread(target=serve, args=("127.0.0.1", port, ai, cfg.obs_dim,
                                                     "always_surrogate", 0, ""), daemon=True)
                for ai in range(1, n_agents)]
    for t in threads:
        t.start()
    out = run_coordinator("always_surrogate", cfg, 0, conns=accept_agents(server, n_agents),
                          deadline_s=deadline_s)
    server.close()
    assert out["bids_per_agent"].get(0, 0) >= out["n_rounds"] - 3
