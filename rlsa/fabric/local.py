"""Run the coordinator and N agents over 127.0.0.1 TCP in one process.

No testbed needed: every bid crosses a real socket, so this exercises the full
wire path. Used by the tests and as a pre-flight check before a FABRIC run.
"""

from __future__ import annotations

import socket
import threading
from typing import Any, Dict

from rlsa.envs.auction import AuctionConfig
from rlsa.fabric.agent import serve
from rlsa.fabric.coordinator import accept_agents, run_coordinator


def run_local(
    algo: str,
    n_agents: int,
    n_jobs: int,
    seed: int,
    policy_path: str = "",
    deadline_s: float = 5.0,
) -> Dict[str, Any]:
    cfg = AuctionConfig(n_agents=n_agents, n_jobs=n_jobs)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(n_agents)
    port = server.getsockname()[1]

    threads = []
    for ai in range(n_agents):
        t = threading.Thread(
            target=serve,
            args=("127.0.0.1", port, ai, cfg.obs_dim, algo, seed, policy_path),
            daemon=True,
        )
        t.start()
        threads.append(t)

    conns = accept_agents(server, n_agents)
    out = run_coordinator(
        algo, cfg, seed, conns=conns, deadline_s=deadline_s,
        extra={"mode": "local", "n_agents_run": n_agents},
    )
    server.close()
    for t in threads:
        t.join(timeout=5.0)
    return out


def main() -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", default="always_surrogate")
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy-path", default="")
    a = ap.parse_args()
    t0 = time.time()
    out = run_local(a.algo, a.n_agents, a.n_jobs, a.seed, a.policy_path)
    out["wall_s"] = time.time() - t0
    keys = ("algo", "n_agents", "n_jobs", "n_rounds", "makespan", "jain_fairness",
            "llm_calls_per_job", "wan_rtt_ms_p50", "round_walltime_ms_mean", "wall_s")
    print({k: out[k] for k in keys if k in out})


if __name__ == "__main__":
    main()
