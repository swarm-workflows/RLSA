"""Auction coordinator for the real-network (FABRIC) deployment.

The coordinator owns the :class:`~rlsa.envs.auction.AuctionEnv` (the single
source of truth for jobs, cluster and metrics); agents are remote TCP peers
running :mod:`rlsa.fabric.agent`. Each round the job is broadcast to all live
agents concurrently and bids are collected with ``select()`` until a shared
deadline. Replies tagged with an older ``round_id`` are drained and discarded, so
an agent that misses one deadline rejoins in the next round instead of being
excluded. Connections that fail are marked dead and skipped.

The result dict has the same metric keys as the simulated auction, plus
``wan_rtt_ms_{p50,p95,p99}`` and ``round_walltime_ms_{mean,p95}``.

Run on the coordinator VM::

    python -m rlsa.fabric.coordinator --algo rlsa --n-agents 4 --out result.json
"""
from __future__ import annotations

import select
import socket
import threading
import time
from typing import Any, Dict, List

import numpy as np

from rlsa.envs.auction import AuctionConfig, AuctionEnv
from rlsa.fabric.wire import recv_msg, send_msg


def accept_agents(server: socket.socket, n: int, timeout_s: float = 120.0):
    """Block until n agents have dialed in and sent hello. Returns {agent_id: sock}."""
    conns: Dict[int, socket.socket] = {}
    server.settimeout(timeout_s)
    while len(conns) < n:
        sock, _ = server.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hello = recv_msg(sock)
        if hello is None or hello.get("type") != "hello":
            sock.close(); continue
        conns[int(hello["agent_id"])] = sock
    return conns


def run_coordinator(algo: str, cfg: AuctionConfig, seed: int,
                    host: str = "0.0.0.0", port: int = 0,
                    deadline_s: float = 5.0,
                    server: socket.socket = None,
                    conns: Dict[int, socket.socket] = None,
                    extra: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run one auction episode against remote agents.

    If ``conns`` is given the agents are already connected (local harness);
    otherwise a server socket is opened (or ``server`` is used) and the call
    blocks until ``cfg.n_agents`` agents have registered.
    """
    env = AuctionEnv(cfg=cfg, seed=seed)
    env.reset()
    n = cfg.n_agents

    own_server = False
    if conns is None:
        if server is None:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(n)
            own_server = True
        conns = accept_agents(server, n)

    rtt_ms: List[float] = []
    round_walltime_ms: List[float] = []
    n_rounds = 0
    dead_conns: set = set()

    def _send(ai, payload, send_errors):
        try:
            send_msg(conns[ai], payload)
        except Exception as exc:  # noqa: BLE001 -- record, don't lose in the thread
            send_errors[ai] = exc

    try:
        while not env.done():
            env.next_job()
            job = env.get_job_spec()
            true_bid = env.get_true_bid()
            sent_at = {}
            payloads = {}
            for ai in range(n):
                if ai in dead_conns:
                    continue
                obs, state = env.get_obs(ai)
                state_tuple = (state.cpu_util, state.gpu_util, state.ram_util,
                               state.disk_util, state.queue_length,
                               state.neighbor_load, state.historical_success_rate)
                job_kw = dict(cpus=job.cpus, gpus=job.gpus, ram_gb=job.ram_gb,
                              disk_gb=job.disk_gb, estimated_seconds=job.estimated_seconds,
                              user=job.user, job_id=str(env._next_job_id))
                payloads[ai] = {"type": "job", "round_id": n_rounds, "obs": obs.tolist(),
                                "state_tuple": state_tuple, "job_kw": job_kw,
                                "true_bid": true_bid}
            # Concurrent broadcast (don't serialize sends across N), skip dead conns.
            round_t0 = time.time()
            send_errors: Dict[int, Exception] = {}
            threads = []
            for ai in range(n):
                if ai in dead_conns:
                    continue
                sent_at[ai] = time.time()
                t = threading.Thread(target=_send, args=(ai, payloads[ai], send_errors))
                t.start(); threads.append(t)
            for t in threads:
                t.join()
            # A failed send means that agent won't get the job -> don't wait for its
            # bid; surface the error and mark it dead for all future rounds.
            for ai, exc in send_errors.items():
                print(f"[coord] send to agent {ai} failed: {exc}", flush=True)
                dead_conns.add(ai)

            # ---- select()-based bid collection within the shared deadline -------
            # Wait on whichever live sockets become readable; classify each reply
            # by round_id. Current-round bids are recorded; stale (prior-round)
            # replies are drained but NOT counted, so an agent that fell behind can
            # catch up this round instead of being permanently excluded.
            collected = {}
            deadline = time.time() + deadline_s
            # Live agents still owed a bid this round (got the job, not collected).
            pending = {ai for ai in sent_at if ai not in dead_conns}
            sock_to_ai = {conns[ai]: ai for ai in pending}
            # Blocking recv: select() guarantees readability; recv_msg may need
            # several recv() calls to read a full frame, and a deadline-bounded
            # blocking read is simpler/correct vs a per-frame timeout.
            for ai in pending:
                conns[ai].settimeout(None)
            while pending:
                remaining = max(0.0, deadline - time.time())
                if remaining <= 0.0:
                    break
                rlist = [conns[ai] for ai in pending if conns[ai] in sock_to_ai]
                if not rlist:
                    break
                ready, _, _ = select.select(rlist, [], [], remaining)
                if not ready:
                    break  # deadline reached with no more data
                for sock in ready:
                    ai = sock_to_ai.get(sock)
                    if ai is None or ai not in pending:
                        continue
                    # Drain everything currently buffered on this socket so a
                    # one-round-behind agent can catch up within the same round.
                    while True:
                        try:
                            reply = recv_msg(sock)
                        except Exception as exc:  # noqa: BLE001
                            # Mid-message EOF (EOFError) or any other error -> dead.
                            print(f"[coord] recv from agent {ai} failed: {exc}",
                                  flush=True)
                            dead_conns.add(ai)
                            pending.discard(ai)
                            break
                        if reply is None:
                            # Peer closed cleanly before a message started.
                            dead_conns.add(ai)
                            pending.discard(ai)
                            break
                        rid = reply.get("round_id")
                        if reply.get("type") == "bid" and rid == n_rounds:
                            collected[ai] = (reply["bid"], reply["gate"],
                                             reply["llm_latency"])
                            rtt_ms.append((time.time() - sent_at[ai]) * 1000.0)
                            pending.discard(ai)
                            break  # got this round's bid; stop draining
                        if rid is not None and rid < n_rounds:
                            # Stale reply from a prior round: discard, keep listening.
                            # Peek for more buffered data before blocking on recv.
                            r, _, _ = select.select([sock], [], [], 0)
                            if not r:
                                break  # nothing more buffered; wait via outer select
                            continue
                        # Unknown / future / non-bid message: ignore, peek for more.
                        r, _, _ = select.select([sock], [], [], 0)
                        if not r:
                            break
            round_walltime_ms.append((time.time() - round_t0) * 1000.0)
            for ai, (bid, gate, llm_lat) in collected.items():
                env.submit_bid(ai, bid=bid, gate=gate, llm_latency=llm_lat)
            if collected:
                env.resolve_round()
            n_rounds += 1
    finally:
        for ai in range(n):
            if ai in dead_conns:
                continue
            try:
                send_msg(conns[ai], {"type": "stop"})
            except Exception:
                pass
        if own_server and server is not None:
            try:
                server.close()
            except Exception:
                pass

    out = env.finalize()
    out["algo"] = algo
    out["seed"] = seed
    out["n_rounds"] = n_rounds
    out["wan_rtt_ms_p50"] = float(np.percentile(rtt_ms, 50)) if rtt_ms else 0.0
    out["wan_rtt_ms_p95"] = float(np.percentile(rtt_ms, 95)) if rtt_ms else 0.0
    out["wan_rtt_ms_p99"] = float(np.percentile(rtt_ms, 99)) if rtt_ms else 0.0
    out["round_walltime_ms_mean"] = float(np.mean(round_walltime_ms)) if round_walltime_ms else 0.0
    out["round_walltime_ms_p95"] = float(np.percentile(round_walltime_ms, 95)) if round_walltime_ms else 0.0
    out["prototype"] = "fabric"
    if extra:
        out.update(extra)
    return out


def main():
    """Command-line entry point; binds 0.0.0.0 and writes the result JSON."""
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--n-agents", type=int, required=True)
    ap.add_argument("--n-jobs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--deadline-s", type=float, default=5.0)
    ap.add_argument("--out", required=True, help="path to write result JSON on the VM")
    ap.add_argument("--mode", default="", help="pass-through tag for the result")
    ap.add_argument("--netem-loss", type=int, default=0,
                    help="pass-through netem loss%% tag for the result")
    a = ap.parse_args()

    cfg = AuctionConfig(n_agents=a.n_agents, n_jobs=a.n_jobs)
    out = run_coordinator(
        a.algo, cfg, a.seed,
        host="0.0.0.0", port=a.port, deadline_s=a.deadline_s,
        extra={"mode": a.mode, "netem_loss_pct": a.netem_loss,
               "n_agents_run": a.n_agents},
    )
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"[coord] algo={out.get('algo')} n_agents={a.n_agents} "
        f"n_rounds={out.get('n_rounds')} makespan={out.get('makespan')} "
        f"jain_fairness={out.get('jain_fairness')} "
        f"llm_calls_per_job={out.get('llm_calls_per_job')} "
        f"wan_rtt_ms_p50={out.get('wan_rtt_ms_p50')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
