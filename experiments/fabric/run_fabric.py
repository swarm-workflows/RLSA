#!/usr/bin/env python3
"""Drive the real-WAN robustness and scaling sweeps on an ACTIVE FABRIC slice.

This script only orchestrates; it never takes part in the auction. FABNet IPv4
addresses are routable only inside FABRIC, so the coordinator runs on the
coordinator VM and this driver talks to the VMs through fablib.

One cell = one (mode, algo, N, loss, seed) point:
  1. set (or clear) ``tc netem`` loss on the first N agents' FABNet interface
  2. (re)start exactly N agents pointed at the coordinator; stop the others
  3. run ``python3 -m rlsa.fabric.coordinator`` on the coordinator VM (blocking)
  4. download the result JSON to ``--out-dir``
  5. clear netem

All cells evaluate one frozen policy (the paper used the N=4 headline RLSA
checkpoint of seed 0). The observation size does not depend on N, so the same
policy is loaded on every agent for N = 4, 8, 16.

Typical session (from the repository root)::

    python -m experiments.fabric.setup_slice --n-agents 16 --n-sites 3 --yes
    python -m experiments.fabric.setup_slice --pack
    python -m experiments.fabric.run_fabric --mode robustness --provision \
        --ckpt results/auction/headline_mock/policies/rlsa_seed0.pt \
        --ns 4 --netem-loss 0 5 10 20 --out-dir results/fabric/robustness
    python -m experiments.fabric.run_fabric --mode scaling \
        --ckpt results/auction/headline_mock/policies/rlsa_seed0.pt \
        --ns 4 8 16 --out-dir results/fabric/scaling
    python -m experiments.fabric.teardown

Each cell is wrapped in try/except so one failure does not abort the sweep.
"""
from __future__ import annotations

import argparse
import os
import shlex
import time
from pathlib import Path

from experiments.fabric.setup_slice import REMOTE_DIR
from rlsa.envs.auction import AuctionConfig

# ----------------------------------------------------------------------
# fablib helpers (all operate on an ACTIVE slice; no provisioning)
# ----------------------------------------------------------------------

def _get_slice(slice_name):
    from fabrictestbed_extensions.fablib.fablib import FablibManager as fablib_manager
    fablib = fablib_manager()
    return fablib.get_slice(slice_name)


def _nodes_by_name(slice):
    return {n.get_name(): n for n in slice.get_nodes()}


def _agent_iface_dev(node):
    """OS device name of the node's FABNet interface (e.g. 'enp7s0')."""
    site = node.get_site()
    return node.get_interface(network_name=f"net_{site}").get_device_name()


def _agent_iface_ip(node):
    site = node.get_site()
    return node.get_interface(network_name=f"net_{site}").get_ip_addr()


def _apply_netem(slice, n_agents, loss_pct):
    """Set/clear netem loss on the FABNet egress device of agent0..agent{n-1}.

    Always deletes any existing root qdisc first (idempotent), then adds a
    netem loss qdisc only if loss_pct > 0. loss_pct == 0 leaves the device
    clean (no qdisc), which is the natural baseline.
    """
    nodes = _nodes_by_name(slice)
    for ai in range(n_agents):
        node = nodes[f"agent{ai}"]
        dev = _agent_iface_dev(node)
        node.execute(f"sudo tc qdisc del dev {dev} root 2>/dev/null; true")
        if loss_pct > 0:
            node.execute(f"sudo tc qdisc add dev {dev} root netem loss {loss_pct}%")


def _clear_netem(slice, n_agents):
    _apply_netem(slice, n_agents, 0)


def _restart_agents(slice, n_agents, coord_ip, coord_port, algo, obs_dim,
                    seed, ckpt_name):
    """(Re)start exactly n agent servers; pkill stale ones everywhere.

    Agents agent0..agent{n-1} are (re)started pointed at coord_ip:coord_port.
    Any agent with index >= n is pkilled so that, at a smaller N in the scaling
    sweep, exactly n live agents remain (the rest are idle).
    """
    nodes = _nodes_by_name(slice)
    agent_names = sorted(
        (name for name in nodes if name.startswith("agent")),
        key=lambda s: int(s.replace("agent", "")),
    )
    ckpt_arg = ""
    if ckpt_name and algo == "rlsa":
        # Agents start in ~/rlsa, where the checkpoint was uploaded as <basename>.
        ckpt_arg = f"--policy-path {os.path.basename(ckpt_name)}"

    for name in agent_names:
        ai = int(name.replace("agent", ""))
        node = nodes[name]
        # Kill any stale server first (idempotent; `true` so it never fails).
        node.execute("pkill -f rlsa.fabric.agent || true")
        if ai >= n_agents:
            continue  # leave this agent idle for this cell
        # fablib's node.execute() blocks until every process holding the SSH
        # channel exits; `nohup ... &` is not enough, `setsid -f` fully detaches.
        inner = (
            f"cd {REMOTE_DIR} && setsid -f python3 -m rlsa.fabric.agent "
            f"--host {coord_ip} --port {coord_port} --agent-id {ai} "
            f"--obs-dim {obs_dim} --algo {algo} --seed {seed} {ckpt_arg} "
            f"> agent_{ai}.log 2>&1 < /dev/null"
        )
        node.execute(f"bash -lc {shlex.quote(inner)}")


def _download_result(coord_node, remote_path, local_path):
    """Download the result JSON from the coordinator VM. Prefer download_file;
    fall back to `cat` over execute() if download_file is unavailable."""
    try:
        coord_node.download_file(str(local_path), remote_path)
        return
    except (AttributeError, NotImplementedError):
        pass
    stdout, _ = coord_node.execute(f"cat {remote_path}")
    with open(local_path, "w") as f:
        f.write(stdout)


# ----------------------------------------------------------------------
# one cell
# ----------------------------------------------------------------------

def run_cell(slice, coord_node, coord_ip, coord_port, algo, n, n_jobs, seed,
             loss, mode, obs_dim, ckpt_name, out_dir, deadline_s):
    """Drive one (mode, algo, n, loss, seed) cell. Returns local JSON path."""
    # 1. netem on the n agents.
    _apply_netem(slice, n, loss)
    # 2. (re)start exactly n agent servers.
    _restart_agents(slice, n, coord_ip, coord_port, algo, obs_dim, seed, ckpt_name)
    # Give the agents a moment to dial in before the coordinator binds/accepts.
    time.sleep(3.0)

    # 3. run the coordinator ON the coord VM (BLOCKING until auction done).
    vm_out = f"/tmp/coord_{mode}_{algo}_n{n}_loss{loss}_s{seed}.json"
    coord_cmd = (
        f"cd {REMOTE_DIR} && python3 -m rlsa.fabric.coordinator "
        f"--algo {algo} --n-agents {n} --n-jobs {n_jobs} --seed {seed} "
        f"--port {coord_port} --deadline-s {deadline_s} "
        f"--mode {mode} --netem-loss {loss} --out {vm_out}"
    )
    stdout, stderr = coord_node.execute(coord_cmd)
    if stdout:
        print(f"      coord stdout: {stdout.strip().splitlines()[-1] if stdout.strip() else ''}")

    # 4. download the result JSON.
    local_name = f"{mode}_{algo}_n{n}_loss{loss}_s{seed}.json"
    local_path = Path(out_dir) / local_name
    _download_result(coord_node, vm_out, local_path)

    # 5. clear netem so the next cell starts clean.
    _clear_netem(slice, n)
    return local_path


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Orchestrate the RLSA real-WAN robustness/scaling sweep on "
                    "an ACTIVE FABRIC slice (coordinator runs on the coord VM)."
    )
    ap.add_argument("--mode", choices=["robustness", "scaling"], required=True)
    ap.add_argument("--slice-name", default="rlsa-fabric")
    ap.add_argument("--coord-port", type=int, default=9009)
    ap.add_argument("--algos", nargs="+", default=["rlsa", "always_llm"])
    ap.add_argument("--ns", nargs="+", type=int, default=[4])
    ap.add_argument("--netem-loss", nargs="+", type=int, default=[0, 5, 10, 20])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=400)
    ap.add_argument("--deadline-s", type=float, default=5.0)
    ap.add_argument(
        "--ckpt", default="results/auction/headline_mock/policies/rlsa_seed0.pt",
        help="frozen RLSA policy used by every cell (uploaded by --provision)",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--provision", action="store_true",
        help="install the runtime and upload code + checkpoint before the sweep "
             "(run `python -m experiments.fabric.setup_slice --pack` first)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # robustness: sweep the requested losses at fixed N. scaling: losses == [0].
    if args.mode == "robustness":
        losses = args.netem_loss
    else:
        losses = [0]

    obs_dim = AuctionConfig(n_agents=max(args.ns)).obs_dim  # independent of N

    slice = _get_slice(args.slice_name)
    nodes = _nodes_by_name(slice)
    coord_node = nodes["coordinator"]
    coord_ip = _agent_iface_ip(coord_node)
    print(f"[setup] coordinator VM FABNet IP = {coord_ip}:{args.coord_port}")

    if args.provision:
        # Install + upload once (provision also starts agents; every cell
        # restarts them anyway).
        from experiments.fabric.setup_slice import provision

        if not os.path.exists(args.ckpt):
            raise SystemExit(f"checkpoint not found: {args.ckpt}")
        print(f"[provision] install runtime + upload code/ckpt ({args.ckpt})")
        provision(args.slice_name, args.algos[0], obs_dim, 0, args.ckpt,
                  coord_port=args.coord_port)

    results = []
    failures = []
    for algo in args.algos:
        for n in args.ns:
            for loss in losses:
                for seed in range(args.seeds):
                    ckpt_name = args.ckpt if algo == "rlsa" else ""
                    label = f"{args.mode} algo={algo} n={n} loss={loss} seed={seed}"
                    print(f"[cell] {label}")
                    try:
                        path = run_cell(
                            slice, coord_node, coord_ip, args.coord_port,
                            algo, n, args.n_jobs, seed, loss, args.mode,
                            obs_dim, ckpt_name, args.out_dir, args.deadline_s,
                        )
                        print(f"       -> {path}")
                        results.append(str(path))
                    except Exception as exc:  # noqa: BLE001 -- one bad cell != abort
                        print(f"       !! FAILED: {exc}")
                        failures.append((label, repr(exc)))
                        # Best-effort cleanup so a failed cell doesn't leave
                        # netem on the wire for the next one.
                        try:
                            _clear_netem(slice, n)
                        except Exception:
                            pass

    print(f"\n[done] {len(results)} cells OK, {len(failures)} failed.")
    for label, err in failures:
        print(f"  FAIL: {label} :: {err}")


if __name__ == "__main__":
    main()
