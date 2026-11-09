#!/usr/bin/env python3
"""Provision a FABRIC slice for the RLSA real-wide-area experiment.

Topology: 1 coordinator VM + N agent VMs, spread across multiple real
FABRIC sites so that cross-site links are genuine wide-area links. Nodes
are connected via FABNet IPv4.

FABNet IPv4 requires ONE L3 network PER SITE (verified against the official
example ``fablib_api/create_l3network_fabnet_ipv4_auto.ipynb``). Each node
attaches to its own site's network and gets a route so cross-site traffic
flows over the FABNet IPv4 subnet. A single shared network only works
intra-site and would defeat the real-WAN goal.

SAFETY: ``slice.submit()`` actually provisions VMs and CONSUMES allocation.
This script only submits when ``--yes`` is passed. Without ``--yes`` it is a
pure dry run: it prints the planned topology and provisions nothing.

Requires a FABRIC account and ``pip install -e ".[fabric]"`` (fablib reads its
credentials from the usual ``FABRIC_*`` environment / ``fabric_rc``).

Usage (from the repository root)::

    # dry run (no allocation consumed):
    python -m experiments.fabric.setup_slice --n-agents 16 --n-sites 3

    # actually provision (consumes allocation):
    python -m experiments.fabric.setup_slice --n-agents 16 --n-sites 3 --yes

    # build the code tarball that is uploaded to every VM:
    python -m experiments.fabric.setup_slice --pack
"""

import argparse
import os
import subprocess

SLICE_NAME = "rlsa-fabric"

CODE_TARBALL = "rlsa_min.tar.gz"
REMOTE_DIR = "rlsa"  # code is unpacked to ~/rlsa on each VM

# Import closure of rlsa.fabric.{agent,coordinator}; only numpy + torch are
# needed on the VMs.
MIN_PATHS = [
    "rlsa/__init__.py",
    "rlsa/rewards.py",
    "rlsa/bidders/__init__.py",
    "rlsa/bidders/base.py",
    "rlsa/bidders/mock.py",
    "rlsa/envs/__init__.py",
    "rlsa/envs/auction.py",
    "rlsa/gating/__init__.py",
    "rlsa/gating/policy.py",
    "rlsa/gating/confidence_gate.py",
    "rlsa/fabric/__init__.py",
    "rlsa/fabric/wire.py",
    "rlsa/fabric/agent.py",
    "rlsa/fabric/coordinator.py",
]


def pack(out_path=CODE_TARBALL):
    """Tar the minimal ``rlsa`` subset the VMs need. Run from the repository root."""
    missing = [p for p in MIN_PATHS if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"pack: required paths missing (run from the repo root): {missing}")
    subprocess.run(["tar", "--exclude=__pycache__", "-czf", out_path, *MIN_PATHS], check=True)
    print(f"[PACK] wrote {out_path} with {len(MIN_PATHS)} paths")


INSTALL_CMD = (
    "sudo apt-get update -q && sudo apt-get install -y python3-pip iproute2 && "
    "pip3 install --quiet numpy torch --index-url https://download.pytorch.org/whl/cpu"
)


def provision(slice_name, algo, obs_dim, seed, ckpt_name, coord_port=9009):
    """Install the runtime and upload code (+ checkpoint) to every node of an
    ACTIVE slice, then start the agent servers. Assumes ``pack()`` already wrote
    CODE_TARBALL in the current directory. Returns (coord_ip, {agent_id: ip}, port).

    The checkpoint is uploaded to ``~/rlsa/<basename>``; agents are launched from
    ``~/rlsa`` so the policy path is just the basename.
    """
    from fabrictestbed_extensions.fablib.fablib import FablibManager as fablib_manager
    fablib = fablib_manager()
    slice = fablib.get_slice(slice_name)
    nodes = {n.get_name(): n for n in slice.get_nodes()}
    coord = nodes["coordinator"]
    coord_ip = coord.get_interface(network_name=f"net_{coord.get_site()}").get_ip_addr()

    # Install runtime + upload code (+ ckpt) on every node (coordinator needs code too).
    for name, node in nodes.items():
        print(f"[PROVISION] {name}: install runtime")
        node.execute(INSTALL_CMD)
        node.upload_file(CODE_TARBALL, CODE_TARBALL)
        node.execute(f"mkdir -p {REMOTE_DIR} && tar xzf {CODE_TARBALL} -C {REMOTE_DIR}")
        if ckpt_name:
            node.upload_file(ckpt_name, f"{REMOTE_DIR}/{os.path.basename(ckpt_name)}")

    # Start agent servers (agents only, not coordinator).
    agent_ips = {}
    for name, node in nodes.items():
        if not name.startswith("agent"):
            continue
        ai = int(name.replace("agent", ""))
        agent_ips[ai] = node.get_interface(network_name=f"net_{node.get_site()}").get_ip_addr()
        ckpt_arg = f"--policy-path {os.path.basename(ckpt_name)}" if ckpt_name else ""
        cmd = (f"cd {REMOTE_DIR} && setsid -f python3 -m rlsa.fabric.agent "
               f"--host {coord_ip} --port {coord_port} --agent-id {ai} "
               f"--obs-dim {obs_dim} --algo {algo} --seed {seed} {ckpt_arg} "
               f"> agent_{ai}.log 2>&1 < /dev/null")
        print(f"[PROVISION] start {name} -> agent-id {ai} on {node.get_site()}")
        node.execute(cmd)
    return coord_ip, agent_ips, coord_port


def plan(n_agents, sites, site_assignment):
    """Print the planned topology. Makes no fablib calls that cost anything."""
    distinct_sites = sorted(set(site_assignment.values()))
    total_vms = n_agents + 1
    print("[PLAN] FABRIC slice topology")
    print(f"  slice name      : {SLICE_NAME}")
    print("  coordinator VMs : 1")
    print(f"  agent VMs       : {n_agents}")
    print(f"  total VMs       : {total_vms}")
    print(f"  sites requested : {sites}")
    print(f"  distinct sites  : {distinct_sites} ({len(distinct_sites)} used)")
    print("  networking      : FABNet IPv4, one L3 network per site")
    print("  VM -> site assignment:")
    for name, site in site_assignment.items():
        print(f"    {name:>12s} -> {site}")
    print(f"  L3 networks to create: {[f'net_{s}' for s in distinct_sites]}")


def build(fablib, n_agents, sites, site_assignment, image, cores, ram, disk):
    """Construct the slice with the correct per-site-network topology.

    One l3network is created per DISTINCT site actually used; every node is
    attached to its own site's network and gets a route over the FABNet IPv4
    subnet. Does NOT submit -- the caller is responsible for submit().
    """
    slice = fablib.new_slice(name=SLICE_NAME)

    # One L3 network per distinct site that is actually used.
    distinct_sites = sorted(set(site_assignment.values()))
    nets = {}
    for site in distinct_sites:
        nets[site] = slice.add_l3network(name=f"net_{site}", type="IPv4")

    # Add every node, attach it to its site's network, and add the route.
    for name, site in site_assignment.items():
        node = slice.add_node(
            name=name,
            site=site,
            cores=cores,
            ram=ram,
            disk=disk,
            image=image,
        )
        iface = node.add_component(model="NIC_Basic", name="nic1").get_interfaces()[0]
        iface.set_mode("auto")
        nets[site].add_interface(iface)
        node.add_route(subnet=fablib.FABNETV4_SUBNET, next_hop=nets[site].get_gateway())

    return slice


def main():
    parser = argparse.ArgumentParser(
        description="Provision (or dry-run plan) the RLSA FABRIC slice."
    )
    parser.add_argument("--n-agents", type=int, default=16)
    parser.add_argument("--n-sites", type=int, default=3)
    parser.add_argument("--image", default="default_ubuntu_22")
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--ram", type=int, default=8)
    parser.add_argument("--disk", type=int, default=20)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually submit the slice (consumes allocation). "
        "Without this flag the script only prints the plan.",
    )
    parser.add_argument("--pack", action="store_true",
                        help="Tar the minimal rlsa subset for upload and exit.")
    args = parser.parse_args()

    if args.pack:
        pack()
        return

    from fabrictestbed_extensions.fablib.fablib import FablibManager as fablib_manager

    fablib = fablib_manager()

    # This live call needs valid credentials; let it fail loudly if they are
    # missing -- surfacing credential state early is the point of the dry run.
    sites = fablib.get_random_sites(count=args.n_sites)

    # coordinator -> sites[0]; agent i -> sites[i % n_sites] (round-robin).
    site_assignment = {"coordinator": sites[0]}
    for i in range(args.n_agents):
        site_assignment[f"agent{i}"] = sites[i % args.n_sites]

    plan(args.n_agents, sites, site_assignment)

    if not args.yes:
        print("[DRY RUN] re-run with --yes to submit. NOTHING was provisioned.")
        return

    print("[SUBMIT] --yes given: provisioning slice -- this CONSUMES allocation.")
    slice = build(
        fablib,
        args.n_agents,
        sites,
        site_assignment,
        args.image,
        args.cores,
        args.ram,
        args.disk,
    )
    slice.submit()

    print("[DONE] slice submitted. Node addresses:")
    for name, site in site_assignment.items():
        node = slice.get_node(name=name)
        ip = node.get_interface(network_name=f"net_{site}").get_ip_addr()
        print(f"  {name:>12s}  site={site}  ip={ip}")


if __name__ == "__main__":
    main()
