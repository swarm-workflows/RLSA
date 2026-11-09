# FABRIC deployment

The wide-area study runs the same auction with real processes on real machines.
The **coordinator** ([`rlsa.fabric.coordinator`](api.md#fabric)) owns the
environment (job stream, cluster, metrics) and runs the auction; each **agent**
([`rlsa.fabric.agent`](api.md#fabric)) is a separate process that receives jobs
over TCP and answers with a bid from a frozen policy. Only `numpy` and `torch` are
needed on the VMs.

```text
coordinator VM                                agent VMs (x N)
--------------                                ---------------
AuctionEnv: next job, observations   --job-->  frozen GatingPolicy + mock bidder
select() until deadline              <--bid--  (bid, gate, llm_latency)
argmax -> assign -> metrics
```

Each round is broadcast to all live agents concurrently and bids are collected with
`select()` until a shared deadline (5 s). Replies from an earlier round are
discarded, so an agent that misses one deadline rejoins at the next round. The
wire format is a 4-byte length prefix followed by a pickled dict; **only use it
between trusted hosts.**

## Try it locally first

The local harness runs the coordinator and N agents over `127.0.0.1` with the real
wire protocol:

```bash
python -m rlsa.fabric.local --algo rlsa --n-agents 4 --n-jobs 50 \
    --policy-path results/auction/headline_mock/policies/rlsa_seed0.pt
```

## Running on FABRIC

You need a [FABRIC](https://portal.fabric-testbed.net) project, a configured
`fablib` (`fabric_rc`, bastion key, slice key) and `pip install -e ".[fabric]"`.
A slice consumes allocation while it exists, so always tear it down.

```bash
# 1. plan, then provision 1 coordinator + 16 agents round-robin over 3 random sites
python -m experiments.fabric.setup_slice --n-agents 16 --n-sites 3          # dry run
python -m experiments.fabric.setup_slice --n-agents 16 --n-sites 3 --yes    # submit

# 2. package the code that is uploaded to every VM (-> rlsa_min.tar.gz)
python -m experiments.fabric.setup_slice --pack

# 3. robustness: N=4, tc-netem loss 0/5/10/20 %, 3 seeds (installs code + checkpoint first)
python -m experiments.fabric.run_fabric --mode robustness --provision \
    --ckpt results/auction/headline_mock/policies/rlsa_seed0.pt \
    --algos rlsa always_llm --ns 4 --netem-loss 0 5 10 20 --seeds 3 \
    --out-dir runs/fabric/robustness

# 4. scaling: N = 4, 8, 16, no injected loss
python -m experiments.fabric.run_fabric --mode scaling \
    --ckpt results/auction/headline_mock/policies/rlsa_seed0.pt \
    --algos rlsa always_llm --ns 4 8 16 --seeds 3 --out-dir runs/fabric/scaling

# 5. delete the slice
python -m experiments.fabric.teardown
```

The same steps are in `experiments/reproduce/fabric.sh`.

For each cell, `run_fabric` sets `tc netem` loss on the agents' FABNet interface,
restarts exactly N agents (detached with `setsid -f`), and runs the coordinator on
the coordinator VM. It then downloads the result JSON and clears the loss setting.
The driver never joins the auction itself: FABNet addresses are only routable
inside FABRIC.

The paper's slice used the Starlight (Chicago), Michigan and Massachusetts sites;
`setup_slice.py` picks random available sites, so expect different RTTs.
