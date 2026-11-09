"""Prompt variants for HFLocalBidder.

We keep multiple candidate prompt templates here and let the bidder
pick one by name. This makes it easy to A/B test prompt design.
"""

from __future__ import annotations

# v0: original baseline. Underweights free capacity when job is "big".
PROMPT_BASELINE = """You are a bidding agent in a decentralized HPC scheduling system. \
Score how well your resources match the incoming job request on a 0..100 integer \
scale, where 100 = perfect match (job fits easily and uses idle resources) and \
0 = job cannot run or would severely overcommit.

YOUR CURRENT RESOURCES:
  CPU utilization : {cpu_util:.0%}
  GPU utilization : {gpu_util:.0%}
  RAM utilization : {ram_util:.0%}
  Disk utilization: {disk_util:.0%}
  Queue length    : {queue_len} jobs waiting
  Neighbor load   : {neighbor_load:.0%}
  Historical success rate: {hist_success:.0%}

JOB REQUEST:
  CPUs : {cpus}
  GPUs : {gpus}
  RAM  : {ram_gb:.1f} GB
  Disk : {disk_gb:.1f} GB
  Estimated runtime: {est_seconds:.0f} s
  User : {user}

Reply with a single integer score in [0, 100]. No explanation, no units. Just the number.
"""


# v1: explicit "free capacity beats job size" heuristic
PROMPT_EXPLICIT = """You are a bidding agent in a decentralized HPC scheduling system. \
Output one integer in [0, 100] that scores how well your free resources match this job.

SCORING RULE:
  * If your free capacity comfortably covers the job's requirements,
    score HIGH (70-100) regardless of how large the job is.
  * If accepting the job would push utilization above ~90% on any
    dimension, score LOW (0-40).
  * Idle nodes should give the highest scores even for large jobs,
    because they have the most headroom.

YOUR CURRENT RESOURCES (lower utilization = more free capacity):
  CPU utilization : {cpu_util:.0%}   (free: {cpu_free:.0%})
  GPU utilization : {gpu_util:.0%}   (free: {gpu_free:.0%})
  RAM utilization : {ram_util:.0%}   (free: {ram_free:.0%})
  Disk utilization: {disk_util:.0%}   (free: {disk_free:.0%})
  Queue length    : {queue_len} jobs waiting
  Neighbor load   : {neighbor_load:.0%}
  Historical success rate: {hist_success:.0%}

JOB REQUEST:
  CPUs : {cpus}
  GPUs : {gpus}
  RAM  : {ram_gb:.1f} GB
  Disk : {disk_gb:.1f} GB
  Estimated runtime: {est_seconds:.0f} s
  User : {user}

Reply with a single integer score in [0, 100]. No explanation, no units. Just the number.
"""


# v2: chain-of-thought-lite (the model emits "<reason> Answer: NN" so we parse the int)
# Use a slightly longer max_new_tokens with this one.
PROMPT_COT_LITE = """You are a bidding agent in a decentralized HPC scheduling system. \
First think briefly about whether your free resources are enough for the job, then \
output a single integer in [0, 100].

YOUR CURRENT RESOURCES:
  CPU free: {cpu_free:.0%}  GPU free: {gpu_free:.0%}  RAM free: {ram_free:.0%}  Disk free: {disk_free:.0%}
  Queue length: {queue_len} jobs   Neighbor load: {neighbor_load:.0%}   Historical success: {hist_success:.0%}

JOB REQUEST:
  CPUs {cpus}, GPUs {gpus}, RAM {ram_gb:.1f} GB, Disk {disk_gb:.1f} GB, ETA {est_seconds:.0f} s

THINK STEP BY STEP:
  1. Do you have enough of the tightest resource the job needs?
  2. Would accepting overcommit any dimension?
  3. Is the queue too deep?
Pick a score in [0, 100] reflecting these answers (high = great match, low = poor).

Respond ONLY with: Answer: NN
"""


# v3: one-shot worked example, then ask
PROMPT_ONESHOT = """You are a bidding agent in a decentralized HPC scheduling system. \
Score how well your free resources match an incoming job request on a 0..100 integer scale.

WORKED EXAMPLE:
  Resources: CPU free 90%, GPU free 100%, RAM free 85%, Disk free 90%, Queue 1
  Job: 32 CPUs, 4 GPUs, 128 GB RAM, 500 GB disk
  Reasoning: We are nearly idle and have plenty of free capacity even for this
  large job. Score should be high.
  Answer: 85

NOW SCORE THIS JOB:
  Resources: CPU free {cpu_free:.0%}, GPU free {gpu_free:.0%}, RAM free {ram_free:.0%}, \
Disk free {disk_free:.0%}, Queue {queue_len}, Neighbor load {neighbor_load:.0%}
  Job: {cpus} CPUs, {gpus} GPUs, {ram_gb:.1f} GB RAM, {disk_gb:.1f} GB disk, ETA {est_seconds:.0f}s

Reply with ONLY: Answer: NN
"""


PROMPT_REGISTRY = {
    "baseline": PROMPT_BASELINE,
    "explicit": PROMPT_EXPLICIT,
    "cot_lite": PROMPT_COT_LITE,
    "oneshot": PROMPT_ONESHOT,
}


def format_prompt(template: str, agent_state, job) -> str:
    """Render a prompt template with the given AgentState and JobSpec."""
    return template.format(
        cpu_util=agent_state.cpu_util,
        gpu_util=agent_state.gpu_util,
        ram_util=agent_state.ram_util,
        disk_util=agent_state.disk_util,
        cpu_free=1.0 - agent_state.cpu_util,
        gpu_free=1.0 - agent_state.gpu_util,
        ram_free=1.0 - agent_state.ram_util,
        disk_free=1.0 - agent_state.disk_util,
        queue_len=agent_state.queue_length,
        neighbor_load=agent_state.neighbor_load,
        hist_success=agent_state.historical_success_rate,
        cpus=job.cpus,
        gpus=job.gpus,
        ram_gb=job.ram_gb,
        disk_gb=job.disk_gb,
        est_seconds=job.estimated_seconds,
        user=job.user or "anonymous",
    )
