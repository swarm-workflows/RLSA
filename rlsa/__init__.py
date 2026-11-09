"""RLSA: Adaptive LLM Gating for Multi-Agent HPC Scheduling Agents.

RLSA trains a per-agent policy that decides, for every incoming job, whether a
bidding agent should pay for a slow LLM bid or trust a fast learned surrogate.

Subpackages are imported lazily so that light-weight deployments (for example a
FABRIC agent VM with only ``numpy`` + ``torch``) do not pull in optional
dependencies such as ``gymnasium`` or ``transformers``:

* :mod:`rlsa.rewards`         composite reward (makespan + utilization + LLM cost)
* :mod:`rlsa.gating`          gating policy network and the confidence-gate baseline
* :mod:`rlsa.bidders`         bidder interface, mock oracle, HuggingFace LLM bidder
* :mod:`rlsa.envs`            controlled gating env, auction env, distributed auction
* :mod:`rlsa.ppo`             PPO trainer for the single-agent gating env
* :mod:`rlsa.fabric`          TCP coordinator / agent used for the FABRIC deployment
"""

__version__ = "1.0.0"
