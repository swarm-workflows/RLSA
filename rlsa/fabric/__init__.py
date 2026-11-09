"""Real-network deployment: a TCP auction coordinator and bidding agents.

* :mod:`rlsa.fabric.wire`        - length-prefixed message framing
* :mod:`rlsa.fabric.coordinator` - runs the auction, owns the environment
* :mod:`rlsa.fabric.agent`       - serves bids from a frozen policy

Only ``numpy`` and ``torch`` are needed on agent / coordinator hosts.
"""
