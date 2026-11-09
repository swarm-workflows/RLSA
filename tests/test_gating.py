import torch

from rlsa.gating import (
    ConfidenceGate,
    GatingConfig,
    GatingPolicy,
    ResidualHead,
    train_residual_head,
)


def test_policy_shapes_and_ranges():
    policy = GatingPolicy(GatingConfig(obs_dim=20, hidden_dim=64))
    bid, p_gate, value, latent = policy(torch.randn(8, 20))
    assert bid.shape == p_gate.shape == value.shape == (8,)
    assert latent.shape == (8, 64)
    assert 0.0 <= bid.min().item() and bid.max().item() <= 100.0
    assert 0.0 <= p_gate.min().item() and p_gate.max().item() <= 1.0


def test_act_and_threshold():
    policy = GatingPolicy(GatingConfig(obs_dim=10))
    obs = torch.randn(1, 10)
    out = policy.act(obs)
    assert set(out) == {"gate", "bid", "p_gate", "value", "latent"}
    assert out["gate"].item() in (0, 1)
    # moving the threshold changes the deterministic gate monotonically
    policy.cfg.gate_threshold = -1.0
    assert policy.act(obs, deterministic=True)["gate"].item() == 1
    policy.cfg.gate_threshold = 2.0
    assert policy.act(obs, deterministic=True)["gate"].item() == 0


def test_evaluate():
    policy = GatingPolicy(GatingConfig(obs_dim=10))
    actions = torch.bernoulli(torch.full((16,), 0.5))
    log_p, v, ent = policy.evaluate(torch.randn(16, 10), actions)
    assert log_p.shape == v.shape == ent.shape == (16,)
    assert (log_p <= 0).all() and (ent >= 0).all()


def test_state_dict_keys_are_stable():
    # Released checkpoints in results/ rely on these parameter names.
    keys = set(GatingPolicy(GatingConfig(obs_dim=20)).state_dict())
    assert {"encoder.net.0.weight", "surrogate.head.weight", "gate.head.weight",
            "value.head.weight"} <= keys


def test_confidence_gate_monotone_in_tau():
    policy = GatingPolicy(GatingConfig(obs_dim=20))
    head = ResidualHead(hidden_dim=128)
    obs = torch.randn(1, 20)
    assert ConfidenceGate(policy, head, tau=0.0).decide(obs)[0] == 1
    gate, bid, err = ConfidenceGate(policy, head, tau=1e9).decide(obs)
    assert gate == 0 and 0.0 <= bid <= 100.0 and err >= 0.0


def test_train_residual_head_fits():
    torch.manual_seed(0)
    policy = GatingPolicy(GatingConfig(obs_dim=20))
    obs = torch.rand(64, 20)
    err = 10.0 * obs[:, 8]  # error grows with the difficulty slot
    head, mse = train_residual_head(policy, obs, err, epochs=300)
    assert mse < err.var().item()
