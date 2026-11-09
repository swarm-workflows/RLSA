import numpy as np
import pytest

from rlsa.bidders import capacity_bid
from rlsa.envs.auction import AuctionConfig, AuctionEnv
from rlsa.envs.distributed import DistributedConfig, _compute_gae, run_distributed

ALGOS = ["random", "capacity", "always_surrogate", "always_llm", "oracle", "rlsa"]


def small_cfg(n_agents=3, n_jobs=40, **kw):
    return DistributedConfig(base=AuctionConfig(n_agents=n_agents, n_jobs=n_jobs),
                             pretrain_steps=8, **kw)


# -- synchronous auction env -------------------------------------------------

def test_job_stream_is_seeded():
    a, b = AuctionEnv(seed=1), AuctionEnv(seed=1)
    assert [x[3] for x in a._stream] == [x[3] for x in b._stream]
    assert [x[3] for x in a._stream] != [x[3] for x in AuctionEnv(seed=2)._stream]


def test_synchronous_auction_capacity():
    cfg = AuctionConfig(n_agents=3, n_jobs=30)
    env = AuctionEnv(cfg, seed=0)
    env.reset()
    while not env.done():
        env.next_job()
        for ai in range(cfg.n_agents):
            obs, st = env.get_obs(ai)
            assert obs.shape == (cfg.obs_dim,)
            env.submit_bid(ai, capacity_bid(st.cpu_util, st.gpu_util, st.ram_util, st.disk_util),
                           gate=0, llm_latency=0.0)
        rewards = env.resolve_round()
        assert sum(r != 0 for r in rewards.values()) <= 1
    m = env.finalize()
    assert m["n_jobs"] == 30 and m["makespan"] > 0 and m["llm_calls_per_job"] == 0
    # identical bids + deterministic tiebreak -> agent 0 wins everything
    assert m["jain_fairness"] == pytest.approx(1 / 3)


# -- GAE -----------------------------------------------------------------------

def test_gamma_zero_is_bandit_advantage():
    rew, val = [1.0, -2.0, 0.5, 3.0], [0.2, 0.1, 0.0, 1.0]
    adv, ret = _compute_gae(rew, val, gamma=0.0, lam=0.95)
    assert np.allclose(adv, np.array(rew) - np.array(val))
    assert np.allclose(ret, rew)


def test_gamma_positive_propagates():
    _, ret = _compute_gae([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], gamma=0.99, lam=1.0)
    assert ret[0] == pytest.approx(0.99**2)


# -- threaded distributed auction -----------------------------------------------

@pytest.mark.parametrize("algo", ALGOS)
def test_run_distributed_all_algos(algo):
    out = run_distributed(algo, small_cfg(), seed=0)
    assert out["makespan"] > 0
    assert sum(out["wins_per_agent"].values()) + out["dropped_msgs"] >= 1
    if algo in ("random", "capacity", "always_surrogate"):
        assert out["llm_calls_per_job"] == 0
    if algo == "always_llm":
        assert out["llm_calls_per_job"] == pytest.approx(3.0, abs=0.5)
    assert out["distributed_config"]["n_agents"] == 3


def test_random_tiebreak_spreads_wins():
    det = run_distributed("always_surrogate", small_cfg(n_jobs=60), seed=0)
    rnd = run_distributed("always_surrogate", small_cfg(n_jobs=60, tiebreak="random"), seed=0)
    assert det["jain_fairness"] == pytest.approx(1 / 3)
    assert rnd["jain_fairness"] > det["jain_fairness"]


def test_dtde_runs():
    out = run_distributed("rlsa", small_cfg(sharing_mode="dtde"), seed=0)
    assert out["distributed_config"]["sharing_mode"] == "dtde"


def test_save_load_and_confidence_gate(tmp_path):
    torch = pytest.importorskip("torch")
    from rlsa.gating import GatingConfig, GatingPolicy, train_residual_head

    pol, data = tmp_path / "rlsa.pt", tmp_path / "resid.npz"
    out = run_distributed("rlsa", small_cfg(n_jobs=80, save_policy_path=str(pol),
                                            dump_residual_data=str(data)), seed=0)
    assert pol.exists() and out["saved_policy_path"] == str(pol)

    frozen = run_distributed("rlsa", small_cfg(load_policy_path=str(pol)), seed=1)
    assert frozen["distributed_config"]["freeze_policy"] is True

    if not data.exists():
        pytest.skip("policy never gated after pretraining in this short run")
    d = np.load(data)
    policy = GatingPolicy(GatingConfig(obs_dim=20))
    policy.load_state_dict(torch.load(pol, weights_only=True))
    head, _ = train_residual_head(policy, torch.tensor(d["obs"]), torch.tensor(d["err"]), epochs=50)
    head_path = tmp_path / "residual.pt"
    torch.save(head.state_dict(), head_path)

    never = run_distributed("confidence_gate", small_cfg(load_policy_path=str(pol),
                            residual_path=str(head_path), confidence_tau=1e9), seed=0)
    always = run_distributed("confidence_gate", small_cfg(load_policy_path=str(pol),
                             residual_path=str(head_path), confidence_tau=-1.0), seed=0)
    assert never["llm_calls_per_job"] == 0
    assert always["llm_calls_per_job"] > 2.5
