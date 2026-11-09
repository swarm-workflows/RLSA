"""The released results reproduce the numbers printed in the paper."""

import pytest

from experiments.auction.summarize import summarize
from experiments.common import RESULTS

AUCTION = RESULTS / "auction"


def row(rows, algo):
    return next(r for r in rows if r["algo"] == algo)


def test_table1_mock():
    rows = summarize(AUCTION / "headline_mock", n_boot=100)
    assert row(rows, "rlsa")["seeds"] == 10
    assert row(rows, "rlsa")["makespan_mean"] == pytest.approx(7686, abs=1)
    assert row(rows, "rlsa")["llm_per_job_mean"] == pytest.approx(2.06, abs=0.005)
    assert row(rows, "rlsa")["jain_mean"] == pytest.approx(0.679, abs=0.001)
    assert row(rows, "always_llm")["makespan_mean"] == pytest.approx(7285, abs=1)
    assert row(rows, "random")["makespan_mean"] == pytest.approx(14663, abs=1)
    assert row(rows, "oracle")["llm_per_job_mean"] == pytest.approx(1.22, abs=0.005)


def test_table1_confidence_gate():
    rows = summarize(AUCTION / "confidence_gate", algos="confidence_gate", prefix="tau10_", n_boot=100)
    r = row(rows, "confidence_gate")
    assert r["makespan_mean"] == pytest.approx(7562, abs=1)
    assert r["llm_per_job_mean"] == pytest.approx(1.38, abs=0.005)
    assert r["jain_mean"] == pytest.approx(0.822, abs=0.001)


def test_table1_qwen7b():
    rows = summarize(AUCTION / "headline_qwen7b", n_boot=100)
    assert row(rows, "rlsa")["makespan_mean"] == pytest.approx(6120, abs=1)
    assert row(rows, "rlsa")["llm_per_job_mean"] == pytest.approx(3.44, abs=0.005)
    assert row(rows, "always_llm")["makespan_mean"] == pytest.approx(6222, abs=1)


def test_gamma_ablation():
    llm = {g: row(summarize(AUCTION / "gamma", algos="rlsa", prefix=f"g{g}_", n_boot=10), "rlsa")
           ["llm_per_job_mean"] for g in ("0.0", "0.95", "0.99")}
    assert llm == pytest.approx({"0.0": 2.02, "0.95": 2.89, "0.99": 3.35}, abs=0.01)


def test_long_horizon():
    r = row(summarize(AUCTION / "long_horizon", prefix="long_", n_boot=10), "rlsa")
    assert r["llm_per_job_mean"] == pytest.approx(1.03, abs=0.005)
    assert r["makespan_mean"] == pytest.approx(14583, abs=2)


def test_ablation_table():
    import csv

    rows = list(csv.DictReader(open(RESULTS / "single_agent/ablation/ablation.csv")))
    full = [float(r["gate_hard"]) for r in rows if r["arm"] == "full"]
    disc = [float(r["gate_hard"]) for r in rows if r["arm"] == "discounted"]
    assert 100 * sum(full) / len(full) == pytest.approx(95.6, abs=0.1)
    assert max(disc) == 0.0
