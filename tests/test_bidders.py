import pytest

from rlsa.bidders import AgentState, JobSpec, MockLLMBidder, capacity_bid
from rlsa.bidders.prompts import PROMPT_REGISTRY, format_prompt

STATE = AgentState(0.2, 0.0, 0.5, 0.1, queue_length=3, neighbor_load=0.4)
JOB = JobSpec(cpus=8, gpus=1, ram_gb=16.0, disk_gb=50.0, estimated_seconds=60.0)


def test_mock_returns_true_bid_and_reports_latency():
    r = MockLLMBidder(seed=0).bid(STATE, JOB, true_bid=42.0)
    assert r.score == 42.0 and r.succeeded and r.latency_seconds >= 0.01


def test_mock_clips_and_falls_back():
    b = MockLLMBidder(seed=0)
    assert b.bid(STATE, JOB, true_bid=250.0).score == 100.0
    assert 0.0 <= b.bid(STATE, JOB).score <= 100.0


def test_capacity_bid():
    assert capacity_bid(0, 0, 0, 0) == pytest.approx(100.0)
    assert capacity_bid(1, 1, 1, 1) == pytest.approx(0.0)
    assert capacity_bid(1, 0, 1, 1) == pytest.approx(10.0)  # GPU weight 0.1


@pytest.mark.parametrize("name", sorted(PROMPT_REGISTRY))
def test_prompts_render(name):
    text = format_prompt(PROMPT_REGISTRY[name], STATE, JOB)
    assert "{" not in text and "8" in text


def test_hf_score_parsing():
    pytest.importorskip("torch")
    from rlsa.bidders.hf_local import HFLocalBidder

    parse = HFLocalBidder._parse_score
    assert parse("73", 50.0) == 73.0
    assert parse("Reasoning... Answer: 85", 50.0) == 85.0
    assert parse("150", 50.0) == 100.0
    assert parse("no number", 50.0) == 50.0
