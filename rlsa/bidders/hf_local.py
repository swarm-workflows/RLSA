"""HuggingFace Transformers-backed local LLM bidder (used with Qwen-2.5 in the paper).

Loads an instruction-tuned model once and answers each ``bid()`` call with a
single integer in [0, 100]. Requires the ``hf`` extra (``pip install rlsa[hf]``).

Design choices
--------------
1. Short ``max_new_tokens`` plus an integer parse instead of grammar-constrained
   decoding; if no integer can be parsed the bidder returns ``fallback_score``.
   Load or generation errors also return ``fallback_score`` with
   ``succeeded=False``.
2. Lazy model load on the first ``bid()`` call, so importing this module is cheap.
3. Greedy decoding by default (``temperature=0``) for reproducibility.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from rlsa.bidders.base import AgentState, BidResult, JobSpec
from rlsa.bidders.prompts import PROMPT_REGISTRY, format_prompt


@dataclass
class HFLocalBidder:
    """Local transformers-based LLM bidder.

    Args:
        model_name: HuggingFace model id (e.g. 'Qwen/Qwen2.5-3B-Instruct').
        dtype: 'float16' or 'bfloat16' (bf16 on A100+); 'auto' lets HF decide.
        max_new_tokens: cap on generation length per bid.
        temperature: 0.0 for greedy (default, reproducible).
        device_map: 'auto' uses accelerate to place on GPU(s).
        fallback_score: returned when generation fails to parse.
    """

    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    dtype: str = "bfloat16"
    max_new_tokens: int = 4
    temperature: float = 0.0
    device_map: str = "auto"
    fallback_score: float = 50.0
    prompt_name: str = "baseline"

    def __post_init__(self):
        # Prompts that include explicit "Answer: NN" framing need a bit
        # more room. We bump max_new_tokens for them iff caller left the
        # default tiny value.
        if self.prompt_name in ("cot_lite",) and self.max_new_tokens <= 8:
            self.max_new_tokens = 16
        if self.prompt_name in ("oneshot",) and self.max_new_tokens <= 8:
            self.max_new_tokens = 8

    _model: object = field(default=None, init=False)
    _tokenizer: object = field(default=None, init=False)
    _device: str = field(default="cpu", init=False)
    _load_seconds: float = field(default=0.0, init=False)

    @property
    def provider_name(self) -> str:
        return f"hf:{self.model_name}"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.perf_counter()
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": "auto",
        }.get(self.dtype, "auto")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Some models have no pad token; use eos for safety
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch_dtype,
            device_map=self.device_map,
        )
        self._model.eval()
        # Best-effort device probe for logging
        try:
            self._device = str(next(self._model.parameters()).device)
        except Exception:
            self._device = "unknown"
        self._load_seconds = time.perf_counter() - t0

    def _format_prompt(self, agent_state: AgentState, job: JobSpec) -> str:
        template = PROMPT_REGISTRY.get(self.prompt_name, PROMPT_REGISTRY["baseline"])
        return format_prompt(template, agent_state, job)

    @staticmethod
    def _parse_score(text: str, fallback: float) -> float:
        # Prefer the "Answer: NN" pattern if present (used by cot_lite and
        # oneshot prompts) — otherwise fall back to first integer found.
        m = re.search(r"answer[:\s]*(-?\d+)", text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"-?\d+", text)
        if not m:
            return fallback
        try:
            v = int(m.group(1) if m.lastindex else m.group())
        except (ValueError, IndexError):
            return fallback
        return float(max(0, min(100, v)))

    def bid(
        self,
        agent_state: AgentState,
        job: JobSpec,
        true_bid: Optional[float] = None,  # ignored; for parity with MockLLMBidder
    ) -> BidResult:
        import torch

        try:
            self._ensure_loaded()
        except Exception as e:
            return BidResult(
                score=self.fallback_score,
                latency_seconds=0.0,
                succeeded=False,
                explanation=f"load failed: {type(e).__name__}: {e}"[:200],
                provider=self.provider_name,
                model=self.model_name,
            )

        prompt = self._format_prompt(agent_state, job)
        # Use chat template if the tokenizer has one (Qwen/Llama do)
        try:
            messages = [{"role": "user", "content": prompt}]
            chat_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            chat_prompt = prompt

        inputs = self._tokenizer(chat_prompt, return_tensors="pt").to(self._device)
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            pad_token_id=self._tokenizer.pad_token_id,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                out = self._model.generate(**inputs, **gen_kwargs)
            latency = time.perf_counter() - t0
            new_tokens = out[0, inputs["input_ids"].shape[1]:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            score = self._parse_score(text, self.fallback_score)
            return BidResult(
                score=score,
                latency_seconds=latency,
                succeeded=True,
                explanation=text[:80],
                provider=self.provider_name,
                model=self.model_name,
                extra={"load_seconds_first_call": self._load_seconds},
            )
        except Exception as e:
            latency = time.perf_counter() - t0
            return BidResult(
                score=self.fallback_score,
                latency_seconds=latency,
                succeeded=False,
                explanation=f"generate failed: {type(e).__name__}: {e}"[:200],
                provider=self.provider_name,
                model=self.model_name,
            )
