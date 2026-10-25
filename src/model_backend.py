"""Frozen Qwen3-8B backend shared by instruction proposal and classification.

Hard constraints enforced here (per project requirements):
  - transformers + official tokenizer/chat template; enable_thinking=False always.
  - BF16 weights.
  - Every parameter has requires_grad=False; model.eval().
  - torch.inference_mode() wraps every call.
  - Trainable-parameter count is asserted to be exactly zero.
  - One checkpoint is loaded once and reused for both instruction proposal and
    classification (no second LLM).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"


@dataclass
class GenerationResult:
    raw_outputs: list[str]
    input_token_counts: list[int]
    output_token_counts: list[int]
    wall_clock_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    num_invocations: int = 1


@dataclass
class QwenBackend:
    model_revision: str
    tokenizer: object = None
    model: object = None
    device: str = "cuda"

    @classmethod
    def load(cls, revision: str | None = None, dtype=torch.bfloat16):
        revision = revision or _resolve_revision()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=revision, torch_dtype=dtype, device_map="cuda"
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == 0, f"Expected zero trainable parameters, found {trainable}"

        return cls(model_revision=revision, tokenizer=tokenizer, model=model)

    @torch.inference_mode()
    def generate_batch(self, prompt_strs: list[str], max_new_tokens: int, do_sample: bool,
                        temperature: float = 1.0, top_p: float = 1.0,
                        seed: int | None = None) -> GenerationResult:
        if seed is not None:
            torch.manual_seed(seed)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        enc = self.tokenizer(prompt_strs, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        input_token_counts = enc["attention_mask"].sum(dim=1).tolist()

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None

        out = self.model.generate(**enc, **gen_kwargs)

        input_len = enc["input_ids"].shape[1]
        new_tokens = out[:, input_len:]
        output_token_counts = (new_tokens != self.tokenizer.pad_token_id).sum(dim=1).tolist()
        raw_outputs = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        wall = time.perf_counter() - t0
        return GenerationResult(
            raw_outputs=raw_outputs,
            input_token_counts=input_token_counts,
            output_token_counts=output_token_counts,
            wall_clock_seconds=wall,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )

    def software_config(self) -> dict:
        import transformers
        return {
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "model_id": MODEL_ID,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.model_revision,
            "dtype": "bfloat16",
        }


def _resolve_revision() -> str:
    from huggingface_hub import HfApi
    return HfApi().model_info(MODEL_ID).sha
