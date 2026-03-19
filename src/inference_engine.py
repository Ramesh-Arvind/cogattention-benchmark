"""vLLM inference wrapper with OOM resilience and context truncation."""

import gc
import logging
import time
from typing import Any, Dict, Optional, Tuple

import torch

from src.eval_config import GENERATION_PARAMS, ModelConfig, get_hf_token

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Wraps vLLM offline LLM for benchmark inference."""

    def __init__(self, model_config: ModelConfig):
        from vllm import LLM, SamplingParams

        self.config = model_config
        self.max_model_len = model_config.max_model_len
        self.max_tokens = GENERATION_PARAMS["max_tokens"]

        logger.info(
            "Loading model %s (tp=%d, max_len=%d)",
            model_config.hf_id,
            model_config.tensor_parallel_size,
            model_config.max_model_len,
        )

        hf_token = get_hf_token()

        self.llm = LLM(
            model=model_config.hf_id,
            tensor_parallel_size=model_config.tensor_parallel_size,
            max_model_len=model_config.max_model_len,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=0.80,
            download_dir=None,
            **model_config.extra_kwargs,
        )

        self.tokenizer = self.llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=GENERATION_PARAMS["temperature"],
            top_p=GENERATION_PARAMS["top_p"],
            max_tokens=self.max_tokens,
        )

        logger.info("Model %s loaded successfully.", model_config.name)

    # ------------------------------------------------------------------
    # Context truncation
    # ------------------------------------------------------------------

    def _truncate_if_needed(self, token_ids: list) -> list:
        """Truncate middle tokens if prompt exceeds budget, keeping edges."""
        budget = self.max_model_len - self.max_tokens - 16  # small safety margin
        if len(token_ids) <= budget:
            return token_ids

        keep_edge = 512
        head = token_ids[:keep_edge]
        tail = token_ids[-keep_edge:]
        truncated = head + tail
        logger.warning(
            "Prompt truncated: %d -> %d tokens (budget %d)",
            len(token_ids),
            len(truncated),
            budget,
        )
        return truncated

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, prompt: str, task_id: str) -> Tuple[str, Dict[str, Any]]:
        """Run inference on a single prompt. Returns (response_text, metadata).

        On OOM or RuntimeError, returns ("[OOM_ERROR]", metadata) instead of raising.
        """
        metadata: Dict[str, Any] = {
            "task_id": task_id,
            "model": self.config.name,
        }

        try:
            # Apply chat template
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize to count & possibly truncate
            token_ids = self.tokenizer.encode(formatted)
            metadata["prompt_tokens"] = len(token_ids)
            logger.info(
                "[%s] prompt_tokens=%d", task_id, len(token_ids)
            )

            token_ids = self._truncate_if_needed(token_ids)
            metadata["prompt_tokens_after_trunc"] = len(token_ids)

            # Run inference via vLLM
            from vllm import TokensPrompt

            t0 = time.time()
            outputs = self.llm.generate(
                TokensPrompt(prompt_token_ids=token_ids),
                sampling_params=self.sampling_params,
            )
            elapsed = time.time() - t0

            response_text = outputs[0].outputs[0].text
            metadata["generation_tokens"] = len(outputs[0].outputs[0].token_ids)
            metadata["latency_s"] = round(elapsed, 2)

            return response_text, metadata

        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            logger.error("[%s] OOM/RuntimeError: %s", task_id, exc)
            torch.cuda.empty_cache()
            gc.collect()
            metadata["error"] = str(exc)
            return "[OOM_ERROR]", metadata

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        """Release GPU memory."""
        logger.info("Shutting down engine for %s", self.config.name)
        if hasattr(self, "llm"):
            del self.llm
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Engine shutdown complete.")
