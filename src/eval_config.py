"""Configuration & Task Registry for local LLM evaluation."""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from src.generators.base import TaskInstance
from src.scorers.base import ScoreResult

# ---------------------------------------------------------------------------
# Model Configurations
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    hf_id: str
    tensor_parallel_size: int
    max_model_len: int
    # Optional vLLM kwargs
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "qwen72b": ModelConfig(
        name="qwen72b",
        hf_id="Qwen/Qwen2.5-72B-Instruct",
        tensor_parallel_size=2,
        max_model_len=32768,
    ),
    "llama8b": ModelConfig(
        name="llama8b",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        max_model_len=32768,
    ),
    "phi3": ModelConfig(
        name="phi3",
        hf_id="microsoft/Phi-3.5-mini-instruct",
        tensor_parallel_size=1,
        max_model_len=16384,
    ),
}

# ---------------------------------------------------------------------------
# Generation Parameters (deterministic)
# ---------------------------------------------------------------------------

GENERATION_PARAMS: Dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 2048,
}

# ---------------------------------------------------------------------------
# HuggingFace Token
# ---------------------------------------------------------------------------

_TOKEN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "huggingface", "token.txt"
)


def get_hf_token() -> str:
    """Read HF token from config file."""
    with open(_TOKEN_PATH) as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Task Registry: task_type -> (generator_fn, scorer_fn)
# ---------------------------------------------------------------------------

# Lazy imports to avoid loading heavy modules at config import time.
_TASK_REGISTRY_CACHE: Dict[str, Tuple[Callable[..., List[TaskInstance]], Callable[[TaskInstance, str], ScoreResult]]] = {}


def _build_registry() -> Dict[str, Tuple[Callable, Callable]]:
    if _TASK_REGISTRY_CACHE:
        return _TASK_REGISTRY_CACHE

    from src.generators.capacity import generate_capacity_dataset
    from src.generators.novel_capacity import generate_interference_dataset
    from src.generators.attentional_blink import generate_blink_dataset
    from src.generators.sustained import generate_sustained_dataset
    from src.generators.novel_sustained import generate_stream_dataset
    from src.generators.context_dilution import generate_dilution_dataset
    from src.generators.semantic_niah import generate_sniah_dataset
    from src.generators.multihop_attention import generate_multihop_dataset
    from src.generators.selective import generate_selective_dataset
    from src.generators.novel_selective import generate_stroop_dataset
    from src.generators.flanker import generate_flanker_dataset
    from src.generators.shifting import generate_shifting_dataset
    from src.generators.inhibition_return import generate_ior_dataset
    from src.generators.anomaly import generate_anomaly_dataset

    from src.scorers.capacity import score_capacity, score_interference, score_blink
    from src.scorers.sustained import score_sustained, score_stream, score_dilution, score_sniah, score_multihop
    from src.scorers.selective import score_selective, score_stroop, score_flanker
    from src.scorers.shifting import score_shifting, score_ior
    from src.scorers.anomaly import score_anomaly

    registry = {
        "capacity": (generate_capacity_dataset, score_capacity),
        "interference": (generate_interference_dataset, score_interference),
        "blink": (generate_blink_dataset, score_blink),
        "sustained": (generate_sustained_dataset, score_sustained),
        "stream_segregation": (generate_stream_dataset, score_stream),
        "context_dilution": (generate_dilution_dataset, score_dilution),
        "semantic_niah": (generate_sniah_dataset, score_sniah),
        "multihop": (generate_multihop_dataset, score_multihop),
        "selective": (generate_selective_dataset, score_selective),
        "stroop": (generate_stroop_dataset, score_stroop),
        "flanker": (generate_flanker_dataset, score_flanker),
        "shifting": (generate_shifting_dataset, score_shifting),
        "inhibition_return": (generate_ior_dataset, score_ior),
        "anomaly": (generate_anomaly_dataset, score_anomaly),
    }
    _TASK_REGISTRY_CACHE.update(registry)
    return _TASK_REGISTRY_CACHE


def get_task_registry() -> Dict[str, Tuple[Callable, Callable]]:
    """Return the task registry, building it on first call."""
    return _build_registry()
