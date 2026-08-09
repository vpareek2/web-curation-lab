"""Project-owned language model definitions."""

from .gpt_oss_dense import ARCHITECTURES as GPT_OSS_DENSE_ARCHITECTURES
from .gpt_oss_dense import (
    GPT_OSS_DENSE_REFERENCE_150M,
    GPT_OSS_DENSE_WIDE_150M,
    DenseGptOssArchitecture,
)
from .gpt_oss_dense import model_registry as gpt_oss_dense_model_registry
from .qwen3 import ARCHITECTURES as QWEN3_ARCHITECTURES
from .qwen3 import (
    REFERENCE_150M,
    WIDE_150M,
    Qwen3Architecture,
)
from .qwen3 import model_registry as qwen3_model_registry
from .qwen3_5 import ARCHITECTURES as QWEN35_ARCHITECTURES
from .qwen3_5 import HYBRID_REFERENCE_150M, HYBRID_WIDE_150M, Qwen35Architecture
from .qwen3_5 import model_registry as qwen35_model_registry

__all__ = [
    "HYBRID_REFERENCE_150M",
    "HYBRID_WIDE_150M",
    "GPT_OSS_DENSE_ARCHITECTURES",
    "GPT_OSS_DENSE_REFERENCE_150M",
    "GPT_OSS_DENSE_WIDE_150M",
    "QWEN3_ARCHITECTURES",
    "QWEN35_ARCHITECTURES",
    "REFERENCE_150M",
    "WIDE_150M",
    "Qwen3Architecture",
    "Qwen35Architecture",
    "DenseGptOssArchitecture",
    "gpt_oss_dense_model_registry",
    "qwen3_model_registry",
    "qwen35_model_registry",
]
