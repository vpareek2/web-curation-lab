"""Text-only Qwen3.5 hybrid architectures for TorchTitan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from torchtitan.models.qwen3_5.model import Qwen35Model
    from torchtitan.protocols.model_spec import ModelSpec


@dataclass(frozen=True, slots=True)
class Qwen35Architecture:
    """Dimensions for a dense text decoder with Qwen3.5 hybrid attention."""

    name: str
    vocab_size: int
    dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    ffn_hidden_dim: int
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    full_attention_interval: int = 4
    conv_kernel_size: int = 4
    max_seq_len: int = 4_096

    @property
    def num_full_attention_layers(self) -> int:
        return self.num_layers // self.full_attention_interval

    @property
    def num_linear_attention_layers(self) -> int:
        return self.num_layers - self.num_full_attention_layers

    @property
    def full_attention_parameter_count(self) -> int:
        query_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        projections = self.dim * (3 * query_dim + 2 * kv_dim)
        return projections + 2 * self.head_dim

    @property
    def linear_attention_parameter_count(self) -> int:
        key_dim = self.num_key_heads * self.key_head_dim
        value_dim = self.num_value_heads * self.value_head_dim
        projections = self.dim * (
            2 * key_dim + 3 * value_dim + 2 * self.num_value_heads
        )
        convolutions = self.conv_kernel_size * (2 * key_dim + value_dim)
        recurrent = self.value_head_dim + 2 * self.num_value_heads
        return projections + convolutions + recurrent

    @property
    def parameter_count(self) -> int:
        """Count parameters for the tied, bias-free text decoder."""

        embedding = self.vocab_size * self.dim
        feed_forward = 3 * self.dim * self.ffn_hidden_dim
        block_norms = 2 * self.dim
        full_blocks = self.num_full_attention_layers * (
            self.full_attention_parameter_count + feed_forward + block_norms
        )
        linear_blocks = self.num_linear_attention_layers * (
            self.linear_attention_parameter_count + feed_forward + block_norms
        )
        return embedding + full_blocks + linear_blocks + self.dim

    def __post_init__(self) -> None:
        if self.num_layers < self.full_attention_interval:
            raise ValueError("num_layers must include at least one full-attention block")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.num_value_heads % self.num_key_heads != 0:
            raise ValueError("num_value_heads must be divisible by num_key_heads")
        if self.head_dim % 16 != 0 or self.rotary_dim % 16 != 0:
            raise ValueError("head_dim and rotary_dim must be divisible by 16")


HYBRID_REFERENCE_150M = Qwen35Architecture(
    name="qwen3_5_150m_reference",
    vocab_size=32_000,
    dim=768,
    num_layers=16,
    num_heads=12,
    num_kv_heads=3,
    head_dim=128,
    rotary_dim=32,
    ffn_hidden_dim=1_024,
    num_key_heads=12,
    num_value_heads=12,
    key_head_dim=128,
    value_head_dim=128,
)

HYBRID_WIDE_150M = Qwen35Architecture(
    name="qwen3_5_150m_wide",
    vocab_size=32_000,
    dim=1_024,
    num_layers=10,
    num_heads=16,
    num_kv_heads=4,
    head_dim=128,
    rotary_dim=32,
    ffn_hidden_dim=704,
    num_key_heads=16,
    num_value_heads=16,
    key_head_dim=128,
    value_head_dim=128,
)

ARCHITECTURES = {
    HYBRID_REFERENCE_150M.name: HYBRID_REFERENCE_150M,
    HYBRID_WIDE_150M.name: HYBRID_WIDE_150M,
}

def _skip_param_init(parameter: nn.Parameter) -> None:
    del parameter


_EMBEDDING_SKIP_INIT: dict[str, Callable] = {"weight": _skip_param_init}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    std = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
        "bias": nn.init.zeros_,
    }


def _build_model_config(
    architecture: Qwen35Architecture,
    *,
    attention_backend: str,
) -> Qwen35Model.Config:
    from torchtitan.models.common import Embedding, Linear
    from torchtitan.models.qwen3_5 import _build_qwen35_layers, _offset_norm
    from torchtitan.models.qwen3_5.rope import MRoPE

    from .qwen3_5_text_model import TextQwen35Model

    if attention_backend != "varlen":
        raise ValueError(
            "Qwen3.5 benchmarks require the varlen backend so packed-document "
            "DeltaNet state resets at document boundaries."
        )

    return TextQwen35Model.Config(
        dim=architecture.dim,
        vocab_size=architecture.vocab_size,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=architecture.vocab_size,
            embedding_dim=architecture.dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=_offset_norm(architecture.dim),
        lm_head=Linear.Config(
            in_features=architecture.dim,
            out_features=architecture.vocab_size,
            param_init=_output_linear_init(architecture.dim),
        ),
        layers=_build_qwen35_layers(
            n_layers=architecture.num_layers,
            dim=architecture.dim,
            n_heads=architecture.num_heads,
            n_kv_heads=architecture.num_kv_heads,
            head_dim=architecture.head_dim,
            rotary_dim=architecture.rotary_dim,
            rope=MRoPE.Config(
                dim=architecture.rotary_dim,
                max_seq_len=architecture.max_seq_len,
                theta=10_000_000.0,
                mrope_section=[6, 5, 5],
            ),
            hidden_dim=architecture.ffn_hidden_dim,
            n_key_heads=architecture.num_key_heads,
            n_value_heads=architecture.num_value_heads,
            key_head_dim=architecture.key_head_dim,
            value_head_dim=architecture.value_head_dim,
            full_attention_interval=architecture.full_attention_interval,
            attn_backend=attention_backend,
            fla_backend="fla_chunked",
        ),
    )


def model_registry(
    flavor: str,
    *,
    attention_backend: str = "varlen",
) -> ModelSpec:
    """Build a TorchTitan model specification for a text-only Qwen3.5 model."""

    from torchtitan.components.optimizer import register_moe_load_balancing_hook
    from torchtitan.distributed.pipeline_parallel import pipeline_llm
    from torchtitan.models.qwen3_5.parallelize import parallelize_qwen3_5
    from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
    from torchtitan.protocols.model_spec import ModelSpec

    try:
        architecture = ARCHITECTURES[flavor]
    except KeyError as error:
        supported = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"Unknown model flavor {flavor!r}; supported: {supported}") from error

    return ModelSpec(
        name="web_curation_qwen3_5",
        flavor=architecture.name,
        model=_build_model_config(
            architecture,
            attention_backend=attention_backend,
        ),
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Qwen35StateDictAdapter,
    )
