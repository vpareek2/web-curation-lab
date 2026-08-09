"""Small dense GPT-OSS-style architectures for TorchTitan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec


@dataclass(frozen=True, slots=True)
class DenseGptOssArchitecture:
    """Dimensions for alternating GPT-OSS SWA/global dense decoders."""

    name: str
    vocab_size: int
    dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    ffn_hidden_dim: int
    sliding_window_size: int = 128
    max_seq_len: int = 4_096

    @property
    def num_sliding_window_layers(self) -> int:
        return (self.num_layers + 1) // 2

    @property
    def num_global_attention_layers(self) -> int:
        return self.num_layers // 2

    @property
    def parameter_count(self) -> int:
        """Count the tied dense model, including attention biases and sinks."""

        embedding = self.vocab_size * self.dim
        attention_weights = (
            2 * self.dim * (self.num_heads + self.num_kv_heads) * self.head_dim
        )
        attention_biases = (
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim + self.dim
        )
        attention_sinks = self.num_heads
        feed_forward = 3 * self.dim * self.ffn_hidden_dim
        block_norms = 2 * self.dim
        block = (
            attention_weights
            + attention_biases
            + attention_sinks
            + feed_forward
            + block_norms
        )
        return embedding + self.num_layers * block + self.dim

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim % 16 != 0:
            raise ValueError("head_dim must be divisible by 16")
        if self.sliding_window_size < 1:
            raise ValueError("sliding_window_size must be positive")


GPT_OSS_DENSE_REFERENCE_150M = DenseGptOssArchitecture(
    name="gpt_oss_dense_150m_reference",
    vocab_size=32_000,
    dim=768,
    num_layers=16,
    num_heads=6,
    num_kv_heads=2,
    head_dim=128,
    ffn_hidden_dim=2_688,
)

GPT_OSS_DENSE_WIDE_150M = DenseGptOssArchitecture(
    name="gpt_oss_dense_150m_wide",
    vocab_size=32_000,
    dim=1_024,
    num_layers=10,
    num_heads=8,
    num_kv_heads=2,
    head_dim=128,
    ffn_hidden_dim=3_072,
)

ARCHITECTURES = {
    GPT_OSS_DENSE_REFERENCE_150M.name: GPT_OSS_DENSE_REFERENCE_150M,
    GPT_OSS_DENSE_WIDE_150M.name: GPT_OSS_DENSE_WIDE_150M,
}

_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT: dict[str, Callable] = {"weight": nn.init.ones_}


def _skip_param_init(parameter: nn.Parameter) -> None:
    del parameter


_EMBEDDING_SKIP_INIT: dict[str, Callable] = {"weight": _skip_param_init}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    std = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    std = 0.02 / (2 * (layer_id + 1)) ** 0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=std),
        "bias": nn.init.zeros_,
    }


def _build_model_config(architecture: DenseGptOssArchitecture):
    from torchtitan.models.common import CosSinRoPE, Embedding, Linear, RMSNorm
    from torchtitan.models.common.config_utils import make_ffn_config
    from torchtitan.models.gpt_oss import _make_gptoss_attn_config

    from .gpt_oss_dense_model import DenseGptOssModel, DenseGptOssTransformerBlock

    rope = CosSinRoPE.Config(
        dim=architecture.head_dim,
        max_seq_len=architecture.max_seq_len,
        theta=1_000_000.0,
    )
    layers = []
    for layer_id in range(architecture.num_layers):
        layers.append(
            DenseGptOssTransformerBlock.Config(
                attention=_make_gptoss_attn_config(
                    dim=architecture.dim,
                    layer_id=layer_id,
                    attn_backend="varlen",
                    n_heads=architecture.num_heads,
                    n_kv_heads=architecture.num_kv_heads,
                    head_dim=architecture.head_dim,
                    sliding_window_size=(
                        architecture.sliding_window_size
                        if layer_id % 2 == 0
                        else None
                    ),
                    fuse_qkv=True,
                    rope=rope,
                ),
                attention_norm=RMSNorm.Config(
                    normalized_shape=architecture.dim,
                    eps=1e-6,
                    param_init=_NORM_INIT,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=architecture.dim,
                    eps=1e-6,
                    param_init=_NORM_INIT,
                ),
                feed_forward=make_ffn_config(
                    dim=architecture.dim,
                    hidden_dim=architecture.ffn_hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )

    return DenseGptOssModel.Config(
        dim=architecture.dim,
        vocab_size=architecture.vocab_size,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=architecture.vocab_size,
            embedding_dim=architecture.dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(
            normalized_shape=architecture.dim,
            eps=1e-6,
            param_init=_NORM_INIT,
        ),
        lm_head=Linear.Config(
            in_features=architecture.dim,
            out_features=architecture.vocab_size,
            param_init=_output_linear_init(architecture.dim),
        ),
        layers=layers,
    )


def model_registry(flavor: str) -> ModelSpec:
    """Build a TorchTitan model spec for a dense GPT-OSS-style architecture."""

    from torchtitan.distributed.pipeline_parallel import pipeline_llm
    from torchtitan.models.gpt_oss.parallelize import parallelize_gptoss
    from torchtitan.protocols.model_spec import ModelSpec

    try:
        architecture = ARCHITECTURES[flavor]
    except KeyError as error:
        supported = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"Unknown model flavor {flavor!r}; supported: {supported}") from error

    return ModelSpec(
        name="web_curation_gpt_oss_dense",
        flavor=architecture.name,
        model=_build_model_config(architecture),
        parallelize_fn=parallelize_gptoss,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
