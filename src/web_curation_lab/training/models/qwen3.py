"""Project-owned Qwen3 architecture definitions for TorchTitan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from torchtitan.models.qwen3.model import Qwen3Model, Qwen3TransformerBlock
    from torchtitan.protocols.model_spec import ModelSpec


@dataclass(frozen=True, slots=True)
class Qwen3Architecture:
    """Dimensions that define one dense decoder architecture."""

    name: str
    vocab_size: int
    dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    ffn_hidden_dim: int
    max_seq_len: int = 4096

    @property
    def parameter_count(self) -> int:
        """Count parameters for the tied, bias-free dense decoder."""

        embedding = self.vocab_size * self.dim
        attention = 2 * self.dim * (self.num_heads + self.num_kv_heads) * self.head_dim
        qk_norms = 2 * self.head_dim
        feed_forward = 3 * self.dim * self.ffn_hidden_dim
        block_norms = 2 * self.dim
        final_norm = self.dim
        block = attention + qk_norms + feed_forward + block_norms
        return embedding + self.num_layers * block + final_norm

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim % 16 != 0:
            raise ValueError("head_dim must be divisible by 16")


REFERENCE_150M = Qwen3Architecture(
    name="qwen3_150m_reference",
    vocab_size=32_000,
    dim=768,
    num_layers=16,
    num_heads=6,
    num_kv_heads=2,
    head_dim=128,
    ffn_hidden_dim=2_688,
)

WIDE_150M = Qwen3Architecture(
    name="qwen3_150m_wide",
    vocab_size=32_000,
    dim=1_024,
    num_layers=10,
    num_heads=8,
    num_kv_heads=2,
    head_dim=128,
    ffn_hidden_dim=3_072,
)

ARCHITECTURES = {
    REFERENCE_150M.name: REFERENCE_150M,
    WIDE_150M.name: WIDE_150M,
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


def _build_layers(
    architecture: Qwen3Architecture,
    *,
    attention_backend: str,
) -> list[Qwen3TransformerBlock.Config]:
    from torchtitan.models.common import CosSinRoPE, RMSNorm
    from torchtitan.models.common.config_utils import (
        get_attention_config,
        make_ffn_config,
        make_gqa_config,
    )
    from torchtitan.models.qwen3.model import Qwen3TransformerBlock

    rope = CosSinRoPE.Config(
        dim=architecture.head_dim,
        max_seq_len=architecture.max_seq_len,
        theta=1_000_000.0,
    )
    inner_attention = get_attention_config(attention_backend)

    return [
        Qwen3TransformerBlock.Config(
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
            attention=make_gqa_config(
                dim=architecture.dim,
                n_heads=architecture.num_heads,
                n_kv_heads=architecture.num_kv_heads,
                head_dim=architecture.head_dim,
                wqkv_param_init=_LINEAR_INIT,
                wo_param_init=_depth_init(layer_id),
                inner_attention=inner_attention,
                fuse_qkv=True,
                rope=rope,
                qk_norm=RMSNorm.Config(
                    normalized_shape=architecture.head_dim,
                    eps=1e-6,
                    param_init=_NORM_INIT,
                ),
            ),
            feed_forward=make_ffn_config(
                dim=architecture.dim,
                hidden_dim=architecture.ffn_hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            ),
        )
        for layer_id in range(architecture.num_layers)
    ]


def _build_model_config(
    architecture: Qwen3Architecture,
    *,
    attention_backend: str,
) -> Qwen3Model.Config:
    from torchtitan.models.common import Embedding, Linear, RMSNorm
    from torchtitan.models.qwen3.model import Qwen3Model

    return Qwen3Model.Config(
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
        layers=_build_layers(
            architecture,
            attention_backend=attention_backend,
        ),
    )


def model_registry(
    flavor: str,
    *,
    attention_backend: str = "flex",
) -> ModelSpec:
    """Build a TorchTitan model specification for a project architecture."""

    from torchtitan.components.optimizer import register_moe_load_balancing_hook
    from torchtitan.distributed.pipeline_parallel import pipeline_llm
    from torchtitan.models.qwen3.parallelize import parallelize_qwen3
    from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
    from torchtitan.protocols.model_spec import ModelSpec

    try:
        architecture = ARCHITECTURES[flavor]
    except KeyError as error:
        supported = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"Unknown model flavor {flavor!r}; supported: {supported}") from error

    return ModelSpec(
        name="web_curation_qwen3",
        flavor=architecture.name,
        model=_build_model_config(
            architecture,
            attention_backend=attention_backend,
        ),
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
