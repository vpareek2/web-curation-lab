"""Dense decoder blocks using TorchTitan's GPT-OSS attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchtitan.models.common.attention import AttentionMasksType, VarlenAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.gpt_oss.model import Attention, GptOssModel
from torchtitan.models.utils import get_dense_model_nparams_and_flops

from .attention_compat import enable_prebuilt_hopper_fa3

enable_prebuilt_hopper_fa3()


class DenseGptOssTransformerBlock(TransformerBlock):
    """GPT-OSS attention followed by a dense SwiGLU feed-forward network."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        assert isinstance(config.attention, Attention.Config)
        assert config.feed_forward is not None

        self.attn_mask_key = (
            "sliding_window_mask"
            if config.attention.sliding_window_size is not None
            else "basic_mask"
        )
        self.attention = config.attention.build()
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(attention_masks, dict):
            attention_masks = attention_masks[self.attn_mask_key]

        x = x + self.attention(self.attention_norm(x), attention_masks, positions)
        return x + self.feed_forward(self.ffn_norm(x))


class DenseGptOssModel(GptOssModel):
    """GPT-OSS-style local/global attention with dense feed-forward blocks."""

    @dataclass(kw_only=True, slots=True)
    class Config(GptOssModel.Config):
        def update_from_config(self, *, config, **kwargs) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            if parallelism.context_parallel_degree > 1 and isinstance(
                self.layers[0].attention.inner_attention,
                VarlenAttention.Config,
            ):
                raise NotImplementedError(
                    "Context parallelism does not support Varlen attention."
                )

            import spmd_types as spmd
            from torchtitan.models.common.decoder_sharding import (
                dense_activation_placement,
                dense_sequence_parallel_placement,
                set_dense_ffn_sharding,
            )
            from torchtitan.models.gpt_oss.sharding import set_gpt_oss_sharding_config

            enable_sp = parallelism.enable_sequence_parallel
            set_gpt_oss_sharding_config(
                self,
                enable_sp=enable_sp,
                enable_ep=False,
            )
            attn_x_layout = (
                dense_sequence_parallel_placement()
                if enable_sp
                else dense_activation_placement(tp=spmd.I)
            )
            for layer in self.layers:
                assert layer.feed_forward is not None
                set_dense_ffn_sharding(
                    layer.feed_forward,
                    attn_x_layout=attn_x_layout,
                    enable_sp=enable_sp,
                )

        def get_nparams_and_flops(
            self,
            model: torch.nn.Module,
            seq_len: int,
        ) -> tuple[int, int]:
            attention = self.layers[0].attention
            assert isinstance(attention, Attention.Config)
            return get_dense_model_nparams_and_flops(
                model,
                n_layers=len(self.layers),
                n_heads=attention.n_heads,
                head_dims=2 * attention.head_dim,
                seq_len=seq_len,
                enable_weight_tying=self.enable_weight_tying,
            )
