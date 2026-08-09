"""Hopper FlashAttention-3 integration for TorchTitan's varlen wrapper."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch


_FLASH_ATTENTION_3_INTERFACE = None


@cache
def _get_flash_attention_3_interface():
    """Load and cache the versioned Hopper FA3 binary before compilation."""

    from kernels import get_kernel

    return get_kernel(
        "kernels-community/flash-attn3", version=1
    ).flash_attn_interface


def _flash_attention_3_varlen_forward(
    self,
    q_BLNH: torch.Tensor,
    k_BLNH: torch.Tensor,
    v_BLNH: torch.Tensor,
    *,
    attention_masks,
    scale: float | None = None,
    out_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    """Run the prebuilt Hopper FA3 varlen kernel behind TorchTitan's API."""

    del kwargs
    from torchtitan.models.common.attention import VarlenMetadata

    assert isinstance(attention_masks, VarlenMetadata)
    B, L, _, H = q_BLNH.shape
    q_TNH = q_BLNH.reshape(B * L, -1, H).to(torch.bfloat16)
    k_TNH = k_BLNH.reshape(B * L, -1, H).to(torch.bfloat16)
    v_TNH = v_BLNH.reshape(B * L, -1, H).to(torch.bfloat16)

    # This function is called from compiled Transformer blocks. Do not call the
    # cache-decorated loader here: Dynamo traces through cache wrappers and then
    # attempts to trace the kernel package's host-system inspection. The loader
    # is eagerly resolved by ``enable_prebuilt_hopper_fa3`` instead.
    if _FLASH_ATTENTION_3_INTERFACE is None:
        raise RuntimeError("Prebuilt Hopper FA3 was not initialized")

    result = _FLASH_ATTENTION_3_INTERFACE.flash_attn_varlen_func(
        q_TNH,
        k_TNH,
        v_TNH,
        attention_masks.cu_seq_q,
        attention_masks.cu_seq_k,
        attention_masks.max_q,
        attention_masks.max_k,
        softmax_scale=scale,
        causal=self.window_size[1] == 0,
        window_size=self.window_size,
        # TorchTitan requests LSE only for the GPT-OSS attention-sink epilogue.
        return_attn_probs=out_transform is not None,
    )

    if out_transform is None:
        return result.reshape(B, L, -1, H).to(q_BLNH.dtype)

    out_TNH, lse_NT = result
    out_BLNH = out_TNH.reshape(B, L, -1, H).to(q_BLNH.dtype)
    lse_BLN = lse_NT.transpose(0, 1).reshape(B, L, -1)
    return out_transform(out_BLNH, lse_BLN)


def enable_prebuilt_hopper_fa3() -> None:
    """Route TorchTitan varlen attention to a prebuilt CUDA 13 Hopper FA3 kernel.

    Torch 2.13 expects an FA3 package that registers operators under
    ``torch.ops.flash_attn_3``. The maintained `kernels-community/flash-attn3`
    binary exposes the same FA3 kernels through its own namespaced operators,
    so it is called directly here. This keeps H100 recipes on FA3 without a
    local CUDA compilation and retains TorchTitan's packed-document semantics.
    """

    from torchtitan.models.common.attention import VarlenAttention
    from torchtitan.tools import utils as titan_utils

    global _FLASH_ATTENTION_3_INTERFACE

    if getattr(VarlenAttention, "_web_curation_uses_prebuilt_fa3", False):
        return

    _FLASH_ATTENTION_3_INTERFACE = _get_flash_attention_3_interface()

    # Avoid Torch's incompatible auto-activation path; forward() below uses
    # the prebuilt FA3 operator directly.
    titan_utils.get_cuda_flash_attention_impl = lambda: None
    VarlenAttention.forward = _flash_attention_3_varlen_forward
    VarlenAttention._web_curation_uses_prebuilt_fa3 = True
