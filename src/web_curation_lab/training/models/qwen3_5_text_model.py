"""TorchTitan Qwen3.5 decoder specialized for text-only training."""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.models.common.decoder import Decoder
from torchtitan.models.qwen3_5.model import Qwen35Model

from .attention_compat import enable_prebuilt_hopper_fa3

enable_prebuilt_hopper_fa3()


class TextQwen35Model(Qwen35Model):
    """Qwen3.5 hybrid decoder without the unused vision tower."""

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Model.Config):
        vision_encoder: None = None

        def update_from_config(self, *, config, **kwargs) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism
            if (
                parallelism.tensor_parallel_degree > 1
                or parallelism.expert_parallel_degree > 1
            ):
                raise NotImplementedError(
                    "The project text-only Qwen3.5 wrapper currently supports DP only."
                )

    def __init__(self, config: Config):
        Decoder.__init__(self, config)
        self.vision_encoder = None

    def forward(self, tokens, positions=None, attention_masks=None):
        return Decoder.forward(
            self,
            tokens,
            positions=positions,
            attention_masks=attention_masks,
        )
