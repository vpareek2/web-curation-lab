"""Project-owned mixed Muon/AdamW optimizer integration for TorchTitan."""

from dataclasses import dataclass

import torch
from torch import nn
from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.components.optimizer import OptimizersContainer


def use_muon(name: str, parameter: nn.Parameter) -> bool:
    """Return whether a parameter is a hidden 2D matrix suitable for Muon.

    Muon is intended for hidden matrix parameters. Embeddings and output heads
    remain on AdamW even though their weights are also two-dimensional.
    """

    canonical_name = canonical_fqn(name)
    excluded_modules = ("tok_embeddings.", "lm_head.", "output.")
    return parameter.ndim == 2 and not canonical_name.startswith(excluded_modules)


class MuonWithAdamW(OptimizersContainer):
    """TorchTitan optimizer container using Muon with an AdamW fallback."""

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        muon_lr: float = 0.02
        muon_momentum: float = 0.95
        muon_weight_decay: float = 0.1
        muon_ns_steps: int = 5
        muon_adjust_lr_fn: str | None = "match_rms_adamw"
        adamw_lr: float = 6e-4
        adamw_betas: tuple[float, float] = (0.9, 0.95)
        adamw_eps: float = 1e-8
        adamw_weight_decay: float = 0.1

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        all_params: list[nn.Parameter] = []
        self.optimizers = []
        self.model_parts = model_parts

        for part_idx, model in enumerate(model_parts):
            muon_params: list[nn.Parameter] = []
            muon_names: list[str] = []
            adamw_params: list[nn.Parameter] = []
            adamw_names: list[str] = []

            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if use_muon(name, parameter):
                    muon_params.append(parameter)
                    muon_names.append(canonical_fqn(name))
                else:
                    adamw_params.append(parameter)
                    adamw_names.append(canonical_fqn(name))

            if muon_params:
                muon = torch.optim.Muon(
                    [
                        {
                            "params": muon_params,
                            "param_names": muon_names,
                            "lr": config.muon_lr,
                            "momentum": config.muon_momentum,
                            "weight_decay": config.muon_weight_decay,
                            "nesterov": True,
                            "ns_steps": config.muon_ns_steps,
                            "adjust_lr_fn": config.muon_adjust_lr_fn,
                        }
                    ]
                )
                self.optimizers.append(muon)
                self._log_optimizer(muon, part_idx, ["hidden 2D matrices"])
                all_params.extend(muon_params)

            if adamw_params:
                adamw = torch.optim.AdamW(
                    [
                        {
                            "params": adamw_params,
                            "param_names": adamw_names,
                            "lr": config.adamw_lr,
                            "betas": config.adamw_betas,
                            "eps": config.adamw_eps,
                            "weight_decay": config.adamw_weight_decay,
                        }
                    ],
                    fused=config.implementation == "fused",
                    foreach=config.implementation == "foreach",
                )
                self.optimizers.append(adamw)
                self._log_optimizer(adamw, part_idx, ["AdamW fallback"])
                all_params.extend(adamw_params)

        self._validate_params(all_params)
        self._post_init(all_params)


def muon_with_adamw(
    *,
    muon_lr: float = 0.02,
    adamw_lr: float = 6e-4,
) -> MuonWithAdamW.Config:
    """Build the initial mixed-optimizer experiment configuration."""

    return MuonWithAdamW.Config(muon_lr=muon_lr, adamw_lr=adamw_lr)
