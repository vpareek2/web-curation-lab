from torch import nn

from web_curation_lab.training.optimizer import MuonWithAdamW, use_muon


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(16, 8)
        self.hidden = nn.Linear(8, 12, bias=False)
        self.norm = nn.LayerNorm(12)
        self.lm_head = nn.Linear(12, 16, bias=False)


def test_muon_parameter_policy() -> None:
    model = TinyModel()
    parameters = dict(model.named_parameters())

    assert use_muon("hidden.weight", parameters["hidden.weight"])
    assert not use_muon("tok_embeddings.weight", parameters["tok_embeddings.weight"])
    assert not use_muon("lm_head.weight", parameters["lm_head.weight"])
    assert not use_muon("norm.weight", parameters["norm.weight"])
    assert not use_muon("norm.bias", parameters["norm.bias"])


def test_mixed_optimizer_assigns_every_parameter_once() -> None:
    model = TinyModel()
    config = MuonWithAdamW.Config()
    optimizers = config.build(model_parts=[model])

    assert [type(optimizer).__name__ for optimizer in optimizers] == ["Muon", "AdamW"]

    muon_names = optimizers.optimizers[0].param_groups[0]["param_names"]
    adamw_names = optimizers.optimizers[1].param_groups[0]["param_names"]
    assert muon_names == ["hidden.weight"]
    assert set(adamw_names) == {
        "tok_embeddings.weight",
        "norm.weight",
        "norm.bias",
        "lm_head.weight",
    }
    assert len(muon_names) + len(adamw_names) == len(list(model.parameters()))
