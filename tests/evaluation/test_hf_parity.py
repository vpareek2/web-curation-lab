import pytest
import torch

from web_curation_lab.evaluation.hf_parity import compare_logits


def test_identical_logits_pass_parity_measurements() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    result = compare_logits(logits, logits.clone())

    assert result["argmax_match"] is True
    assert result["kl_divergence"] == pytest.approx(0.0)
    assert result["max_absolute_logit_error"] == pytest.approx(0.0)


def test_shape_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        compare_logits(torch.zeros(1, 2), torch.zeros(2, 2))
