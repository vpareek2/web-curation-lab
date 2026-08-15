import math

import pytest

from web_curation_eval.metrics import Totals, merge_totals, summarize


def test_metric_formulas() -> None:
    totals = Totals(nll_nats=math.log(4), predicted_tokens=2, utf8_bytes=4, documents=1)
    result = totals.derived()

    assert result["loss_nats_per_token"] == pytest.approx(math.log(2))
    assert result["perplexity"] == pytest.approx(2.0)
    assert result["bits_per_token"] == pytest.approx(1.0)
    assert result["bits_per_byte"] == pytest.approx(0.5)


def test_distributed_merge_matches_single_rank() -> None:
    single = {"source/domain": Totals(9.0, 6, 12, 3)}
    distributed = merge_totals(
        [
            {"source/domain": Totals(4.0, 2, 5, 1)},
            {"source/domain": Totals(5.0, 4, 7, 2)},
        ]
    )

    assert distributed == single
    assert summarize(distributed) == summarize(single)


def test_empty_or_zero_denominators_fail() -> None:
    with pytest.raises(ValueError, match="no domains"):
        summarize({})
    with pytest.raises(ValueError, match="predicted tokens"):
        Totals().derived()
