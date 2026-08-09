import pytest

from web_curation_lab.training.models import (
    GPT_OSS_DENSE_REFERENCE_150M,
    GPT_OSS_DENSE_WIDE_150M,
    HYBRID_REFERENCE_150M,
    HYBRID_WIDE_150M,
    REFERENCE_150M,
    WIDE_150M,
)

EXPECTED_PARAMETER_COUNT = 148_861_696
EXPECTED_WIDE_PARAMETER_COUNT = 153_378_304
EXPECTED_PROCESSED_TOKENS = 500_170_752
EXPECTED_HYBRID_PARAMETER_COUNT = 150_089_248
EXPECTED_HYBRID_WIDE_PARAMETER_COUNT = 153_443_072
EXPECTED_GPT_OSS_DENSE_PARAMETER_COUNT = 148_890_464
EXPECTED_GPT_OSS_DENSE_WIDE_PARAMETER_COUNT = 153_401_424
TRITON_REQUIRED = "TorchTitan requires Triton, which is unavailable on macOS"


def test_reference_architecture_dimensions() -> None:
    assert REFERENCE_150M.num_layers == 16
    assert REFERENCE_150M.dim == 768
    assert REFERENCE_150M.num_heads == 6
    assert REFERENCE_150M.num_kv_heads == 2
    assert REFERENCE_150M.head_dim == 128
    assert REFERENCE_150M.ffn_hidden_dim == 2_688
    assert REFERENCE_150M.vocab_size == 32_000


def test_reference_model_parameter_count() -> None:
    assert REFERENCE_150M.parameter_count == EXPECTED_PARAMETER_COUNT


def test_wide_architecture_dimensions_and_parameter_count() -> None:
    assert WIDE_150M.num_layers == 10
    assert WIDE_150M.dim == 1_024
    assert WIDE_150M.num_heads == 8
    assert WIDE_150M.num_kv_heads == 2
    assert WIDE_150M.head_dim == 128
    assert WIDE_150M.ffn_hidden_dim == 3_072
    assert WIDE_150M.vocab_size == 32_000
    assert WIDE_150M.parameter_count == EXPECTED_WIDE_PARAMETER_COUNT


def test_hybrid_reference_architecture() -> None:
    assert HYBRID_REFERENCE_150M.num_layers == 16
    assert HYBRID_REFERENCE_150M.dim == 768
    assert HYBRID_REFERENCE_150M.num_linear_attention_layers == 12
    assert HYBRID_REFERENCE_150M.num_full_attention_layers == 4
    assert HYBRID_REFERENCE_150M.rotary_dim == 32
    assert HYBRID_REFERENCE_150M.ffn_hidden_dim == 1_024
    assert HYBRID_REFERENCE_150M.parameter_count == EXPECTED_HYBRID_PARAMETER_COUNT


def test_hybrid_wide_architecture() -> None:
    assert HYBRID_WIDE_150M.num_layers == 10
    assert HYBRID_WIDE_150M.dim == 1_024
    assert HYBRID_WIDE_150M.num_linear_attention_layers == 8
    assert HYBRID_WIDE_150M.num_full_attention_layers == 2
    assert HYBRID_WIDE_150M.rotary_dim == 32
    assert HYBRID_WIDE_150M.ffn_hidden_dim == 704
    assert HYBRID_WIDE_150M.parameter_count == EXPECTED_HYBRID_WIDE_PARAMETER_COUNT


def test_hybrid_training_only_compiles_the_loss() -> None:
    pytest.importorskip("triton", reason=TRITON_REQUIRED)
    pytest.importorskip("fla", reason="Qwen3.5 requires flash-linear-attention")

    from web_curation_lab.training.config_registry import curation_qwen3_5_150m_reference

    config = curation_qwen3_5_150m_reference()

    assert config.compile.enable
    assert config.compile.components == ["loss"]


def test_dense_gpt_oss_training_only_compiles_the_loss() -> None:
    pytest.importorskip("triton", reason=TRITON_REQUIRED)

    from web_curation_lab.training.config_registry import (
        curation_gpt_oss_dense_150m_reference,
    )

    config = curation_gpt_oss_dense_150m_reference()

    assert config.compile.enable
    assert config.compile.components == ["loss"]


def test_dense_gpt_oss_architectures() -> None:
    assert GPT_OSS_DENSE_REFERENCE_150M.sliding_window_size == 128
    assert GPT_OSS_DENSE_REFERENCE_150M.num_sliding_window_layers == 8
    assert GPT_OSS_DENSE_REFERENCE_150M.num_global_attention_layers == 8
    assert (
        GPT_OSS_DENSE_REFERENCE_150M.parameter_count
        == EXPECTED_GPT_OSS_DENSE_PARAMETER_COUNT
    )

    assert GPT_OSS_DENSE_WIDE_150M.sliding_window_size == 128
    assert GPT_OSS_DENSE_WIDE_150M.num_sliding_window_layers == 5
    assert GPT_OSS_DENSE_WIDE_150M.num_global_attention_layers == 5
    assert (
        GPT_OSS_DENSE_WIDE_150M.parameter_count
        == EXPECTED_GPT_OSS_DENSE_WIDE_PARAMETER_COUNT
    )


def test_reference_training_token_budget() -> None:
    pytest.importorskip("triton", reason=TRITON_REQUIRED)

    from web_curation_lab.training.config_registry import (
        GLOBAL_BATCH_SIZE,
        NUM_GPUS,
        SEQUENCE_LENGTH,
        TARGET_TOKENS,
        TOKENS_PER_STEP,
        TRAINING_STEPS,
        curation_qwen3_150m_reference,
    )

    config = curation_qwen3_150m_reference()

    assert NUM_GPUS == 8
    assert GLOBAL_BATCH_SIZE == 128
    assert TOKENS_PER_STEP == 262_144
    assert TRAINING_STEPS == 1_908
    assert TRAINING_STEPS * TOKENS_PER_STEP == EXPECTED_PROCESSED_TOKENS
    assert EXPECTED_PROCESSED_TOKENS >= TARGET_TOKENS
    assert config.training.seq_len == SEQUENCE_LENGTH
    assert config.training.steps == TRAINING_STEPS


def test_reference_training_uses_replicated_ddp() -> None:
    pytest.importorskip("triton", reason=TRITON_REQUIRED)

    from web_curation_lab.training.config_registry import (
        NUM_GPUS,
        curation_qwen3_150m_reference,
    )

    config = curation_qwen3_150m_reference()

    assert config.parallelism.data_parallel_replicate_degree == NUM_GPUS
    assert config.parallelism.data_parallel_shard_degree == 1
    assert config.parallelism.tensor_parallel_degree == 1
    assert config.parallelism.pipeline_parallel_degree == 1
    assert config.parallelism.context_parallel_degree == 1
    assert config.activation_checkpoint is None
    assert config.compile.enable
    assert config.compile.components == ["model", "loss"]


def test_reference_torchtitan_model_build() -> None:
    pytest.importorskip("triton", reason=TRITON_REQUIRED)

    import torch

    from web_curation_lab.training.models import qwen3_model_registry

    model_spec = qwen3_model_registry("qwen3_150m_reference")
    with torch.device("meta"):
        model = model_spec.model.build()

    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETER_COUNT
    assert model.tok_embeddings.weight is model.lm_head.weight


def test_wide_torchtitan_model_build() -> None:
    pytest.importorskip("triton", reason=TRITON_REQUIRED)

    import torch

    from web_curation_lab.training.models import qwen3_model_registry

    model_spec = qwen3_model_registry("qwen3_150m_wide")
    with torch.device("meta"):
        model = model_spec.model.build()

    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == EXPECTED_WIDE_PARAMETER_COUNT
    )
    assert model.tok_embeddings.weight is model.lm_head.weight


@pytest.mark.parametrize(
    ("flavor", "expected_count", "full_layers"),
    [
        ("qwen3_5_150m_reference", EXPECTED_HYBRID_PARAMETER_COUNT, 4),
        ("qwen3_5_150m_wide", EXPECTED_HYBRID_WIDE_PARAMETER_COUNT, 2),
    ],
)
def test_hybrid_torchtitan_model_build(
    flavor: str,
    expected_count: int,
    full_layers: int,
) -> None:
    pytest.importorskip("triton", reason=TRITON_REQUIRED)
    pytest.importorskip("fla", reason="Qwen3.5 requires flash-linear-attention")

    import torch
    from torchtitan.models.common.attention import VarlenAttention

    from web_curation_lab.training.models import qwen35_model_registry

    model_spec = qwen35_model_registry(flavor)
    with torch.device("meta"):
        model = model_spec.model.build()

    assert sum(parameter.numel() for parameter in model.parameters()) == expected_count
    assert model.tok_embeddings.weight is model.lm_head.weight
    assert model.vision_encoder is None
    assert sum(layer.full_attn for layer in model.layers.values()) == full_layers
    assert isinstance(model.config.first_attention.inner_attention, VarlenAttention.Config)


@pytest.mark.parametrize(
    ("flavor", "expected_count", "expected_swa", "expected_global"),
    [
        (
            "gpt_oss_dense_150m_reference",
            EXPECTED_GPT_OSS_DENSE_PARAMETER_COUNT,
            8,
            8,
        ),
        (
            "gpt_oss_dense_150m_wide",
            EXPECTED_GPT_OSS_DENSE_WIDE_PARAMETER_COUNT,
            5,
            5,
        ),
    ],
)
def test_dense_gpt_oss_torchtitan_model(
    flavor: str,
    expected_count: int,
    expected_swa: int,
    expected_global: int,
) -> None:
    pytest.importorskip("triton", reason=TRITON_REQUIRED)

    import torch
    from torchtitan.models.common.attention import VarlenAttention

    from web_curation_lab.training.models import gpt_oss_dense_model_registry

    model_spec = gpt_oss_dense_model_registry(flavor)
    windows = [
        layer.attention.inner_attention.window_size for layer in model_spec.model.layers
    ]
    with torch.device("meta"):
        model = model_spec.model.build()

    assert windows.count((127, 0)) == expected_swa
    assert windows.count((-1, 0)) == expected_global
    assert all(
        isinstance(layer.attention.inner_attention, VarlenAttention.Config)
        for layer in model_spec.model.layers
    )
    assert all(
        layer.attention.qkv_linear.wqkv.bias for layer in model_spec.model.layers
    )
    assert all(layer.attention.wo.bias for layer in model_spec.model.layers)
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_count
    assert model.tok_embeddings.weight is model.lm_head.weight
    assert all(hasattr(layer.attention, "sinks") for layer in model.layers.values())
    assert all(hasattr(layer, "feed_forward") for layer in model.layers.values())
