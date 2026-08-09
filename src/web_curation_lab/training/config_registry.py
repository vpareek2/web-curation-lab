"""TorchTitan training configurations for Web Curation Lab."""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from .models.gpt_oss_dense import model_registry as gpt_oss_dense_model_registry
from .models.qwen3 import model_registry as qwen3_model_registry
from .models.qwen3_5 import model_registry as qwen35_model_registry

NUM_GPUS = 8
SEQUENCE_LENGTH = 2_048
LOCAL_BATCH_SIZE = 16
GLOBAL_BATCH_SIZE = NUM_GPUS * LOCAL_BATCH_SIZE
TARGET_TOKENS = 500_000_000
TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * SEQUENCE_LENGTH
TRAINING_STEPS = (TARGET_TOKENS + TOKENS_PER_STEP - 1) // TOKENS_PER_STEP
HF_ASSETS_PATH = "./assets/hf/Mistral-7B-v0.1"


def _benchmark_config(model_flavor: str, *, family: str = "qwen3") -> Trainer.Config:
    """Build the shared end-to-end 500M-token benchmark configuration."""

    registries = {
        "gpt_oss_dense": gpt_oss_dense_model_registry,
        "qwen3": qwen3_model_registry,
        "qwen3_5": qwen35_model_registry,
    }
    registry = registries[family]
    model_spec = registry(model_flavor)
    # flash-linear-attention's varlen causal-convolution autograd kernel is
    # explicitly excluded from torch.compile. Keep the Qwen3.5 model eager so
    # it runs correctly, while retaining compiled cross-entropy. The dense
    # Qwen3 and GPT-OSS recipes compile each Transformer block as normal.
    compile_components = ["loss"] if family == "qwen3_5" else ["model", "loss"]
    return Trainer.Config(
        model_spec=model_spec,
        hf_assets_path=HF_ASSETS_PATH,
        dump_folder=f"./outputs/benchmarks/{model_flavor}",
        loss=ChunkedLossWrapper.Config(
            num_chunks=8,
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        optimizer=default_adamw(lr=6e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=round(TRAINING_STEPS * 0.10),
            decay_ratio=0.15,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=LOCAL_BATCH_SIZE,
            global_batch_size=GLOBAL_BATCH_SIZE,
            seq_len=SEQUENCE_LENGTH,
            steps=TRAINING_STEPS,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
            infinite=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=NUM_GPUS,
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        activation_checkpoint=None,
        compile=CompileConfig(
            enable=True,
            components=compile_components,
        ),
        validator=Validator.Config(enable=False),
        debug=DebugConfig(seed=42),
    )


def curation_qwen3_150m_reference() -> Trainer.Config:
    """Sixteen-layer, 148.86M-parameter Qwen3 reference architecture."""

    return _benchmark_config("qwen3_150m_reference")


def curation_qwen3_150m_wide() -> Trainer.Config:
    """Ten-layer, 153.38M-parameter Qwen3 architecture with larger GEMMs."""

    return _benchmark_config("qwen3_150m_wide")


def curation_gpt_oss_dense_150m_reference() -> Trainer.Config:
    """Sixteen-layer dense GPT-OSS attention architecture."""

    return _benchmark_config(
        "gpt_oss_dense_150m_reference",
        family="gpt_oss_dense",
    )


def curation_gpt_oss_dense_150m_wide() -> Trainer.Config:
    """Ten-layer wide dense GPT-OSS attention architecture."""

    return _benchmark_config(
        "gpt_oss_dense_150m_wide",
        family="gpt_oss_dense",
    )


def curation_qwen3_5_150m_reference() -> Trainer.Config:
    """Sixteen-layer, 150.09M-parameter Qwen3.5 hybrid architecture."""

    return _benchmark_config("qwen3_5_150m_reference", family="qwen3_5")


def curation_qwen3_5_150m_wide() -> Trainer.Config:
    """Ten-layer, 153.44M-parameter Qwen3.5 hybrid architecture."""

    return _benchmark_config("qwen3_5_150m_wide", family="qwen3_5")
