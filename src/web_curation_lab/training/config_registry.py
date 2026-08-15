"""TorchTitan training configurations for Web Curation Lab."""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS, HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from .models.gpt_oss_dense import model_registry as gpt_oss_dense_model_registry
from .models.qwen3 import model_registry as qwen3_model_registry
from .models.qwen3_5 import model_registry as qwen35_model_registry
from .optimizer import muon_with_adamw

NUM_GPUS = 8
SEQUENCE_LENGTH = 2_048
LOCAL_BATCH_SIZE = 16
GLOBAL_BATCH_SIZE = NUM_GPUS * LOCAL_BATCH_SIZE
TARGET_TOKENS = 500_000_000
TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * SEQUENCE_LENGTH
TRAINING_STEPS = (TARGET_TOKENS + TOKENS_PER_STEP - 1) // TOKENS_PER_STEP
HF_ASSETS_PATH = "./assets/hf/Mistral-7B-v0.1"
SCORED_TARGET_TOKENS = 100_000_000_000
SCORED_TRAINING_STEPS = (SCORED_TARGET_TOKENS + TOKENS_PER_STEP - 1) // TOKENS_PER_STEP
HEALTH_VALIDATION_TOKENS = 2_000_000
HEALTH_VALIDATION_STEPS = (
    HEALTH_VALIDATION_TOKENS + TOKENS_PER_STEP - 1
) // TOKENS_PER_STEP
HEALTH_DATA_PATH = "./data/evaluation/paloma_health.jsonl"
PREFLIGHT_NUM_GPUS = 2
PREFLIGHT_LOCAL_BATCH_SIZE = 4
PREFLIGHT_GLOBAL_BATCH_SIZE = PREFLIGHT_NUM_GPUS * PREFLIGHT_LOCAL_BATCH_SIZE
PREFLIGHT_DUMP_FOLDER = "./outputs/preflight/qwen3_150m_wide"


def _load_health_dataset(path: str):
    from datasets import load_dataset

    return load_dataset("json", data_files=path, split="train")


DATASETS.setdefault(
    "paloma_health",
    DatasetConfig(
        path=HEALTH_DATA_PATH,
        loader=_load_health_dataset,
        sample_processor=lambda sample: sample["text"],
    ),
)


def _benchmark_config(
    model_flavor: str,
    *,
    family: str = "qwen3",
    use_muon: bool = False,
) -> Trainer.Config:
    """Build the shared end-to-end 500M-token benchmark configuration."""

    registries = {
        "gpt_oss_dense": gpt_oss_dense_model_registry,
        "qwen3": qwen3_model_registry,
        "qwen3_5": qwen35_model_registry,
    }
    registry = registries[family]
    model_spec = registry(model_flavor)
    # FLA's causal-convolution and the prebuilt Hopper FA3 wheel both use
    # custom autograd functions that PyTorch 2.13 cannot compile. Keep those
    # model families eager while retaining compiled cross-entropy. The plain
    # Qwen3 recipes compile each Transformer block as normal.
    compile_components = (
        ["loss"] if family in {"qwen3_5", "gpt_oss_dense"} else ["model", "loss"]
    )
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
        optimizer=muon_with_adamw() if use_muon else default_adamw(lr=6e-4),
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

    return _benchmark_config("qwen3_150m_wide", use_muon=True)


def curation_qwen3_150m_wide_scored() -> Trainer.Config:
    """Production 100B-token recipe with health validation and native HF export."""

    config = _benchmark_config("qwen3_150m_wide", use_muon=True)
    config.dump_folder = "./outputs/runs/qwen3_150m_wide_scored"
    config.training.steps = SCORED_TRAINING_STEPS
    config.lr_scheduler.warmup_steps = round(SCORED_TRAINING_STEPS * 0.10)
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        folder="checkpoints",
        interval=max(1, round(SCORED_TRAINING_STEPS * 0.01)),
        keep_latest_k=3,
        last_save_model_only=True,
        last_save_in_hf=True,
        export_dtype="bfloat16",
    )
    config.validator = Validator.Config(
        enable=True,
        freq=max(1, round(SCORED_TRAINING_STEPS * 0.02)),
        steps=HEALTH_VALIDATION_STEPS,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="paloma_health",
            dataset_path=HEALTH_DATA_PATH,
            infinite=True,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        ),
    )
    return config


def _preflight_config(*, steps: int, final_hf_export: bool) -> Trainer.Config:
    """Two-GPU, local-data recipe for checkpoint/resume and HF export checks."""

    config = _benchmark_config("qwen3_150m_wide", use_muon=True)
    config.dump_folder = PREFLIGHT_DUMP_FOLDER
    config.training.local_batch_size = PREFLIGHT_LOCAL_BATCH_SIZE
    config.training.global_batch_size = PREFLIGHT_GLOBAL_BATCH_SIZE
    config.training.steps = steps
    config.lr_scheduler.warmup_steps = 2
    config.parallelism.data_parallel_replicate_degree = PREFLIGHT_NUM_GPUS
    config.dataloader = HuggingFaceTextDataLoader.Config(
        dataset="paloma_health",
        dataset_path=HEALTH_DATA_PATH,
        infinite=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1, enable_tensorboard=True)
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        folder="checkpoints",
        interval=3,
        keep_latest_k=3,
        last_save_model_only=final_hf_export,
        last_save_in_hf=final_hf_export,
        export_dtype="float32",
    )
    config.validator = Validator.Config(
        enable=True,
        freq=6,
        steps=2,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="paloma_health",
            dataset_path=HEALTH_DATA_PATH,
            infinite=True,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        ),
    )
    return config


def curation_qwen3_150m_wide_preflight_checkpoint() -> Trainer.Config:
    """Run six steps and leave a complete resumable DCP checkpoint."""

    return _preflight_config(steps=6, final_hf_export=False)


def curation_qwen3_150m_wide_preflight_export() -> Trainer.Config:
    """Resume the six-step run, finish at step 12, and export FP32 HF weights."""

    return _preflight_config(steps=12, final_hf_export=True)


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
