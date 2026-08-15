"""GPU preflight comparing a packaged Qwen3 checkpoint across implementations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from web_curation_lab.training.models.qwen3 import model_registry


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    """Return the exact acceptance measurements for two logit tensors."""

    if reference.shape != candidate.shape:
        raise ValueError(f"Logit shapes differ: {reference.shape} != {candidate.shape}")
    ref_log_probs = functional.log_softmax(reference.float(), dim=-1)
    candidate_log_probs = functional.log_softmax(candidate.float(), dim=-1)
    kl = functional.kl_div(
        candidate_log_probs,
        ref_log_probs,
        reduction="batchmean",
        log_target=True,
    )
    return {
        "argmax_match": bool(
            torch.equal(reference.argmax(dim=-1), candidate.argmax(dim=-1))
        ),
        "kl_divergence": float(kl.item()),
        "max_absolute_logit_error": float(
            (reference.float() - candidate.float()).abs().max().item()
        ),
    }


@torch.inference_mode()
def run_parity(model_path: Path, prompt: str, kl_threshold: float) -> dict:
    """Load the same HF weights through Transformers and TorchTitan and compare."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Install the root eval-preflight extra before running parity: "
            "uv sync --extra eval-preflight"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("HF parity preflight requires a CUDA GPU")
    device = torch.device("cuda")
    manifest = json.loads((model_path / "evaluation_manifest.json").read_text())
    architecture = manifest["architecture"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids.to(device)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
    ).to(device)
    hf_model.eval()
    hf_logits = hf_model(input_ids=input_ids).logits

    model_spec = model_registry(architecture, attention_backend="flex")
    titan_model = model_spec.model.build().to(device=device, dtype=torch.float32)
    adapter = model_spec.state_dict_adapter(model_spec.model, str(model_path))
    titan_state = adapter.from_hf(dict(hf_model.state_dict()))
    missing, unexpected = titan_model.load_state_dict(titan_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"TorchTitan state mismatch; missing={missing}, unexpected={unexpected}"
        )
    titan_model.eval()
    positions = torch.arange(input_ids.shape[1], device=device).expand_as(input_ids)
    attention_masks = titan_model.get_attention_masks(positions)
    titan_logits = titan_model(
        input_ids,
        positions=positions,
        attention_masks=attention_masks,
    )
    measurements = compare_logits(hf_logits, titan_logits)
    if not measurements["argmax_match"]:
        raise RuntimeError(f"TorchTitan/Transformers argmax mismatch: {measurements}")
    if measurements["kl_divergence"] > kl_threshold:
        raise RuntimeError(f"TorchTitan/Transformers KL threshold exceeded: {measurements}")

    # Production-format smoke: the packaged BF16 artifact must load and generate.
    bf16_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to(device)
    generated = bf16_model.generate(input_ids, max_new_tokens=4, do_sample=False)
    new_tokens = generated.shape[1] - input_ids.shape[1]
    # Generation may stop before max_new_tokens when this small preflight model
    # emits EOS.  Loading, preserving the prompt, and producing at least one
    # token are the properties this smoke test needs to establish.
    if not 1 <= new_tokens <= 4:
        raise RuntimeError(
            f"BF16 greedy-generation smoke produced {new_tokens} new tokens"
        )
    if not torch.equal(generated[:, : input_ids.shape[1]], input_ids):
        raise RuntimeError("BF16 greedy-generation smoke did not preserve the prompt")
    return {
        "model": str(model_path.resolve()),
        "prompt_tokens": input_ids.shape[1],
        "generated_tokens": new_tokens,
        "kl_threshold": kl_threshold,
        **measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="Web data quality depends on")
    parser.add_argument("--kl-threshold", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_parity(args.model, args.prompt, args.kl_threshold)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
