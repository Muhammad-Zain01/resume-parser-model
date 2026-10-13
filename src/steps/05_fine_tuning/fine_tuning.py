"""Fine-tune a compact instruction model to parse resumes into the project schema.

Step 05 owns the fine-tuning workflow:

- download and verify the curated splits from the Hugging Face dataset repo;
- build chat-format training examples that reuse the evaluation prompt (a
  before/after comparison also requires identical held-out examples/settings);
- load the base model in 4-bit (QLoRA) and attach LoRA adapters;
- train with Weights & Biases tracking and resumable checkpoints;
- evaluate checkpoints with the step-04 evaluator;
- push adapters/checkpoints to the Hugging Face model repo.

Heavy dependencies (torch, transformers, peft, trl, bitsandbytes, wandb) are
imported lazily inside the functions so this module stays importable in a bare
environment.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
import shutil
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=False)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from importlib import import_module

from src.schema.resume_output import ResumeOutput
from src.utils.paths import MODELS_DIR, ensure_data_directories


# ``04_evaluation`` starts with a digit, so it cannot be a normal import target;
# load it through the import machinery the same way the notebooks do.
_evaluation = import_module("src.steps.04_evaluation.evaluator")
EvaluationItem = _evaluation.EvaluationItem
Evaluator = _evaluation.Evaluator
ModelCallbackOutput = _evaluation.ModelCallbackOutput
render_json_prompt = _evaluation.render_json_prompt


DATA_DIR = PROJECT_ROOT / ".data"
CURATED_DIR = DATA_DIR / "curated"

# Hugging Face account used for the dataset source and the model destination.
# Use your own username please.
HF_USERNAME = "md-zain"
HF_DATASET_REPO = f"{HF_USERNAME}/resume-parser-dataset"
HF_MODEL_REPO = f"{HF_USERNAME}/resume-parser-qwen2.5-3b"

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_SPLITS = ("train", "validation")
SPLIT_FILENAMES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _hf_token() -> str | None:
    """Return a usable HF token from the environment, ignoring placeholders."""
    token = (os.getenv("HF_TOKEN") or "").strip()
    if token and not token.startswith(("hf_your_", "hf_placeholder_")):
        return token
    return None


def dataset_repo_id(repo_id: str | None = None) -> str:
    """Return the dataset repo id (code-level default; no env needed)."""
    if repo_id and repo_id.strip():
        return repo_id.strip()
    return HF_DATASET_REPO


def model_repo_id(repo_id: str | None = None) -> str:
    """Return the model repo id for adapters (code-level default; no env needed)."""
    if repo_id and repo_id.strip():
        return repo_id.strip()
    return HF_MODEL_REPO


def checkpoint_revision(
    num_resumes: int, *, steps: int | None = None, prefix: str = "trained", run_tag: str | None = None
) -> str:
    """Build a revision/branch name that records how much training it holds.

    Use it as the ``revision`` when pushing so each training batch lands on its
    own branch (for example ``trained-2000`` or ``trained-2000-steps-250``).
    """
    if num_resumes < 0:
        raise ValueError("num_resumes must be non-negative.")
    revision = f"{prefix}-{num_resumes}"
    if run_tag:
        revision += f"-{run_tag}"
    if steps is not None:
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        revision += f"-steps-{steps}"
    return re.sub(r"[^0-9A-Za-z_-]+", "-", revision).strip("-_") or "trained"


def _ensure_revision(repo_id: str, revision: str | None, repo_type: str, token: str) -> None:
    """Create the target branch/revision on the Hub if it does not exist yet."""
    if not revision or revision == "main":
        return
    from huggingface_hub import HfApi

    HfApi().create_branch(
        repo_id=repo_id, repo_type=repo_type, branch=revision, exist_ok=True, token=token
    )


# ---------------------------------------------------------------------------
# Dataset: download, load, verify
# ---------------------------------------------------------------------------


def download_split(split: str, *, curated_dir: Path | str = CURATED_DIR, force: bool = False) -> Path:
    """Download one curated split from the dataset repo if it is missing."""
    filename = SPLIT_FILENAMES.get(split)
    if filename is None:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_FILENAMES)}.")
    destination = Path(curated_dir) / filename
    if destination.is_file() and not force and not os.getenv("HF_DATASET_REVISION"):
        return destination

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    repo_id = dataset_repo_id()
    revision = (os.getenv("HF_DATASET_REVISION") or "main").strip() or "main"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=f"curated/{filename}",
            force_download=force,
            token=_hf_token(),
        )
    except EntryNotFoundError:
        # Support the earlier layout where split files sit at the repo root.
        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            force_download=force,
            token=_hf_token(),
        )
    temporary = destination.with_suffix(".jsonl.tmp")
    shutil.copyfile(downloaded, temporary)
    temporary.replace(destination)
    return destination


def download_curated_splits(
    *,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    curated_dir: Path | str = CURATED_DIR,
    force: bool = False,
) -> dict[str, Path]:
    """Download the curated splits needed for training and validation."""
    return {name: download_split(name, curated_dir=curated_dir, force=force) for name in splits}


def load_records(path: Path | str) -> list[dict[str, Any]]:
    """Load JSON Lines records from a curated split file."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _record_digest(record: dict[str, Any]) -> str:
    payload = json.dumps(
        [record.get("resume_text", ""), record.get("target_json")],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _text_digest(record: dict[str, Any]) -> str:
    # Whitespace differences do not make the same resume an independent example.
    text = " ".join(record["resume_text"].split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_records(records: list[dict[str, Any]], split: str) -> None:
    if not records:
        raise ValueError(f"{split} has no records.")
    for index, record in enumerate(records):
        text = record.get("resume_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{split} record {index} has no usable resume_text.")
        if not str(record.get("resume_id") or "").strip():
            raise ValueError(f"{split} record {index} has no resume_id.")
        try:
            ResumeOutput.model_validate(record.get("target_json"))
        except Exception:
            # Do not include private field values from a Pydantic error in logs.
            raise ValueError(f"{split} record {index} has an invalid target_json.") from None


def _deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(_text_digest(record), []).append(record)
    kept = []
    duplicates = conflicts = 0
    for group in groups.values():
        targets = {json.dumps(r["target_json"], sort_keys=True, ensure_ascii=False) for r in group}
        if len(targets) > 1:
            conflicts += len(group)
        else:
            kept.append(group[0])
            duplicates += len(group) - 1
    return kept, {"duplicate_rows_removed": duplicates, "conflicting_rows_removed": conflicts}


def prepare_training_records(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    *,
    test_records: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate records and exclude duplicate/conflicting text and held-out leakage.

    Original files are preserved. Test records only reserve identities/text; they
    are never used as training targets. Validation takes precedence over training.
    """
    _validate_records(train_records, "train")
    _validate_records(validation_records, "validation")
    if test_records is not None:
        _validate_records(test_records, "test")
    reserved = list(test_records or [])
    test_text = {_text_digest(r) for r in reserved}
    test_ids = {str(r["resume_id"]) for r in reserved}
    validation = [r for r in validation_records
                  if _text_digest(r) not in test_text and str(r["resume_id"]) not in test_ids]
    validation, validation_report = _deduplicate_records(validation)
    # Reserve even excluded/conflicting validation rows to avoid moving held-out
    # material into training as an accidental consequence of cleaning.
    reserved += validation_records
    heldout_text = {_text_digest(r) for r in reserved}
    heldout_ids = {str(r["resume_id"]) for r in reserved}
    train = [r for r in train_records
             if _text_digest(r) not in heldout_text and str(r["resume_id"]) not in heldout_ids]
    train_overlap = len(train_records) - len(train)
    train, train_report = _deduplicate_records(train)
    _validate_records(train, "prepared train")
    _validate_records(validation, "prepared validation")
    report = {
        "train": {"input_rows": len(train_records), "heldout_overlap_removed": train_overlap,
                  **train_report, "kept_rows": len(train)},
        "validation": {"input_rows": len(validation_records),
                       "test_overlap_removed": len(validation_records) - len(validation)
                       - sum(validation_report.values()),
                       **validation_report, "kept_rows": len(validation)},
        "test_overlap_checked": test_records is not None,
    }
    return train, validation, report


def select_training_records(
    records: list[dict[str, Any]], max_train: int | None, *, seed: int = 42
) -> list[dict[str, Any]]:
    """Select a reproducible proportional subset with category coverage."""
    if max_train is not None and (
        isinstance(max_train, bool) or not isinstance(max_train, int) or max_train < 1
    ):
        raise ValueError("max_train must be a positive integer or None.")
    rng = random.Random(seed)
    if max_train is None or max_train >= len(records):
        selected = list(records)
    else:
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            groups.setdefault(str(record.get("category")), []).append(record)
        categories = sorted(groups)
        minimum = int(max_train >= len(categories))
        counts = {c: minimum for c in categories}
        available = sum(len(groups[c]) - minimum for c in categories)
        remaining = max_train - minimum * len(categories)
        quotas = {c: remaining * (len(groups[c]) - minimum) / available for c in categories}
        for c in categories:
            counts[c] += math.floor(quotas[c])
        remainder = max_train - sum(counts.values())
        for c in sorted(categories, key=lambda c: (-(quotas[c] % 1), c))[:remainder]:
            counts[c] += 1
        selected = [r for c in categories for r in rng.sample(groups[c], counts[c])]
    rng.shuffle(selected)
    return selected


def _example_token_ids(example: dict[str, Any], tokenizer: Any) -> list[int]:
    return tokenizer.apply_chat_template(
        example["prompt"] + example["completion"], tokenize=True, add_generation_prompt=False,
        return_dict=False,
    )


def filter_records_by_length(
    records: list[dict[str, Any]], tokenizer: Any, *, max_length: int = DEFAULT_MAX_SEQ_LENGTH,
    policy: str = "error",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Check every chat-formatted example; never truncate target JSON."""
    if max_length < 1 or policy not in {"error", "drop"}:
        raise ValueError("Use a positive max_length and policy='error' or 'drop'.")
    kept = []
    longest = 0
    for record in records:
        example = build_training_examples([record])[0]
        length = len(_example_token_ids(example, tokenizer))
        longest = max(longest, length)
        if length <= max_length:
            kept.append(record)
    report = {"input_rows": len(records), "over_max_length": len(records) - len(kept),
              "kept_rows": len(kept), "max": longest, "max_length": max_length}
    if report["over_max_length"] and policy == "error":
        raise ValueError(
            f"{report['over_max_length']} examples exceed {max_length} tokens (longest: {longest}). "
            "Increase the context budget or explicitly filter with policy='drop'; JSON is not truncated."
        )
    if not kept:
        raise ValueError("No examples fit the context budget.")
    return kept, report


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def token_length_report(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_length: int = DEFAULT_MAX_SEQ_LENGTH,
    sample: int | None = 200,
    prompt_fn: Callable[[str], str] = render_json_prompt,
) -> dict[str, Any]:
    """Token-length distribution of prompt+completion pairs, to size max length."""
    rows = records
    if sample is not None and sample < len(records):
        rows = random.Random(42).sample(records, sample)
    lengths: list[int] = []
    for example in build_training_examples(rows, prompt_fn=prompt_fn):
        lengths.append(len(_example_token_ids(example, tokenizer)))
    return {
        "sampled": len(rows),
        "min": min(lengths) if lengths else None,
        "median": _percentile(lengths, 0.5),
        "p90": _percentile(lengths, 0.9),
        "max": max(lengths) if lengths else None,
        "over_max_length": sum(1 for length in lengths if length > max_length),
        "max_length": max_length,
    }


def verify_dataset(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any | None = None,
    max_length: int = DEFAULT_MAX_SEQ_LENGTH,
    sample: int | None = 200,
) -> dict[str, Any]:
    """Review a curated split: counts, schema validity, duplicates, and lengths."""
    categories: Counter[str] = Counter()
    invalid_targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    empty_text = 0
    for record in records:
        categories[str(record.get("category"))] += 1
        if not (record.get("resume_text") or "").strip():
            empty_text += 1
        try:
            ResumeOutput.model_validate(record.get("target_json"))
        except Exception as error:  # noqa: BLE001 - report, never raise while verifying
            invalid_targets.append(
                {"resume_id": record.get("resume_id"), "error": f"{type(error).__name__}: {error}"[:300]}
            )
        digest = _record_digest(record)
        if digest in seen:
            duplicates += 1
        else:
            seen.add(digest)

    summary: dict[str, Any] = {
        "count": len(records),
        "categories": dict(sorted(categories.items())),
        "invalid_targets": len(invalid_targets),
        "invalid_target_examples": invalid_targets[:5],
        "duplicate_records": duplicates,
        "records_missing_resume_text": empty_text,
    }
    if tokenizer is not None and records:
        summary["token_lengths"] = token_length_report(
            records, tokenizer, max_length=max_length, sample=sample
        )
    return summary


# ---------------------------------------------------------------------------
# Training examples
# ---------------------------------------------------------------------------


def build_training_examples(
    records: list[dict[str, Any]],
    *,
    prompt_fn: Callable[[str], str] = render_json_prompt,
) -> list[dict[str, Any]]:
    """Convert records into conversational prompt/completion examples.

    The user turn reuses the evaluation prompt so training and evaluation share
    one format; the assistant turn is the compact target JSON.
    """
    examples: list[dict[str, Any]] = []
    for record in records:
        prompt = prompt_fn(record.get("resume_text", ""))
        completion = json.dumps(record.get("target_json"), ensure_ascii=False)
        examples.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "completion": [{"role": "assistant", "content": completion}],
            }
        )
    return examples


def build_datasets(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_train: int | None = None,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    seed: int = 42,
    test_records: list[dict[str, Any]] | None = None,
) -> tuple[Any, Any]:
    """Build Hugging Face ``Dataset`` objects for training and evaluation."""
    from datasets import Dataset

    train_records, validation_records, cleaning = prepare_training_records(
        train_records, validation_records, test_records=test_records
    )
    if len(train_records) != cleaning["train"]["input_rows"] or len(validation_records) != cleaning["validation"]["input_rows"]:
        warnings.warn(f"Dataset preparation excluded rows: {cleaning}", stacklevel=2)
    train_records, _ = filter_records_by_length(train_records, tokenizer, max_length=max_seq_length)
    validation_records, _ = filter_records_by_length(validation_records, tokenizer, max_length=max_seq_length)
    train_examples = build_training_examples(select_training_records(train_records, max_train, seed=seed))
    validation_examples = build_training_examples(validation_records)
    return Dataset.from_list(train_examples), Dataset.from_list(validation_examples)


# ---------------------------------------------------------------------------
# Model + LoRA
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    load_in_4bit: bool = True,
    device_map: Any | None = None,
    compute_dtype: Any | None = None,
) -> tuple[Any, Any]:
    """Load the base model (4-bit by default) and its chat tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cuda = torch.cuda.is_available()
    if load_in_4bit and not cuda:
        raise RuntimeError("This QLoRA workflow requires a CUDA GPU. Select a GPU runtime in Colab.")
    if compute_dtype is None:
        compute_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16) if cuda else torch.float32
    if compute_dtype == torch.bfloat16 and cuda and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("This GPU does not support bf16; use float16.")
    if device_map is None:
        device_map = {"": torch.cuda.current_device()} if cuda else {"": "cpu"}
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=_hf_token())
    if not tokenizer.chat_template:
        raise ValueError("The tokenizer must provide a chat template.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            ),
            device_map=device_map,
            dtype=compute_dtype,
            token=_hf_token(),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=compute_dtype,
            device_map=device_map,
            token=_hf_token(),
        )
    model.config.use_cache = False
    return model, tokenizer


def build_lora_config(
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    """Build the PEFT LoRA configuration for QLoRA."""
    from peft import LoraConfig

    if r < 1 or alpha < 1 or not 0 <= dropout < 1:
        raise ValueError("LoRA requires positive r/alpha and dropout in [0, 1).")
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or list(DEFAULT_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def configure_wandb_env(
    *,
    api_key: str | None = None,
    entity: str | None = None,
    project: str | None = None,
    run_name: str | None = None,
) -> None:
    """Export W&B settings so the Trainer authenticates without prompting."""
    values = {
        "WANDB_API_KEY": api_key or os.getenv("WANDB_API_KEY"),
        "WANDB_ENTITY": entity or os.getenv("WANDB_ENTITY"),
        "WANDB_PROJECT": project or os.getenv("WANDB_PROJECT") or "resume-parser",
    }
    if run_name:
        values["WANDB_NAME"] = run_name
    for key, value in values.items():
        if value:
            os.environ[key] = value


def build_training_arguments(
    output_dir: Path | str,
    *,
    epochs: float = 2.0,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    save_steps: int = 100,
    eval_steps: int = 100,
    logging_steps: int = 10,
    save_total_limit: int = 2,
    warmup_ratio: float = 0.03,
    lr_scheduler_type: str = "cosine",
    run_name: str = "qlora-finetune",
    use_wandb: bool = True,
    gradient_checkpointing: bool = True,
    bf16: bool | None = None,
    seed: int = 42,
    per_device_eval_batch_size: int = 1,
) -> Any:
    """Build ``SFTConfig`` for a QLoRA run with periodic checkpoints."""
    import torch
    from trl import SFTConfig

    if epochs <= 0 or learning_rate <= 0 or not 0 <= warmup_ratio < 1:
        raise ValueError("Use positive epochs/learning_rate and warmup_ratio in [0, 1).")
    for name, value in {
        "per_device_batch_size": per_device_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_seq_length": max_seq_length, "save_steps": save_steps, "eval_steps": eval_steps,
        "logging_steps": logging_steps, "save_total_limit": save_total_limit,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    cuda = torch.cuda.is_available()
    bf16_supported = cuda and torch.cuda.is_bf16_supported(including_emulation=False)
    if bf16 is None:
        bf16 = bf16_supported
    if bf16 and not bf16_supported:
        raise ValueError("bf16 was requested but this CUDA device does not support it.")
    fields = inspect.signature(SFTConfig).parameters
    if "max_length" not in fields or "completion_only_loss" not in fields:
        raise RuntimeError("Unsupported TRL version. Install the project's pinned requirements and restart the kernel.")
    warmup = {"warmup_ratio": warmup_ratio} if "warmup_ratio" in fields else {"warmup_steps": warmup_ratio}

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        **warmup,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        eval_strategy="steps",
        eval_steps=eval_steps,
        bf16=bf16,
        fp16=cuda and not bf16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        max_length=max_seq_length,
        completion_only_loss=True,
        prediction_loss_only=True,
        packing=False,
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
        seed=seed,
    )


def train(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    *,
    lora_config: Any,
    training_args: Any,
    resume_from_checkpoint: str | bool | None = None,
) -> Any:
    """Run Supervised Fine-Tuning and return the trainer."""
    from trl import SFTTrainer

    if lora_config is not None and (
        getattr(model, "peft_config", None)
        or any(hasattr(layer, "lora_A") for layer in model.modules())
    ):
        raise ValueError("This model already has adapters. Reload the base model for an independent run.")
    if not len(train_dataset) or not len(eval_dataset):
        raise ValueError("Training and evaluation datasets must both be non-empty.")
    if not getattr(training_args, "completion_only_loss", False):
        raise ValueError("Use completion_only_loss=True with the prepared prompt/completion datasets.")
    if resume_from_checkpoint:
        if resume_from_checkpoint is True:
            from transformers.trainer_utils import get_last_checkpoint

            resume_from_checkpoint = get_last_checkpoint(training_args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError("No resumable checkpoint exists in this experiment's output directory.")
        checkpoint = Path(resume_from_checkpoint)
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
        if any(not (checkpoint / filename).is_file() for filename in required):
            raise ValueError("Resume requires a full Trainer checkpoint with optimizer/scheduler state, not an adapter export.")
    # A constructor failure must propagate with its original traceback. Retrying
    # may attach adapters twice and hide the actual preprocessing/config error.
    common = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": lora_config,
    }
    trainer = SFTTrainer(processing_class=tokenizer, **common)
    # Recent TRL versions cast quantized-model adapters to bf16 even when the
    # run uses fp16 (e.g. on a T4). Keep FP32 adapter gradients for FP16 scaling;
    # forward operations still use the Trainer's FP16 autocast context.
    if getattr(training_args, "fp16", False) and getattr(trainer.model, "peft_config", None):
        import torch

        for parameter in trainer.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(torch.float32)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer


def save_adapter(trainer: Any, tokenizer: Any, output_dir: Path | str) -> Path:
    """Persist the trained adapter and tokenizer to a local directory."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(destination))
    trainer.save_state()
    tokenizer.save_pretrained(str(destination))
    return destination


# ---------------------------------------------------------------------------
# Evaluation (reuses the step-04 evaluator)
# ---------------------------------------------------------------------------


def make_generation_callback(
    model: Any,
    tokenizer: Any,
    *,
    model_name: str,
    max_new_tokens: int = 3072,
) -> Callable[[EvaluationItem], ModelCallbackOutput]:
    """Return a callback that generates schema JSON with the tuned model."""
    import torch

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")

    def callback(item: EvaluationItem) -> ModelCallbackOutput:
        prompt = render_json_prompt(item.resume_text)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        context_limit = getattr(model.config, "max_position_embeddings", None)
        if context_limit and inputs["input_ids"].shape[1] + max_new_tokens > context_limit:
            raise ValueError("Prompt plus max_new_tokens exceeds the model context window; generation was not truncated.")
        was_training = model.training
        try:
            model.eval()
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        finally:
            model.train(was_training)
        generated = tokenizer.decode(
            output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        return ModelCallbackOutput(
            raw_output=generated,
            prompt=text,
            model_name=model_name,
            metadata={"max_new_tokens": max_new_tokens},
        )

    return callback


def evaluate_records(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    model_name: str,
    limit: int | None = None,
    max_new_tokens: int = 3072,
    seed: int = 42,
    show_progress: bool = True,
    output_dir: Path | str | None = None,
) -> Any:
    """Score the tuned model on curated records using the step-04 evaluator."""
    items = [
        EvaluationItem(
            resume_id=str(record.get("resume_id", "")),
            resume_text=record.get("resume_text", ""),
            target_json=record.get("target_json"),
            category=record.get("category"),
        )
        for record in records
    ]
    evaluator = Evaluator(items)
    callback = make_generation_callback(
        model, tokenizer, model_name=model_name, max_new_tokens=max_new_tokens
    )
    return evaluator.evaluate(
        callback=callback,
        workers=1,
        test_size=limit,
        seed=seed,
        show_progress=show_progress,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Publish + housekeeping
# ---------------------------------------------------------------------------


def push_adapter_to_hub(
    model: Any,
    tokenizer: Any,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    private: bool = False,
    commit_message: str = "Upload QLoRA adapter for resume parsing",
) -> str:
    """Create/update the model repo and upload the adapter and tokenizer.

    ``revision`` (for example ``trained-2000``) uploads to that branch so each
    training batch is versioned by how much was trained.
    """
    from huggingface_hub import create_repo

    token = _hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is required to push a model.")
    resolved = model_repo_id(repo_id)
    create_repo(resolved, repo_type="model", private=private, exist_ok=True, token=token)
    _ensure_revision(resolved, revision, "model", token)
    model.push_to_hub(resolved, revision=revision, token=token, commit_message=commit_message)
    tokenizer.push_to_hub(resolved, revision=revision, token=token, commit_message=commit_message)
    suffix = f"/tree/{revision}" if revision else ""
    return f"https://huggingface.co/{resolved}{suffix}"


def push_checkpoint_folder(
    folder: Path | str,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    path_in_repo: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
) -> str:
    """Upload a saved checkpoint folder (adapter + optimizer state) to the Hub."""
    from huggingface_hub import create_repo, upload_folder

    token = _hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is required to push checkpoints.")
    resolved = model_repo_id(repo_id)
    folder = Path(folder)
    create_repo(resolved, repo_type="model", private=private, exist_ok=True, token=token)
    _ensure_revision(resolved, revision, "model", token)
    upload_folder(
        folder_path=str(folder),
        repo_id=resolved,
        repo_type="model",
        revision=revision,
        path_in_repo=path_in_repo or folder.name,
        commit_message=commit_message or f"Upload checkpoint {folder.name}",
        token=token,
    )
    suffix = f"/tree/{revision}" if revision else ""
    return f"https://huggingface.co/{resolved}{suffix}"


def free_memory() -> None:
    """Collect unused objects/cache after callers delete live model/trainer references."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


ensure_data_directories()


__all__ = [
    "CURATED_DIR",
    "DATA_DIR",
    "DEFAULT_MAX_SEQ_LENGTH",
    "DEFAULT_MODEL_ID",
    "DEFAULT_TARGET_MODULES",
    "HF_DATASET_REPO",
    "HF_MODEL_REPO",
    "HF_USERNAME",
    "MODELS_DIR",
    "SPLIT_FILENAMES",
    "build_datasets",
    "build_lora_config",
    "build_training_arguments",
    "build_training_examples",
    "checkpoint_revision",
    "configure_wandb_env",
    "dataset_repo_id",
    "download_curated_splits",
    "download_split",
    "evaluate_records",
    "free_memory",
    "filter_records_by_length",
    "load_model_and_tokenizer",
    "load_records",
    "make_generation_callback",
    "model_repo_id",
    "push_adapter_to_hub",
    "push_checkpoint_folder",
    "prepare_training_records",
    "save_adapter",
    "select_training_records",
    "token_length_report",
    "train",
    "verify_dataset",
]
