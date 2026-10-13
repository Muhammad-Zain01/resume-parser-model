"""Fine-tune a compact instruction model to parse resumes into the project schema.

Step 05 owns the fine-tuning workflow:

- download and verify the curated splits from the Hugging Face dataset repo;
- build chat-format training examples that use the *same* prompt as evaluation,
  so post-fine-tuning scores stay comparable to the pre-fine-tuning baseline;
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
import json
import os
import random
import sys
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


def _resolve_repo_id(env_var: str, default_name: str) -> str:
    """Resolve an owner/name repo id from an env var with legacy support."""
    repo_id = (os.getenv(env_var) or "").strip()
    if not repo_id:
        username = (os.getenv("HF_USERNAME") or "").strip()
        if not username:
            raise RuntimeError(f"Set {env_var}=owner/name in .env.")
        repo_id = f"{username}/{default_name}"
    elif "/" not in repo_id:
        username = (os.getenv("HF_USERNAME") or "").strip()
        if not username:
            raise RuntimeError(f"{env_var} must be owner/name, or set HF_USERNAME.")
        repo_id = f"{username}/{repo_id}"
    if repo_id.startswith("/") or repo_id.endswith("/") or repo_id.count("/") != 1:
        raise RuntimeError(f"{env_var} must use the format owner/name.")
    return repo_id


def dataset_repo_id() -> str:
    """Resolve the Hugging Face dataset repository id."""
    return _resolve_repo_id("HF_DATASET_REPO", "resume-parser-dataset")


def model_repo_id(repo_id: str | None = None) -> str:
    """Resolve the Hugging Face model repository id used for adapters."""
    if repo_id and repo_id.strip():
        return repo_id.strip()
    return _resolve_repo_id("HF_MODEL_REPO", "resume-parser-qwen2.5-3b")


# ---------------------------------------------------------------------------
# Dataset: download, load, verify
# ---------------------------------------------------------------------------


def download_split(split: str, *, curated_dir: Path | str = CURATED_DIR, force: bool = False) -> Path:
    """Download one curated split from the dataset repo if it is missing."""
    filename = SPLIT_FILENAMES.get(split)
    if filename is None:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_FILENAMES)}.")
    destination = Path(curated_dir) / filename
    if destination.is_file() and not force:
        return destination

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    repo_id = dataset_repo_id()
    revision = (os.getenv("HF_DATASET_REVISION") or "main").strip() or "main"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=f"curated/{filename}",
            local_dir=str(PROJECT_ROOT / ".data"),
            token=_hf_token(),
        )
    except EntryNotFoundError:
        # Support the earlier layout where split files sit at the repo root.
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=str(destination.parent),
            token=_hf_token(),
        )
    if not destination.is_file():
        raise FileNotFoundError(f"Split {split!r} was not written to {destination}.")
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
    for record in rows:
        prompt = prompt_fn(record.get("resume_text", ""))
        completion = json.dumps(record.get("target_json"), ensure_ascii=False)
        lengths.append(len(tokenizer(prompt + "\n" + completion, add_special_tokens=False)["input_ids"]))
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
    """Convert curated records into chat-format examples.

    The user turn reuses the evaluation prompt so training and evaluation share
    one format; the assistant turn is the compact target JSON.
    """
    examples: list[dict[str, Any]] = []
    for record in records:
        prompt = prompt_fn(record.get("resume_text", ""))
        completion = json.dumps(record.get("target_json"), ensure_ascii=False)
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            }
        )
    return examples


def build_datasets(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    *,
    max_train: int | None = None,
) -> tuple[Any, Any]:
    """Build Hugging Face ``Dataset`` objects for training and evaluation."""
    from datasets import Dataset

    train_examples = build_training_examples(train_records)
    if max_train is not None:
        train_examples = train_examples[:max_train]
    validation_examples = build_training_examples(validation_records)
    return Dataset.from_list(train_examples), Dataset.from_list(validation_examples)


# ---------------------------------------------------------------------------
# Model + LoRA
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    compute_dtype: Any | None = None,
) -> tuple[Any, Any]:
    """Load the base model (4-bit by default) and its chat tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=_hf_token())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if load_in_4bit:
        if compute_dtype is None:
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            ),
            device_map=device_map,
            token=_hf_token(),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
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
) -> Any:
    """Build ``SFTConfig`` for a QLoRA run with periodic checkpoints."""
    import torch
    from trl import SFTConfig

    if bf16 is None:
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        eval_strategy="steps",
        eval_steps=eval_steps,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=gradient_checkpointing,
        max_seq_length=max_seq_length,
        packing=False,
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
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

    common = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": lora_config,
    }
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **common)
    except TypeError:
        # Older trl versions accepted ``tokenizer`` instead of ``processing_class``.
        trainer = SFTTrainer(tokenizer=tokenizer, **common)
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

    def callback(item: EvaluationItem) -> ModelCallbackOutput:
        prompt = render_json_prompt(item.resume_text)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
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
    )


# ---------------------------------------------------------------------------
# Publish + housekeeping
# ---------------------------------------------------------------------------


def push_adapter_to_hub(
    model: Any,
    tokenizer: Any,
    *,
    repo_id: str | None = None,
    private: bool = False,
    commit_message: str = "Upload QLoRA adapter for resume parsing",
) -> str:
    """Create/update the model repo and upload the adapter and tokenizer."""
    from huggingface_hub import create_repo

    token = _hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is required to push a model.")
    resolved = model_repo_id(repo_id)
    create_repo(resolved, repo_type="model", private=private, exist_ok=True, token=token)
    model.push_to_hub(resolved, token=token, commit_message=commit_message)
    tokenizer.push_to_hub(resolved, token=token, commit_message=commit_message)
    return f"https://huggingface.co/{resolved}"


def push_checkpoint_folder(
    folder: Path | str,
    *,
    repo_id: str | None = None,
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
    upload_folder(
        folder_path=str(folder),
        repo_id=resolved,
        repo_type="model",
        path_in_repo=path_in_repo or folder.name,
        commit_message=commit_message or f"Upload checkpoint {folder.name}",
        token=token,
    )
    return f"https://huggingface.co/{resolved}"


def free_memory() -> None:
    """Release Python and CUDA memory between runs."""
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
    "MODELS_DIR",
    "SPLIT_FILENAMES",
    "build_datasets",
    "build_lora_config",
    "build_training_arguments",
    "build_training_examples",
    "configure_wandb_env",
    "dataset_repo_id",
    "download_curated_splits",
    "download_split",
    "evaluate_records",
    "free_memory",
    "load_model_and_tokenizer",
    "load_records",
    "make_generation_callback",
    "model_repo_id",
    "push_adapter_to_hub",
    "push_checkpoint_folder",
    "save_adapter",
    "token_length_report",
    "train",
    "verify_dataset",
]
