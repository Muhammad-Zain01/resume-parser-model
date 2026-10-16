"""Kaggle training worker, launched by the notebook with torchrun.

The notebook prepares all clean records and an uncapped context budget. Workers
receive paths/settings only; credentials remain in the inherited environment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))

    # Unsloth must patch the training stack before Transformers/TRL imports.
    from unsloth import FastLanguageModel  # noqa: F401
    import torch
    import torch.distributed as dist
    from datasets import load_from_disk
    from transformers import TrainerCallback
    from importlib import import_module

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    fine_tuning = import_module("src.steps.05_fine_tuning.fine_tuning")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    primary = int(os.environ.get("RANK", "0")) == 0
    torch.cuda.set_device(local_rank)
    if world_size != job["gpu_count"]:
        raise ValueError("Visible GPU count changed; prepare a new run or restore the original GPUs.")

    hp = job["hyperparameters"]
    if hp["epochs"] != 1 or hp["max_steps"] != -1 or hp["max_train_samples"] is not None:
        raise ValueError("This notebook worker requires one full epoch without sample/step caps.")
    train_dataset = load_from_disk(job["train_dataset"])
    validation_dataset = load_from_disk(job["validation_dataset"])
    if len(train_dataset) != job["train_count"] or len(validation_dataset) != job["validation_count"]:
        raise ValueError("Prepared dataset counts differ from the configured full splits.")

    fine_tuning.configure_wandb_env(
        entity=job["wandb_entity"], project=job["wandb_project"], run_name=job["run_name"],
    )
    os.environ["WANDB_GIT_ROOT"] = str(project_root)
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    training_args = fine_tuning.build_training_arguments(
        job["output_dir"], epochs=1, max_steps=-1,
        learning_rate=hp["learning_rate"], per_device_batch_size=hp["per_device_batch_size"],
        per_device_eval_batch_size=hp["per_device_eval_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        max_seq_length=hp["max_seq_length"], optim=hp["optim"],
        save_steps=hp["save_steps"], save_total_limit=hp["save_total_limit"],
        eval_steps=hp["eval_steps"], evaluate_during_training=True, eval_strategy="steps",
        warmup_ratio=hp["warmup_ratio"], lr_scheduler_type=hp["lr_scheduler_type"],
        run_name=job["run_name"], use_wandb=job["use_wandb"], bf16=False, seed=hp["seed"],
    )
    model, tokenizer = fine_tuning.load_model_and_tokenizer(
        job["model_id"], max_seq_length=hp["max_seq_length"], load_in_4bit=True,
    )
    model = fine_tuning.apply_lora(
        model, r=hp["lora_r"], alpha=hp["lora_alpha"], dropout=hp["lora_dropout"],
        target_modules=hp["target_modules"], seed=hp["seed"],
    )

    class FinalValidationCheckpoint(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # The final partial save interval must also evaluate and checkpoint.
            if state.global_step >= state.max_steps:
                control.should_evaluate = True
                control.should_save = True
            return control

    callbacks = [FinalValidationCheckpoint()]
    checkpoint_callback = None
    if job["hub_checkpoints"]:
        checkpoint_callback = fine_tuning.build_hub_checkpoint_callback(
            repo_id=job["model_repo"], revision=job["revision"],
            private=job["hf_private"], step_revisions=True,
        )
        callbacks.append(checkpoint_callback)

    run = None
    succeeded = False
    try:
        if primary and job["use_wandb"]:
            import wandb

            run = wandb.init(
                project=job["wandb_project"], entity=job["wandb_entity"], id=job["run_id"],
                name=job["run_name"], resume="allow", job_type="train",
                config={"model_id": job["model_id"], **hp,
                        "train_count": len(train_dataset), "validation_count": len(validation_dataset),
                        "gpu_count": world_size},
            )
        torch.cuda.reset_peak_memory_stats(local_rank)
        trainer = fine_tuning.train(
            model, tokenizer, train_dataset, validation_dataset,
            training_args=training_args, resume_from_checkpoint=job["resume_from_checkpoint"],
            callbacks=callbacks,
        )
        if float(trainer.state.epoch or 0) < 1 - 1e-6:
            raise RuntimeError("Training ended before the full epoch; resume its checkpoint before publishing.")

        # save_model participates on all ranks; tokenizer/state writes belong to rank zero.
        trainer.save_model(job["adapter_dir"])
        if primary:
            trainer.save_state()
            tokenizer.save_pretrained(job["adapter_dir"])
        peak = {"gpu": local_rank,
                "allocated_gib": torch.cuda.max_memory_allocated(local_rank) / 2**30,
                "reserved_gib": torch.cuda.max_memory_reserved(local_rank) / 2**30}
        peaks = [None] * world_size
        if dist.is_initialized():
            dist.all_gather_object(peaks, peak)
        else:
            peaks = [peak]
        if primary:
            final_step = trainer.state.global_step
            name = f"resume-parser-step-{final_step}-{job['run_id']}"
            if run is not None:
                run.name = name
                run.summary["training/final_step"] = final_step
                run.summary["training/completed_epoch"] = trainer.state.epoch
            result = {
                "run_id": job["run_id"], "run_name": name, "global_step": final_step,
                "epoch": trainer.state.epoch, "train_count": len(train_dataset),
                "validation_count": len(validation_dataset), "peak_memory": peaks,
                "validation_history": [row for row in trainer.state.log_history if "eval_loss" in row],
                "checkpoint_commits": checkpoint_callback.uploads if checkpoint_callback else [],
            }
            (Path(job["output_dir"]) / "training_result.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8",
            )
            print(f"Completed one epoch: {len(train_dataset)} training resumes; "
                  f"{len(validation_dataset)} validation resumes per check; {world_size} GPUs.")
            print("Completed run name:", name)
        succeeded = True
    finally:
        if run is not None:
            run.finish(exit_code=0 if succeeded else 1)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
