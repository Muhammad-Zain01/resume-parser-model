"""Kaggle post-evaluation inference and anonymous, resumable Step 04 scoring.

No training or model publication is performed. Importing this module does not
load the model, read datasets, or contact TypeSafe.
"""
from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import sys
import time


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_case(path: Path, fingerprint: str) -> dict:
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved['fingerprint'] != fingerprint:
        raise ValueError('Saved scores belong to another model/dataset/configuration.')
    return saved


def make_callback(model, tokenizer, job: dict, fine_tuning, evaluation):
    import torch
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

    class CompleteJson(StoppingCriteria):
        def __init__(self, start):
            self.start = start
            self.complete = False

        def __call__(self, input_ids, scores, **kwargs):
            # Check periodically; do not keep generating after the first complete JSON object.
            if (input_ids.shape[1] - self.start) % 16:
                return False
            text = tokenizer.decode(input_ids[0, self.start:], skip_special_tokens=True)
            try:
                self.complete = isinstance(json.loads(evaluation.extract_json_object(text)), dict)
            except (ValueError, TypeError):
                self.complete = False
            return self.complete

    def callback(item):
        prompt = fine_tuning.render_training_prompt(item.resume_text, style='compact')
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(model.device)
        prompt_tokens = inputs['input_ids'].shape[1]
        budget = min(job['max_new_tokens'], job['context_length'] - prompt_tokens)
        if budget < 1:
            raise ValueError('Input exceeds the native context; this case is counted as a failure.')
        stop = CompleteJson(prompt_tokens)
        config = GenerationConfig(
            max_new_tokens=budget, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        with torch.inference_mode():
            output = model.generate(
                **inputs, generation_config=config,
                stopping_criteria=StoppingCriteriaList([stop]),
            )
        generated = output[0, prompt_tokens:]
        ended_with_eos = bool(len(generated) and int(generated[-1]) == tokenizer.eos_token_id)
        return evaluation.ModelCallbackOutput(
            raw_output=tokenizer.decode(generated, skip_special_tokens=True).strip(),
            model_name=f"{job['model_repo']}@{job['model_sha']}",
            metadata={
                'prompt_tokens': prompt_tokens, 'generated_tokens': len(generated),
                'generation_budget': budget,
                'hit_output_limit': len(generated) >= budget and not ended_with_eos and not stop.complete,
                'stopped_on_complete_json': stop.complete,
            },
        )
    return callback


RETRYABLE = {'CALLBACK_FAILED', 'SCORED_WITH_SEMANTIC_ERROR', 'SEMANTIC_JUDGE_FAILED'}
SCORE_FIELDS = (
    'status', 'json_valid', 'schema_valid', 'programmatic_score', 'semantic_score',
    'overall_score', 'identity_score', 'duration_seconds', 'model_metadata',
)
BREAKDOWN_FIELDS = {
    'path', 'group', 'metric', 'value', 'precision', 'recall', 'f1',
    'true_positive', 'false_positive', 'false_negative', 'exact_cases', 'jev_cases', 'jev_score',
}


def worker(job: dict, rank: int, world_size: int) -> None:
    # The notebook assigns one physical T4 to each child before CUDA imports.
    from unsloth import FastLanguageModel
    import torch
    root = Path(job['project_root'])
    sys.path.insert(0, str(root))
    fine_tuning = import_module('src.steps.05_fine_tuning.fine_tuning')
    evaluation = import_module('src.steps.04_evaluation')
    if not os.getenv('TYPESAFE_API_KEY'):
        raise RuntimeError('Enable TYPESAFE_API_KEY before post evaluation.')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Each evaluation worker must see exactly one CUDA GPU.')
    torch.manual_seed(42)
    records = fine_tuning.load_records(job['evaluation_path'])
    output = Path(job['output_dir'])
    pending = []
    for index in range(rank, len(records), world_size):
        path = output / 'cases' / f'case-{index + 1:05d}.json'
        if path.exists() and load_case(path, job['fingerprint'])['status'] not in RETRYABLE:
            continue
        pending.append((index, path))
    print(f'GPU worker {rank}: {len(pending)} cases remaining.', flush=True)
    if not pending:
        return
    # Unsloth reads adapter_config.json, loads its quantized base, and attaches
    # the existing trained adapter. Do not create a fresh LoRA adapter here.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=job['adapter_path'], max_seq_length=job['context_length'],
        dtype=torch.float16, load_in_4bit=True, device_map={'': 0},
        token=os.getenv('HF_TOKEN') or None,
    )
    if not getattr(model, 'peft_config', None):
        raise RuntimeError('Refusing to evaluate a base model without the trained adapter.')
    if not getattr(model, 'is_loaded_in_4bit', False):
        raise RuntimeError('The evaluation model must be loaded in 4-bit.')
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise RuntimeError('The adapter tokenizer requires its chat template and EOS token.')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.requires_grad_(False)
    model.eval()
    FastLanguageModel.for_inference(model)
    print(f"GPU {rank}: loaded {job['model_repo']}@{job['model_sha']} | "
          'trained QLoRA adapter attached | 4-bit | FP16 | inference ready.', flush=True)
    torch.cuda.reset_peak_memory_stats()
    callback = make_callback(model, tokenizer, job, fine_tuning, evaluation)
    consecutive_judge_errors = 0
    started = time.monotonic()
    for finished, (index, path) in enumerate(pending, 1):
        record = records[index]
        case = f'case-{index + 1:05d}'
        item = evaluation.EvaluationItem(
            resume_id=case, resume_text=record['resume_text'], target_json=record['target_json'],
            category=record.get('category'),
        )
        result = evaluation.Evaluator([item]).evaluate(
            callback, workers=1, test_size=None, show_progress=False, output_dir=None,
        ).results[0]
        saved = {
            'fingerprint': job['fingerprint'], 'case': case, 'category': item.category,
            **{name: getattr(result, name) for name in SCORE_FIELDS},
            'score_breakdown': [
                {key: value for key, value in entry.items() if key in BREAKDOWN_FIELDS}
                for entry in result.score_breakdown
            ],
        }
        # Never save prompts, targets, predictions, identity fields, or raw API errors.
        atomic_json(path, saved)
        print(f'GPU {rank}: {finished}/{len(pending)} | {case} | {result.status} | '
              f'{result.duration_seconds:.1f}s | output limit: '
              f'{bool(result.model_metadata.get("hit_output_limit"))}', flush=True)
        consecutive_judge_errors = (
            consecutive_judge_errors + 1 if result.status in
            {'SCORED_WITH_SEMANTIC_ERROR', 'SEMANTIC_JUDGE_FAILED'} else 0
        )
        if consecutive_judge_errors >= 3:
            print('TypeSafe scoring failed three times consecutively. Check TYPESAFE_API_KEY '
                  'and TypeSafe quota before resuming.', flush=True)
            raise RuntimeError('Three consecutive TypeSafe scoring errors. Scores are saved; check the key/quota before resuming.')
    atomic_json(output / f'gpu-{rank}.json', {
        'gpu': rank, 'allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'processed_this_session': len(pending), 'elapsed_seconds': time.monotonic() - started,
    })


def summarize(job: dict):
    """Rebuild the existing evaluator report from anonymous per-case score files."""
    import pandas as pd
    sys.path.insert(0, job['project_root'])
    evaluation = import_module('src.steps.04_evaluation')
    fine_tuning = import_module('src.steps.05_fine_tuning.fine_tuning')
    records = fine_tuning.load_records(job['evaluation_path'])
    root = Path(job['output_dir'])
    results = []
    table_rows = []
    field_rows = []
    for index, record in enumerate(records, 1):
        case = f'case-{index:05d}'
        path = root / 'cases' / f'{case}.json'
        if not path.exists():
            continue
        saved = load_case(path, job['fingerprint'])
        item = evaluation.EvaluationItem(case, record['resume_text'], record['target_json'], record.get('category'))
        result = evaluation.EvaluationResult(
            item=item, **{key: saved[key] for key in SCORE_FIELDS},
            score_breakdown=saved['score_breakdown'], target=record['target_json'],
            model_name=f"{job['model_repo']}@{job['model_sha']}",
        )
        results.append(result)
        table_rows.append({
            'case': case, 'category': item.category,
            **{key: saved[key] for key in SCORE_FIELDS if key != 'model_metadata'},
            **saved['model_metadata'],
        })
        for entry in saved['score_breakdown']:
            if entry.get('value') is not None:
                field_rows.append({'group': entry['group'], 'field': entry['path'], 'score': entry['value']})
    if not results:
        raise RuntimeError('No saved evaluation scores are available yet.')
    metrics = evaluation.Evaluator([])._summarize(results)
    complete = len(results) == len(records)
    metrics.update({
        'expected_rows': len(records), 'completed_rows': len(results),
        'all_rows_processed': complete,
        'retryable_error_count': sum(r.status in RETRYABLE for r in results),
        'output_limit_count': sum(bool(r.model_metadata.get('hit_output_limit')) for r in results),
        'json_valid_rate': metrics['json_valid_count'] / len(results),
        'schema_valid_rate': metrics['schema_valid_count'] / len(results),
    })
    atomic_json(root / 'metrics.json', metrics)
    table = pd.DataFrame(table_rows)
    table.to_csv(root / 'per_resume.csv', index=False)
    field_table = pd.DataFrame(field_rows, columns=['group', 'field', 'score'])
    if len(field_table):
        field_table = field_table.groupby(['group', 'field'], as_index=False).agg(
            mean_score=('score', 'mean'), scored_comparisons=('score', 'count'),
        )
    field_table.to_csv(root / 'per_field.csv', index=False)
    summary = pd.DataFrame([
        {'metric': key, 'value': value} for key, value in metrics.items()
        if isinstance(value, (int, float, bool)) or value is None
    ])
    summary.to_csv(root / 'summary.csv', index=False)
    report = evaluation.EvaluationReport(metrics, results, None, [])
    if complete:
        charts = evaluation.report_results({'resume-parser — post evaluation': report})
        charts.save(root / 'charts')
    return metrics, summary, table, field_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    arguments = parser.parse_args()
    try:
        worker(json.loads(Path(arguments.job).read_text()), arguments.rank, arguments.world_size)
    except Exception as error:
        # Third-party exception messages may include private resume or API data.
        print(f'Evaluation worker stopped ({type(error).__name__}). '
              'Anonymous completed cases remain saved; inspect status counts and Kaggle secrets/quota.',
              flush=True)
        sys.exit(1)
