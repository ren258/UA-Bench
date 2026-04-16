import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from parallel_api_call import parallel_openai_calls

from io_utils import load_dataset, save_json
from prompts import get_prompt_builder
from parsing import extract_boxed, parse_boxed_result
from judge import judge_scores_parallel
from metrics import Counts


TAIL_PREFIX = "...(last "
TAIL_MID = " chars):"


def _is_truncated_row(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    raw_output = row.get("raw_output", None)
    if raw_output is None or (isinstance(raw_output, str) and raw_output.strip() == ""):
        reasons.append("raw_output_empty")

    boxed = row.get("boxed", None)
    if isinstance(boxed, str):
        s = boxed.strip()
        if s.startswith(TAIL_PREFIX) and TAIL_MID in s:
            reasons.append("boxed_tail_fallback")

    return (len(reasons) > 0), reasons


def _load_existing_results(output_path: str) -> Dict[int, Dict[str, Any]]:
    if not os.path.exists(output_path):
        return {}

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        results = obj.get("results", None)
        if not isinstance(results, list):
            print(f"[resume] Found existing file but 'results' is not a list: {output_path}")
            return {}
        mapping: Dict[int, Dict[str, Any]] = {}
        for r in results:
            if not isinstance(r, dict):
                continue
            rid = r.get("id", None)
            if isinstance(rid, int):
                mapping[rid] = r
        print(f"[resume] Loaded existing results: {len(mapping)} items from {output_path}")
        return mapping
    except Exception as e:
        print(f"[resume] Failed to load existing output file: {output_path}")
        print(f"[resume] Error: {repr(e)}")
        return {}


def evaluate(
    input_path: str,
    output_path: str,
    method: str,
    model: str,
    temperature: float,
    max_workers: int,
    limit: Optional[int] = None,
    judge_model: Optional[str] = None,
    judge_temperature: float = 0.0,
) -> None:
    data = load_dataset(input_path)
    if limit is not None:
        data = data[:limit]

    existing_map = _load_existing_results(output_path)
    build_prompt = get_prompt_builder(method)

    rerun_indices: List[int] = []
    rerun_prompts: List[str] = []
    reuse_count = 0
    rerun_count = 0
    
    raw_outputs_by_idx: List[Optional[str]] = [None] * len(data)

    for i, ex in enumerate(data):
        old = existing_map.get(i, None)
        if old is not None:
            need_rerun, reasons = _is_truncated_row(old)
            if not need_rerun:
                raw_outputs_by_idx[i] = old.get("raw_output", "")
                reuse_count += 1
                continue
            else:
                rerun_indices.append(i)
                q = ex.get("question", "")
                rerun_prompts.append(build_prompt(q))
                rerun_count += 1
                print(f"[resume] Re-run id={i} reasons={reasons}")
        else:
            rerun_indices.append(i)
            q = ex.get("question", "")
            rerun_prompts.append(build_prompt(q))
            rerun_count += 1

    print(f"[resume] Total={len(data)} reuse={reuse_count} rerun={rerun_count}")

    t0 = time.time()
    if rerun_prompts:
        outputs_rerun = parallel_openai_calls(
            rerun_prompts,
            temperature=temperature,
            model=model,
            max_workers=max_workers,
        )
    else:
        outputs_rerun = []
    elapsed = time.time() - t0

    for idx, raw in zip(rerun_indices, outputs_rerun):
        raw_text = raw if isinstance(raw, str) else str(raw)
        raw_outputs_by_idx[idx] = raw_text

    for i in range(len(raw_outputs_by_idx)):
        if raw_outputs_by_idx[i] is None:
            raw_outputs_by_idx[i] = ""

    counts = Counts(n_total=len(data))
    per_item_results: List[Dict[str, Any]] = []

    parsed_cache: List[Dict[str, Any]] = []
    judge_items: List[Tuple[int, str, str, List[str]]] = []

    for i, ex in enumerate(data):
        question = ex.get("question", "")
        reference_answers = ex.get("reference_answers", None)
        should_abstain = bool(ex.get("should_abstain", False))
        metadata_json = ex.get("metadata_json", None)

        gt_label = "ABSTAIN" if should_abstain else "ANSWERABLE"
        counts.observe_gt(gt_label)

        raw_text = raw_outputs_by_idx[i] if isinstance(raw_outputs_by_idx[i], str) else str(raw_outputs_by_idx[i])

        boxed = extract_boxed(raw_text)
        pred_label, pred_answer, parse_err = parse_boxed_result(boxed)
        counts.observe_pred(pred_label=pred_label, parse_error=parse_err)

        if gt_label == "ANSWERABLE" and parse_err is None and pred_label == "ANSWERABLE":
            refs = reference_answers if isinstance(reference_answers, list) else []
            judge_items.append((i, question, pred_answer, refs))

        parsed_cache.append(
            {
                "id": i,
                "question": question,
                "reference_answers": reference_answers,
                "should_abstain": should_abstain,
                "metadata_json": metadata_json,
                "gt_label": gt_label,
                "raw_output": raw_text,
                "boxed": boxed,
                "pred_label": pred_label,
                "pred_answer": pred_answer,
                "parse_error": parse_err,
            }
        )

    jm = judge_model or model
    score_map = judge_scores_parallel(
        judge_items,
        judge_model=jm,
        judge_temperature=judge_temperature,
        max_workers=16,
    )

    for row in parsed_cache:
        i = row["id"]
        gt_label = row["gt_label"]
        pred_label = row["pred_label"]
        parse_err = row["parse_error"]

        is_correct: Optional[bool] = None
        if gt_label == "ANSWERABLE":
            if parse_err is None and pred_label == "ANSWERABLE":
                is_correct = bool(score_map.get(i, 0))
            else:
                is_correct = False

        counts.observe_outcome(
            gt_label=gt_label,
            pred_label=pred_label,
            parse_error=parse_err,
            is_correct=is_correct,
        )

        effective_label = pred_label
        if gt_label == "ANSWERABLE" and parse_err is None and pred_label == "ANSWERABLE" and (is_correct is False):
            effective_label = "MODEL_UNCERTAIN"

        judge_score = None
        if gt_label == "ANSWERABLE" and parse_err is None and pred_label == "ANSWERABLE":
            judge_score = int(bool(is_correct))

        per_item_results.append(
            {
                **row,
                "is_correct": is_correct,
                "judge_score": judge_score,
                "effective_label": effective_label,
            }
        )

    summary: Dict[str, Any] = {
        "meta": {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "method": method,
            "model": model,
            "temperature": temperature,
            "max_workers": max_workers,
            "judge_model": jm,
            "judge_temperature": judge_temperature,
            "n_total": counts.n_total,
            "elapsed_sec": round(elapsed, 3),
            "resume": {
                "existing_loaded": bool(existing_map),
                "reuse": reuse_count,
                "rerun": rerun_count,
            },
        },
        "counts": asdict(counts),
        "percent": counts.to_percent_dict(),
        "matrix_gt_pred_counts": counts.to_matrix_dict(),
    }

    out_obj = {
        "summary": summary,
        "results": per_item_results,
    }
    save_json(output_path, out_obj)

    print("=== Evaluation Done ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to .json or .jsonl dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save results (.json)")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Prompt method name (must match prompts.get_prompt_builder).",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_workers", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Optional: only evaluate first N examples")
    parser.add_argument("--judge_model", type=str, default=None, help="Optional: override judge model")
    parser.add_argument("--judge_temperature", type=float, default=0.0)

    args = parser.parse_args()

    evaluate(
        input_path=args.input_path,
        output_path=args.output_path,
        method=args.method,
        model=args.model,
        temperature=args.temperature,
        max_workers=args.max_workers,
        limit=args.limit,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
    )


if __name__ == "__main__":
    main()
