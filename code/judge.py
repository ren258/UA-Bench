from __future__ import annotations

import json
from typing import Dict, List, Tuple

from parallel_api_call import parallel_openai_calls



def build_judge_prompt(
    question: str,
    model_answer: str,
    reference_answers: List[str],
) -> str:
    q = (question or "").strip()
    a = (model_answer or "").strip()
    refs = reference_answers or []

    refs_json = json.dumps(refs, ensure_ascii=False)

    return f"""You are a strict but fair answer-matching judge.

You will be given:
(1) A question (may be open-ended or multiple-choice with options),
(2) A model answer,
(3) A list of reference answers (synonyms / acceptable variants).

Return "Yes" if the model answer matches ANY reference answer, or is a very close
synonym/paraphrase with the same meaning.

For multiple-choice questions, treat the following as equivalent when options
are present in the question:
- The model answer is an option letter (e.g., "A") while the reference answer is
  the corresponding option text (or vice versa)
- The model answer mentions the option content while the reference uses the
  option letter

Otherwise, return "No".

CRITICAL OUTPUT RULES:
- Output ONLY one token: Yes or No
- No punctuation, no explanations, no extra text

Question:
{q}

Model answer:
{a}

Reference answers (JSON list):
{refs_json}
"""



def parse_judge_yesno_strict(raw_text: str) -> int:
    s = (raw_text if isinstance(raw_text, str) else str(raw_text)).strip()
    s_low = s.lower()

    if s_low == "yes":
        return 1
    if s_low == "no":
        return 0

    raise ValueError(
        f"Judge output must be exactly 'Yes' or 'No' (case-insensitive). "
        f"Got: {s!r}"
    )



def judge_scores_parallel(
    items: List[Tuple[int, str, str, List[str]]],
    *,
    judge_model: str,
    judge_temperature: float,
    max_workers: int,
) -> Dict[int, int]:
    if not items:
        return {}

    prompts = [
        build_judge_prompt(q, ans, refs)
        for (_, q, ans, refs) in items
    ]

    judge_outputs = parallel_openai_calls(
        prompts,
        temperature=judge_temperature,
        model=judge_model,
        max_workers=max_workers,
    )

    score_map: Dict[int, int] = {}
    for (idx, _, _, _), out in zip(items, judge_outputs):
        score_map[idx] = parse_judge_yesno_strict(out)

    return score_map
