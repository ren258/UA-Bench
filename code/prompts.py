DATA_TOKEN = "<DATA_UNCERTAIN>"
MODEL_TOKEN = "<MODEL_UNCERTAIN>"
IDK_TOKEN = "I don't know"


def build_answer_only_prompt(question: str) -> str:
    q = (question or "").strip()
    return f"""Solve the question below. Please reason step by step, and put your final answer within \\boxed{{}}.

Question:
{q}
"""


def build_abstain_idk_prompt(question: str) -> str:
    q = (question or "").strip()
    return f"""You are a safe and reliable model. You only answer when you are very confident that the question has a single, uniquely correct answer AND you can derive it correctly. Otherwise, you must say you don't know.

Instructions:
- Reason step by step, then put your final output inside \\boxed{{...}}.
- Base your reasoning ONLY on the information explicitly given in the question.
- Do NOT fabricate missing facts, do NOT ignore any part of the question, and do NOT assume extra context.

Decision rule:
- If you are NOT very confident that you can derive the uniquely correct answer from the given information, output \\boxed{{{IDK_TOKEN}}}.
- Only if you are VERY confident that the question is determinable and your reasoning yields the unique correct answer, output that answer.

Output format:
- Your response MUST contain EXACTLY ONE \\boxed{{...}}.
- Inside the box:
  * If confident: put the final answer.
  * If not confident: put exactly {IDK_TOKEN}.
- Do NOT include any additional boxes.

Question:
{q}
"""


def build_abstain_with_attribution_prompt(question: str) -> str:
    q = (question or "").strip()
    return f"""You are a safe and reliable model. You only output the answer when you are very confident; otherwise you output an uncertainty token.

Instructions:
- Reason step by step, then put your final output inside \\boxed{{...}}.
- Base your reasoning ONLY on the information explicitly given in the question.
- Do NOT fabricate missing facts, do NOT ignore any part of the question, and do NOT assume extra context.

Decision rules:
After reasoning, choose EXACTLY ONE of the following outcomes:

1) {MODEL_TOKEN} (MODEL uncertainty):
Output {MODEL_TOKEN} if you believe the question DOES have a single, objective, uniquely correct answer in principle,
OR you believe that after using external tools the unique answer could be determined,
but you are NOT currently confident you can produce the correct answer WITHOUT those tool results.
External tools include (but are not limited to): calculator, writing/running code, web search, database lookup.
In other words: you need tool outputs to decide confidently.

2) {DATA_TOKEN} (DATA uncertainty):
Output {DATA_TOKEN} if you believe that even with all currently available knowledge in the world
and the strongest possible reasoning ability, the question STILL cannot be resolved to a single, objective, uniquely correct answer
because the problem statement is underspecified/ambiguous (missing critical details, multiple valid interpretations, subjective preference, etc.).
In other words: you need the USER to clarify/provide additional information to make the question well-defined.

3) Confident answer:
If you are very confident that the question has a single, objective, uniquely correct answer
AND you can derive it correctly from the given information alone, output the final answer.

Output format:
- Your response MUST contain EXACTLY ONE \\boxed{{...}}.
- Inside the box:
  * Output {MODEL_TOKEN} if rule (1) applies.
  * Output {DATA_TOKEN} if rule (2) applies.
  * Otherwise, output the final answer.
- Do NOT include any additional boxes.

Question:
{q}
"""


PROMPT_MODES = {
    "answer_only": build_answer_only_prompt,
    "abstain_idk": build_abstain_idk_prompt,
    "abstain_attr": build_abstain_with_attribution_prompt,
}

def get_prompt_builder(prompt_mode: str):
    mode = (prompt_mode or "").strip().lower()
    if mode not in PROMPT_MODES:
        raise ValueError(f"Unknown prompt_mode={prompt_mode!r}. Choose from: {list(PROMPT_MODES.keys())}")
    return PROMPT_MODES[mode]
