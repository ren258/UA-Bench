from __future__ import annotations

import re
from typing import Optional, Tuple

from prompts import DATA_TOKEN, MODEL_TOKEN, IDK_TOKEN



def extract_boxed(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None

    marker = r"\boxed{"
    start = raw_text.rfind(marker)
    if start != -1:
        i = start + len(marker) - 1
        if i < len(raw_text) and raw_text[i] == "{":
            depth = 0
            j = i
            while j < len(raw_text):
                ch = raw_text[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return raw_text[i + 1 : j].strip()
                    if depth < 0:
                        break
                j += 1

    tail_len = 100
    tail = raw_text[-tail_len:].strip()
    return f"...(last {tail_len} chars): {tail}"


def parse_boxed_result(boxed: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if boxed is None:
        return None, None, "no_boxed_found"

    s = boxed.strip()

    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    s_upper = s.upper()

    if IDK_TOKEN.upper() in s_upper or MODEL_TOKEN.upper() in s_upper:
        return "MODEL_UNCERTAIN", None, None

    if DATA_TOKEN.upper() in s_upper:
        return "DATA_UNCERTAIN", None, None

    return "ANSWERABLE", s, None


if __name__ == "__main__":
    test_outputs = [
        r"Here is my answer: \boxed{\text{42}}",
        r"The result is \boxed{  \text{I don't know}  } due to uncertainty.",
        r"\boxed{<DATA_UNCERTAIN>}",
        r"\boxed{Some final answer here}",
        r"No box here at all.",
        r"Multiple boxes: \boxed{First} and then \boxed{Second}",
        r"Nested box: \boxed{Outer \boxed{Inner}} end.",
        r"_1 = 1/\\sqrt{2} $\n\n✅ Final answer:\n\n\\boxed{\\dfrac{1}{\\sqrt{2}}}",
    ]

    for output in test_outputs:
        boxed = extract_boxed(output)
        label, answer, error = parse_boxed_result(boxed)
        print(f"Raw Output: {output}")
        print(f"Extracted Boxed: {boxed}")
        print(f"Parsed Result: label={label}, answer={answer}, error={error}")
        print("-" * 40)
    