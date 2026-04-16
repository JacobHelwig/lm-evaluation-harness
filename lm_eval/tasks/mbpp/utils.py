import re
from typing import Union

import evaluate as hf_evaluate


try:
    pass_at_k = hf_evaluate.load("code_eval")

    # run simple test to check code execution is enabled before model generation
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_1(
    references: Union[str, list[str]], predictions: Union[str, list[list[str]]]
) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


def extract_code_blocks(text: str) -> str:
    # Pattern to match ```...``` blocks
    pattern = r"```(?:\w+)?\n?(.*?)\n?```"
    # (+ ```) as we add the opening "```python" to the gen_prefix
    matches = re.findall(pattern, r"```" + text, re.DOTALL)
    # if no matches, try to match ```...``` blocks (after removing the language)
    if not matches:
        text_without_lang = re.sub(r"```python", "```", text)
        matches = re.findall(pattern, text_without_lang, re.DOTALL)
    if not matches:
        return ""
    else:
        return matches[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[extract_code_blocks(r) for r in resp] for resp in resps]


_COT_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


_FUNC_NAME_RE = re.compile(r"^\s*assert\s+(\w+)\s*\(", re.MULTILINE)


def _extract_def_span(text: str, func_name: str | None) -> str | None:
    """Find a `def <name>(...)` block in free-form text and return it.
    Captures until a dedent (non-indented line that isn't blank/comment)."""
    if func_name:
        pattern = re.compile(rf"^([ \t]*)def {re.escape(func_name)}\b", re.MULTILINE)
    else:
        pattern = re.compile(r"^([ \t]*)def \w+\b", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    # Take the LAST def (often the model's final attempt after deliberation).
    m = matches[-1]
    start = m.start()
    lines = text[start:].splitlines()
    kept = [lines[0]]
    for line in lines[1:]:
        # Stop when a non-blank, non-indented, non-comment line appears.
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            kept.append(line)
            continue
        if line[:1] in (" ", "\t"):
            kept.append(line)
        else:
            break
    return "\n".join(kept).rstrip()


def build_predictions_cot(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Robust extractor used for both CoT and base mbpp.
    Order of preference:
      1. `def <target_funcname>(...)` block in a ```python``` fence
      2. Any ```python``` fence containing `def `
      3. Longest fenced block
      4. `def <target_funcname>(...):` span in free-form text (model reasoned in prose)
      5. Any `def <name>(...):` span
      6. Raw response (last resort)
    Target function name is sniffed from the first `assert <name>(` in test_list."""
    out = []
    for resp, doc in zip(resps, docs):
        func_name = None
        for t in doc.get("test_list", []):
            m = _FUNC_NAME_RE.search(t)
            if m:
                func_name = m.group(1)
                break
        row = []
        for r in resp:
            blocks = _COT_CODE_BLOCK_RE.findall(r)
            chosen = None
            if blocks:
                if func_name:
                    targeted = [b for b in blocks if f"def {func_name}" in b]
                    if targeted:
                        chosen = targeted[0]
                if chosen is None:
                    with_def = [b for b in blocks if "def " in b]
                    chosen = with_def[0] if with_def else max(blocks, key=len)
            else:
                chosen = _extract_def_span(r, func_name) or _extract_def_span(r, None)
            row.append((chosen or r).strip())
        out.append(row)
    return out


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
