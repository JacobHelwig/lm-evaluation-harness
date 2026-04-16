import re

import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def _pick_code_block(text: str, entry_point: str) -> str | None:
    """Return the ```python``` block most likely to contain the target function.
    Prefers blocks containing `def <entry_point>`, else any block with `def `,
    else the longest block. Returns None if no fenced block exists."""
    matches = _CODE_BLOCK_RE.findall(text)
    if not matches:
        return None
    with_target = [m for m in matches if f"def {entry_point}" in m]
    if with_target:
        return with_target[0]
    with_def = [m for m in matches if "def " in m]
    if with_def:
        return with_def[0]
    return max(matches, key=len)


def build_predictions_cot(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """CoT-friendly filter: extracts the function-bearing ```python``` block.
    If the extracted block contains `def <entry_point>`, uses it verbatim;
    otherwise prepends doc['prompt'] so the function gets a signature."""
    out = []
    for resp, doc in zip(resps, docs):
        row = []
        for r in resp:
            block = _pick_code_block(r, doc["entry_point"])
            if block is None:
                row.append(doc["prompt"] + r)
                continue
            block = block.strip()
            if f"def {doc['entry_point']}" in block:
                row.append(block)
            else:
                row.append(doc["prompt"] + block)
        out.append(row)
    return out
