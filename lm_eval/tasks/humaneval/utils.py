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


def _extract_last_code_block(text: str, fallback_prompt: str = "") -> str:
    """Return the last ```python ... ``` block's body; fall back to the full text."""
    matches = _CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1]
    # No fenced block — assume the whole response is code.
    return text


def build_predictions_cot(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """CoT-friendly filter: extracts the last ```python``` block as the candidate.
    If the extracted block already contains a `def <entry_point>`, use it verbatim
    (model emitted full function). Otherwise prepend doc['prompt'] so it gets a
    signature.
    """
    out = []
    for resp, doc in zip(resps, docs):
        row = []
        for r in resp:
            code = _extract_last_code_block(r).strip()
            if f"def {doc['entry_point']}" in code:
                row.append(code)
            else:
                row.append(doc["prompt"] + code)
        out.append(row)
    return out
