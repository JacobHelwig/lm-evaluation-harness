# Eval findings: boosting pass@1 on HumanEval & MBPP with Qwen3-4B

Notes on what we've learned while evaluating Qwen3-4B(-Instruct) on `humaneval`
and `mbpp` using this harness. Most of these are **parser / harness issues**,
not model capability issues — fixing them moves the numbers substantially.

Run setup: greedy decoding (`temperature=0`, `do_sample=False`),
`max_gen_toks=4096`, 0-shot, vLLM DP=4 on 4× RTX 6000 Ada.

## What our parser does beyond upstream (`c1c4bea3`)

All four tasks (`humaneval`, `humaneval_cot`, `mbpp`, `mbpp_cot`) now share a
single unified extractor, `build_predictions_cot`, in
`lm_eval/tasks/{humaneval,mbpp}/utils.py`. Differences from the stock upstream
parsers (`build_predictions`, `build_predictions_instruct`,
`extract_code_blocks`):

**Block selection**
1. Considers **all** fenced code blocks, not just the first.
2. **Prefers the block containing `def <target_funcname>`** — so trailing
   test-assert blocks don't get picked.
3. Fallback tiers within fences: target `def` → any `def ` → longest block.

**Target function inference**
4. **Sniffs expected function name from `assert <name>(` in `test_list`**
   (mbpp) or uses `entry_point` (humaneval), making block preference
   test-aware.

**Unfenced-prose fallback (mbpp)**
5. **Extracts a `def <name>(...):` span from free-form prose** when the model
   never emits a ` ```python` fence. Walks lines after `def` and captures
   until a dedent (non-indented, non-comment line).
6. **Takes the last `def` match** in prose — catches the model's final
   attempt after deliberation, not its first draft.

**Humaneval-specific safety**
7. **Avoids duplicate-def bug.** Only prepends `doc["prompt"]` if the
   extracted block doesn't already contain `def <entry_point>`. Upstream
   `build_predictions_instruct` blindly prepends, causing syntax errors when
   the model emits the full function.

**Fence-prepend fix**
8. **No spurious ` ``` ` prepend.** Upstream `extract_code_blocks` prepends
   ` ``` ` (relic of `gen_prefix` legacy), which produces an empty match when
   the response starts with its own ` ```python` fence.

**Empty-match handling**
9. **Falls back to the raw response** as last resort instead of returning an
   empty string (upstream `extract_code_blocks` returns `""` when no match,
   killing the grader).

**Covers `mbpp.yaml` at all**
10. Upstream `mbpp.yaml` has **no filter** — raw response went straight to the
    grader. We added `build_predictions_cot` as its filter so instruct-model
    reasoning prose doesn't get graded as Python source.

## Headline numbers

| Task            | Mode | Before fixes | After fixes | Δ |
| --------------- | ---- | -----------: | ----------: | ---: |
| `humaneval`     | nocot | 0.720       | 0.720*      | — |
| `mbpp`          | nocot | 0.320       | **≈0.70**† | +0.38 |
| `humaneval_cot` | cot   | 0.841        | 0.841*     | — |
| `mbpp_cot`      | cot   | 0.602        | 0.656       | +0.054 |

*humaneval already had a sane prompt-completion filter and its failures are
mostly real model errors.
†Projection from offline re-scoring of the saved samples with the new filter;
actual re-run not yet performed.

## Root-cause summary

### 1. Instruct model reasons even in raw-completion mode (biggest mbpp-nocot issue)

`mbpp.yaml` ends its prompt with `[BEGIN]\n` and expects the model to emit
Python code directly, stopping on `[DONE]`. Qwen3-4B-Instruct ignores that
framing and produces long reasoning prose ("Okay, I need to write..."),
sometimes hitting the generation limit before emitting code. Because `mbpp.yaml`
originally had **no extraction filter**, the prose was handed to the grader
as Python source, instantly failing to compile.

- 340 / 500 failures
- 318 / 340 had zero fenced code blocks (prose only)
- 318 / 340 produced a `SyntaxError` in the filtered output
- 280 / 340 responses exceeded 3500 chars (close to `max_gen_toks` limit)

**Fix:** added a robust filter `build_predictions_cot` in
`lm_eval/tasks/mbpp/utils.py` and wired it into `mbpp.yaml`. It prefers:

1. The ```` ```python ``` ```` block containing `def <target_funcname>` (sniffed from `assert <name>(...)` in `test_list`)
2. Any fenced block with `def `
3. The longest fenced block
4. A regex-extracted `def <target_funcname>(...):` span from free-form prose
   (takes the last such def, captures until a dedent)
5. Any `def <name>(...):` span
6. Raw response (last resort)

Projected impact on saved nocot samples: **32.0 % → 70.0 %** (190 of 340
failures become passing; 226 more now at least compile).

### 2. Multi-block responses — function + separate tests block (mbpp CoT)

Under CoT, models often produce the function in one fence and test asserts
in a second fence:

    ```python
    def foo(): ...
    ```

    ```python
    assert foo(...) == ...
    ```

The original `extract_code_blocks` and my initial "last block" filter both
picked the trailing test-only block, leaving the grader with zero function
definition.

- 68 / 199 original mbpp_cot failures (~34 %) were this pattern

**Fix:** filter now prefers the block containing `def <entry_point>`; see
point 1 above. After applying, mbpp_cot went from 60.2 % → 65.6 % (lingering
41 multi-block failures are real model errors, not parse issues).

### 3. Legacy ` ``` ` prepend in `extract_code_blocks`

`mbpp_instruct`'s legacy extractor prepends ` ``` ` to the response because
its template uses `gen_prefix="\n```python\n"`. When CoT responses start with
their own ` ```python`, that prepend collides and the regex matches an
empty block. Switching mbpp_cot off that function fixed it.

### 4. Qwen3's `<think>...</think>` block interacts poorly with chat-template CoT

Qwen3 auto-inserts a `<think>` block after `<|im_start|>assistant`. When
`enable_thinking=False` is passed, the chat template instead emits an **empty**
`<think></think>` pair, signalling "reasoning done". After that the model jumps
straight to code without CoT — even with a strong system instruction.

**Fix:** keep `enable_thinking=False`, and put the "think step by step
before writing code" instruction in the **user message** (via `doc_to_text`
in `humaneval_cot.yaml` / `mbpp_cot.yaml`). The model then reasons in plain
text inside its response, and we extract code from the final fenced block.

### 5. HumanEval CoT: signature mismatch would have eaten credit

HumanEval's grader runs `doc["prompt"] + response + check(...)`. If the CoT
response contains the full function inside a fenced block, concatenating with
`doc["prompt"]` creates a duplicate `def` and a syntax error.

**Fix:** `build_predictions_cot` in `lm_eval/tasks/humaneval/utils.py`
checks whether the extracted block already contains `def <entry_point>`; if
so, uses it verbatim (no signature prepend); otherwise prepends the signature.

## Harness ergonomics we added while debugging

Not performance-boosting, but makes diagnosis much faster:

- **`eval_code.sh`** is driver. `COT=1` switches between raw and CoT task variants; output dir tagged `_cot_`/`_nocot_`. `N_PREVIEW=N` dumps the first N prompt+response pairs into `preview.txt`.
- **Timing** in `evaluator.py`: logs `TIMING: generation=Xs  grading=Ys  total=Zs`. Surfaces in `preview.txt`.
- **Grading progress bar**: wrapped `doc_iterator` with tqdm per task, so the scoring phase no longer looks frozen for minutes.
- **`run.log`** saved next to results for post-hoc grep.

## Remaining issues / ideas

- **Generation-length cliff on mbpp nocot.** 280 of 340 failures had
  responses > 3500 chars — model still reasoning when it hit
  `max_gen_toks=4096`. Bumping to 8192 would likely rescue more; the offline
  re-score only measured the filter, not what a larger gen-budget would add.
- **No-fence no-def cases.** A few mbpp responses have neither fenced code
  nor a syntactically valid `def`. The current filter hands back the raw
  response; a more aggressive fallback (extract the last indented block
  starting with a keyword) could recover some.
- **Truncation after function definition.** Some model outputs end partway
  through test asserts after the def. These pass today because the def is
  extracted cleanly, but they'd look like failures if we ever tightened the
  "must compile end-to-end" check.
- **Native `<think>` mode** (enable_thinking=True). We haven't measured it;
  it'd require setting `think_end_token=</think>` and possibly a larger gen
  budget. Worth a run for comparison.

## Files touched

- `lm_eval/tasks/mbpp/mbpp.yaml` — added `filter_list` pointing at `build_predictions_cot`.
- `lm_eval/tasks/mbpp/mbpp_cot.yaml` — new CoT variant; inline user-prompt CoT instruction; custom filter.
- `lm_eval/tasks/mbpp/utils.py` — added `build_predictions_cot` with 6-tier fallback extractor.
- `lm_eval/tasks/humaneval/humaneval_cot.yaml` — new CoT variant; user-prompt CoT instruction.
- `lm_eval/tasks/humaneval/utils.py` — added `build_predictions_cot` that respects `entry_point`.
- `lm_eval/evaluator.py` — tqdm over grading; generation/grading timing logs.
- `eval_code.sh` — driver script (conda env, vLLM args, preview generation).
- `env.sh` — one-shot env setup (vllm==0.10.2 pin, ray, editable install).
