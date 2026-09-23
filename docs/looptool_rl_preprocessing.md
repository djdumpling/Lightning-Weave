# LoopTool-23k RL preprocessing

`data_curation/prepare_looptool_rl.py` turns
[`zhangkangning/LoopTool-23k`](https://huggingface.co/datasets/zhangkangning/LoopTool-23k)
into a minimally modified, reproducible, RL-ready dataset for
`Qwen/Qwen3-4B-Thinking-2507`. It runs on CPUs only: it uses the tokenizer and
chat template, but no model weights, embeddings, LLM judgments, or
quality/difficulty filters. Later filtering should come from actual rollout
rewards and pass rates.

## Run

```bash
# Unit tests (offline; the real-tokenizer test is skipped unless the pinned tokenizer is cached)
uv run --no-project --python 3.12 --with pytest==8.4.2 --with jsonschema==4.25.1 --with datasketch==1.6.5 \
  --with transformers==4.57.1 --with jinja2==3.1.6 --with pyarrow \
  python -m pytest tests/test_prepare_looptool_rl.py -q | tee outputs/looptool_tests.log

# Smoke run
bash scripts/prepare_looptool_rl.sh --limit 500 --output-dir outputs/looptool_rl_smoke

# Full run (about 3.5 minutes on a 10-core laptop with --workers 8)
bash scripts/prepare_looptool_rl.sh --output-dir outputs/looptool_rl --max-prompt-tokens 10240 \
  --test-log outputs/looptool_tests.log
```

The wrapper runs
`uv run --no-project --python 3.12 --script data_curation/prepare_looptool_rl.py`.
Dependencies are pinned inline (PEP 723): `datasets==4.1.1`,
`transformers==4.57.1`, `jinja2==3.1.6`, `jsonschema==4.25.1`,
`datasketch==1.6.5`, and `bfcl-eval==2026.3.23` (the same BFCL pin as
`configs/bfcl_grpo`). Any CPU container with `uv` and Hugging Face access
works, including a Modal CPU function. No GPU is needed.

Key options (see `--help`):

| option | default |
|---|---|
| `--dataset`, `--dataset-revision` | `zhangkangning/LoopTool-23k`, `b6c572d442ed4f2177f23645d8e9a77522e712c3` |
| `--expected-rows` | `23040`; the run fails if the pinned source differs |
| `--tokenizer`, `--tokenizer-revision` | `Qwen/Qwen3-4B-Thinking-2507`, `768f209d9ea81521153ed38c47d515654e938aea` |
| `--output-dir` | `outputs/looptool_rl` (gitignored); `--overwrite` replaces a previous run |
| `--limit N` | first N rows, after the full row-count check |
| `--audit-only` | write audits and summaries, but not the RL JSONL files |
| `--near-dup-threshold`, `--minhash-permutations`, `--lsh-threshold`, `--seed` | `0.95`, `256`, `0.8`, `42` |
| `--max-prompt-tokens` | override the automatic prompt limit; use `10240` for the selected training split |
| `--bfcl-audit {auto,on,off}`, `--bfcl-data-dir` | `auto`: audit if `bfcl-eval` data is importable. `on` and `--bfcl-data-dir` fail if data is missing |
| `--review-clusters` | `50` near-duplicate clusters in the Markdown report |
| `--no-schema-type-alias-repair` | strict mode: reject Python-style schema type names |

## Output contract

Each line of `looptool_rl_canonical.jsonl` and `looptool_rl_train.jsonl`:

```json
{"id": "sha256:…", "messages": [...], "tools": [{"type": "function", "function": {...}}],
 "target": {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "...", "arguments": {...}}}]},
 "metadata": {"source_dataset": "...", "source_revision": "...", "source_index": 0, "conversation_kind": "single_turn|multi_turn",
              "target_kind": "single_call|parallel_call|text_no_call", "num_user_messages": 1, "num_assistant_messages": 0,
              "num_history_tool_calls": 0, "num_tool_responses": 0, "num_target_tool_calls": 1, "num_tools": 23,
              "target_has_text_and_calls": false, "repair_codes": [], "warning_codes": [], "prompt_tokens": 4970}}
```

- `id` is the SHA-256 of the dataset, revision, source index, and the three raw fields.
- Historical assistant calls are structured `tool_calls` with positional IDs
  (`call_0`, `call_1`, … in conversation order). Each observation is a
  `role="tool"` message carrying the matching `tool_call_id`. The IDs do not
  depend on the source index, so identical conversations hash identically.
- Arguments are JSON objects, never JSON-encoded strings. `target.content` is
  `""` for calls. Text targets have `tool_calls: []`.
- No chain-of-thought, `reasoning_content`, `<think>`, or Qwen framing tokens are stored.
- Lines use UTF-8, sorted keys, and compact separators, sorted by `source_index`.
  Message, tool, response, and call order are preserved. Nothing volatile is written.

With Slime's loader, use `--input-key messages --tool-key tools
--apply-chat-template`. The loader renders the same
`apply_chat_template(messages, tools=tools, add_generation_prompt=True)` prompt
that was measured here. A reward adapter reads `target` from the row.

## What the parser does

The pinned source uses a single instruction template. It consists of the task
policy, a `The current time is …` line, then the exact Qwen `# Tools …
<tools> … </tools> … </tool_call>.` block with one JSON tool per line.

- **Instruction.** The tool block is recognized by exact header and footer
  strings (`looptool.TOOLS_HEADER` / `TOOLS_FOOTERS`) and removed. The Qwen
  template regenerates it from `tools=`; the source's footer has a trailing
  period that the template omits. Everything before the block, policy and time
  line included, becomes the system message.
- **Tools.** Each JSON line becomes `{"type": "function", "function": <source object>}`.
  All source keys (`name`, `description`, `parameters`, and extras such as
  `category` or a tool-level `required`) are kept verbatim, in source order.
  Canonically identical duplicate definitions are removed after the first,
  preserving first-occurrence order, and logged. Definitions with the same
  name but different contents reject the row.
- **Schemas.** Validated against JSON Schema **Draft 7** with
  `jsonschema.Draft7Validator`. Only the following mechanical repairs are
  allowed, each logged with a reason code and JSON pointer:
  - `schema_type_alias`: an exact `type` string from the closed table
    `dict/Dict→object, float→number, int→integer, str→string, bool→boolean, list/List/tuple→array`.
    This is the same primitive table `configs/bfcl_grpo/bfcl_adapter.py` uses.
  - The exact suffix `", optional"` is stripped. Optionality remains controlled
    solely by the containing object's `required` list.
  - `List[T]` is recursively converted to `array` plus `items`. Fixed-length
    `Tuple[T, ...]` is converted to Draft 7 tuple validation (`items` schemas
    plus equal `minItems`/`maxItems`). Only this closed grammar is recognized.
  - `schema_empty_object_properties_added`: a bare `{"type": "object"}` gets `properties: {}`.

  Rows are also rejected for non-object or missing `parameters`, a root type
  other than `object`, meta-schema failures, and duplicate or undeclared
  `required` names.
- **Dialogue.** A small state machine handles `<|im_start|>ROLE` and
  `<|im_end|>`, not one broad regular expression.
  - The plain prefix before the first marker is a user message. Input without
    markers is a single user message; prose labels such as `Inquirer:` are
    never split.
  - A `user` block made only of `<tool_response>` blocks becomes `tool`
    messages, paired in order with the preceding assistant turn's calls.
  - Rows are rejected for text outside role blocks, a stray `<|im_end|>`, an
    `<|im_start|>` inside an open message, unknown roles, mismatched call and
    response counts, unanswered calls, or a prompt ending on an assistant turn.
  - Explicit `<think>…</think>` blocks are removed and logged. An unclosed
    `<think>` rejects the row.
- **Target.** One call, several calls, or text. Text is allowed only before
  the first call; that case is logged as `target_text_with_tool_calls`, and
  none occur in the pinned source. Rows are rejected for unbalanced tags,
  invalid JSON (strict: no NaN, no duplicate keys), extra call keys,
  non-object arguments, unknown tools, or an empty target.
- **Arguments vs. schema.** Historical mismatches are warnings
  (`history_arguments_schema_<validator>`,
  `history_arguments_undeclared_parameter`) and are never coerced because the
  source deliberately contains wrong calls followed by error observations.
  Target mismatches reject the row as `target_arguments_schema_mismatch`:
  targets are supervision and cannot be treated as recovery context.
- **Text.** NFC, CRLF/CR→LF, outer whitespace stripped. Internal whitespace,
  case, and punctuation are untouched. Tool definitions and argument values
  are not normalized.

## Filtering stages

1. **Conflicts.** Rows are grouped by canonical `tools + messages`. A group with
   two or more distinct targets is written in full to `looptool_conflicts.jsonl`
   and excluded. Parallel calls count as an unordered multiset here only.
2. **Exact duplicates.** Keyed by canonical `tools + messages + target`; the
   lowest source index is kept. Every removal is written to
   `looptool_exact_duplicates.jsonl`.
3. **Rendering.** Each row is rendered with
   `apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True)`,
   and `metadata.prompt_tokens` is recorded. Appending the target with
   `add_generation_prompt=False` must render and must extend the prompt text
   as a prefix.
4. **Near duplicates.**
   - *Text compared:* role-labelled non-system message text, historical calls
     (name + canonical arguments), and the target. Shingles are word 5-grams
     after NFKC, casefolding, and whitespace splitting; numbers and
     identifiers are kept.
   - *Candidates:* MinHash + LSH only proposes pairs. A row is removed only if
     its exact shingle Jaccard against the retained representative is at
     least 0.95.
   - *Strata:* comparisons happen only within one stratum, defined by tool
     schema fingerprint, role sequence, historical call-name sequence, target
     kind, target call count, and target call-name multiset. **Documented
     addition:** the system-text fingerprint is also part of the stratum, so
     rows with a different "current time" line never collapse.
   - *Representative:* the lowest prompt tokens, then canonical bytes, then
     source index. Removed rows never become representatives, so chains cannot
     collapse transitively. Texts under five words are not compared.
5. **BFCL contamination audit.** Covers every `BFCL_*.json` question file of
   `bfcl-eval==2026.3.23`. `format_sensitivity` is skipped because it is an id
   index, not questions. Multi-turn tools are resolved through
   `MULTI_TURN_FUNC_DOC_FILE_MAPPING`.
   - *Prompt:* the normalized concatenation of user messages.
   - *Signature:* the sorted `(name, alias-normalized parameters)` of the tool
     inventory. An empty inventory is a valid signature. An inventory that
     cannot be resolved is unknown and never matches.
   - *Removal:* exact prompt with an equal signature, or prompt Jaccard of at
     least 0.99 with an equal signature.
   - *Audit only:* exact prompt with a different schema, an exact single user
     message (five or more words), or Jaccard of at least 0.8. BFCL is never
     executed or used for tuning.
6. **Prompt length.** The limit is chosen from 2,048 / 4,096 / 8,192 / 10,240 /
   12,288 / 16,384:
   the smallest one that keeps ≥95% of rows overall and ≥90% of each main
   stratum (single-turn, multi-turn, single-call, parallel-call, text/no-call),
   with every one of those proportions moving ≤0.5 pp. If none qualifies, the
   limit is 16,384 and the missed criteria are reported. Rows over the limit
   are excluded whole (never truncated) and listed in
   `looptool_length_filtered.jsonl`. This limit is a dataset-selection choice,
   separate from the model's **262,144-token native context length**. The
   selected downstream maximum response is **32,768 generated tokens**, so a
   10,240-token prompt plus that response requires at least **43,008 tokens**
   of combined context capacity.

Behavior classes are purely mechanical:

- `single_turn`: exactly one user message and no historical calls.
- `single_call` / `parallel_call`: one / two or more target calls.
- `text_no_call`: everything else. Clarifications, refusals, and summaries are
  not separated because the source has no reliable label.

## Artifacts

| file | contents |
|---|---|
| `looptool_rl_canonical.jsonl` | valid, conflict-free, deduplicated, contamination-filtered rows (pre-length) |
| `looptool_rl_train.jsonl` | the prompt-length-selected subset |
| `looptool_rejections.jsonl` | `id`, `source_index`, `stage`, `reason`, `details` |
| `looptool_events.jsonl` | repairs, warnings, and info events per row |
| `looptool_conflicts.jsonl` | full rows of conflicting groups, with group id and target key |
| `looptool_exact_duplicates.jsonl` | removed/representative ids, indices, and hash |
| `looptool_duplicate_clusters.jsonl` | near-duplicate clusters: representative, members, exact Jaccard, stratum, excerpts |
| `looptool_length_filtered.jsonl` | rows excluded by the effective prompt limit |
| `looptool_bfcl_overlap.jsonl` | BFCL candidates with `disposition` (`removed` / `audit_only`) and `reason` |
| `looptool_summary.json`, `looptool_summary.md` | revisions, versions, arguments, per-stage counts and distributions, reasons, lengths, limit selection, validation, checksums |

After writing, the pipeline re-reads both RL files and checks each against the
contract (`looptool.check_row`): tool references exist, responses are paired,
and no framing tokens or `<think>` remain. It then re-renders every prompt and
target through the pinned tokenizer and confirms the token counts match
`metadata.prompt_tokens`.

## Results at the pinned revisions

Source `b6c572d…` (23,040 rows) and tokenizer `768f209…`, both resolved to
the requested revisions. Totals: 23,000 canonical rows and 22,849 train rows.

| stage | rows | single-turn | multi-turn | single-call | parallel | text/no-call |
|---|---|---|---|---|---|---|
| source (raw markers) | 23,040 | 24.72% | 75.28% | 62.81% | 24.83% | 12.36% |
| structural validation | 23,004 | 24.74% | 75.26% | 62.85% | 24.82% | 12.33% |
| conflicts / exact / near dedup | 23,004 | unchanged | | | | |
| BFCL contamination | 23,000 | 24.73% | 75.27% | 62.86% | 24.83% | 12.32% |
| prompt length ≤10,240 | 22,849 | 24.89% | 75.11% | 62.93% | 24.80% | 12.27% |

- **Rejected (36).** The prior 139 compound-type schema rejections are all
  recovered by the closed grammar above.
  - 4 `target_arguments_schema_mismatch` (JSON strings supplied where the
    schema requires nested objects)
  - 13 `prompt_ends_with_assistant`
  - 6 `unbalanced_think_tags` (unclosed `<think>`)
  - 4 `im_end_without_im_start`
  - 3 `unbalanced_tool_response_tags`
  - 3 `history_call_unknown_tool`
  - 2 `tool_call_arguments_not_object`
  - 1 `residual_control_tag`
- **Repairs.** 44,498 repeated definitions are removed across 17,434 rows.
  `schema_type_alias` occurs in 3,116 retained rows (20,359 type fields), with
  550 optional-suffix repairs, 39 list expansions, and 2 tuple expansions.
  The 139 formerly rejected schema rows are recovered before the four invalid
  targets are quarantined.
- **Warnings (not filtered).** 1,510 / 270 / 1,075 rows contain historical
  calls that break the schema's `type`, break `required`, or use undeclared
  parameters. These remain useful error/recovery histories.
- **Conflicts, exact duplicates, near duplicates: 0.** Six near-duplicate
  strata contain more than one row (93 rows), but the conservative LSH pass
  proposes no candidate pair, so no row is removed.
- **BFCL.** Removed 4 rows (`live_irrelevance_120..123`): identical user
  prompt and an identical empty tool inventory. No audit-only candidates.
- **Prompt tokens (canonical).** p50 4,660; p75 5,963; p90 7,213; p95 8,014;
  p99 9,713; max 16,451; mean 4,628. The share at or below each candidate is
  2K 13.73%, 4K 37.79%, 8K 95.75%, 10K 99.34%, 12K 99.91%, and 16K 99.996%.
- **Limit.** 8,192 still fails because it shifts the single/multi-turn split
  by 1.10 pp. 10,240 is the smallest candidate satisfying every criterion;
  it retains 22,849 rows (99.34%) and excludes 151 whole rows.
- **Validation and determinism.** Both RL files pass the contract,
  duplicate-tool-name check, re-rendering, and token-count checks. The focused
  offline suite has 50 passing tests. The full run takes about 3.5 minutes on
  CPU.

## Reward caveats

- Tool-call rewards should compare the parsed final action after the policy's
  own reasoning (for example, name plus canonical arguments, with parallel
  calls as a multiset). No reference reasoning is stored.
- `text_no_call` references are kept, but exact string matching against a
  clarification, refusal, or summary is not a valid semantic reward.
