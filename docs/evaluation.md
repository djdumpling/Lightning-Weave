# Evaluation

Evaluate exported Hugging Face checkpoints. Math and code use separately
trained students. Response tokens count the entire generated response, not
only the text within `<think>`.

| Benchmark | Questions | Responses / question | Response cap |
|---|---:|---:|---:|
| AIME 2024 | 30 | 64 | 32,768 |
| AIME 2025 | 30 | 64 | 32,768 |
| HMMT February 2025 | 30 | 64 | 32,768 |
| LiveCodeBench v5, from 2024-08-01 | 279 | 4 | 40,960 |
| LiveCodeBench v6, from 2025-02-01 | 131 | 4 | 40,960 |

All use temperature 0.6 and top-p 0.95. Accuracy is the mean correctness
over **all** responses (Avg@64 / Avg@4), not the probability that at least
one sample succeeds. Token counts use the corresponding student tokenizer.
Average columns give each benchmark equal weight.

## Mathematics

Use a separate Python environment with a vLLM version supporting your model:

```bash
pip install -r requirements-eval.txt
MODEL=checkpoints/student-hf TP_SIZE=1 OUTPUT_DIR=results/math \
  bash evaluation/eval_math.sh
```

The task YAMLs preserve the experiment prompts, stop strings and
`math_verify.parse` / `verify` grading. They configure 64 repeats and an
identity filter that retains every response. The pinned public
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9.1)
does **not** need a patched `--repeats` argument or modified built-in tasks.
The custom `accuracy` result averages all 64 judgments.

Set `TASKS=weave_aime24` to run one benchmark. `MODEL_ARGS` appends vLLM model
arguments, for example a model-specific context size or dtype. The engine
context limit must leave space for both prompt and full response.

Summarize each logged sample file:

```bash
python -m evaluation.summarize \
  --math-samples results/math/MODEL/samples_weave_aime24_TIMESTAMP.jsonl \
  --tokenizer checkpoints/student-hf \
  --output results/math/aime24-summary.json
```

Use the actual filename emitted by the harness. Upstream scheduling/runtime
changes can alter sampled outputs; these portable wrappers have CPU tests,
but their GPU execution has not yet been validated as a full release run.

## Code: prepare data and generate

Install the official [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
package in an evaluation environment with a compatible vLLM build. Its
dependency set can be separate from the math environment.

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git external/LiveCodeBench
pip install -e external/LiveCodeBench
git -C external/LiveCodeBench rev-parse HEAD
python -m evaluation.livecodebench_v6.lock_dataset \
  --release-version release_v6 \
  --output data/evaluation/lcb-v6.lock.json
python -m evaluation.generate_code \
  --model checkpoints/student-code-hf \
  --dataset-lock data/evaluation/lcb-v6.lock.json \
  --output results/code/v6-generations.json
```

Record the upstream checkout and environment with your results. For v5, use
`--release-version release_v5` and separate lock/output filenames. Pin
`--revision` to a dataset commit for repeated runs; locks record source file
hashes and exact question IDs. `--raw-dir` accepts an existing directory
containing the release's `test.jsonl`, `test2.jsonl`, etc.

Generation calls upstream LCB's `CodeQwenInstruct` prompt formatter and code
extractor, matching the experiment's Qwen3 style. It uses the student's own
tokenizer for lengths and vLLM's default model stopping behavior. It does
not execute model responses or deserialize private tests. `--llm-kwargs`
accepts model-specific vLLM engine options as JSON; configure a runtime that
supports the chosen architecture.

## Code: isolated scoring

**Generated programs are untrusted code.** Run `evaluation.score_code` only
inside a disposable container or stronger sandbox with networking disabled,
no credentials, a non-root user, CPU/memory/time limits, read-only source
and raw-data mounts, and only a dedicated results directory writable.
The acknowledgement flag is not a sandbox. LCB's reliability guard does not
provide a security boundary.

Make the raw release JSONL files available inside that sandbox and generate
a lock there with `--raw-dir /data`; lock paths must be valid in the scoring
container. The files should be the same checksum-verified release used for
generation. Do not mount a home directory or the entire workspace.

For example, use an image with Python and upstream LCB already installed.
`RAW_DATA` is a directory containing only the downloaded release JSONL
files; `RESULT_DIR` contains the saved generations and no sensitive files.
Both variables must be absolute paths. Choose resource limits for your host.

```bash
docker run --rm --network none --read-only \
  --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 8 --memory 32g --pids-limit 512 \
  --tmpfs /tmp:rw,nosuid,size=4g \
  --mount "type=bind,src=$(pwd)/evaluation,dst=/workspace/evaluation,readonly" \
  --mount "type=bind,src=${RAW_DATA},dst=/data,readonly" \
  --mount "type=bind,src=${RESULT_DIR},dst=/results" \
  --workdir /workspace --entrypoint bash "${SCORING_IMAGE}" -lc '
python -m evaluation.livecodebench_v6.lock_dataset \
  --release-version release_v6 --raw-dir /data \
  --output /results/lcb-v6.lock.json &&
python -m evaluation.score_code \
  --generation /results/v6-generations.json \
  --dataset-lock /results/lcb-v6.lock.json \
  --output /results/v6-evaluation.json \
  --workers 8 --acknowledge-isolated-execution'
```

The scorer uses the upstream fast test suite, a six-second test timeout,
and computes Avg@4 from the per-sample grades. Output contains the grades,
unmodified responses, tokenizer-based token lengths, and summary metrics.
Generation and scoring remain separate, so scoring does not load a model.
