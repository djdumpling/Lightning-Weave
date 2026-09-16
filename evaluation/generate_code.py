#!/usr/bin/env python3
"""Generate four code responses per problem; never execute generated code."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from evaluation.livecodebench_v6.lock_dataset import load_locked_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=49152)
    parser.add_argument("--llm-kwargs", default="{}", help="Additional vLLM engine arguments as JSON")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    lock, rows = load_locked_rows(args.dataset_lock)

    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.prompts.code_generation import format_prompt_generation
    from lcb_runner.utils.extraction_utils import extract_code
    from vllm import LLM, SamplingParams

    # The paper uses LCB's CodeQwenInstruct style for every student.
    # Only prompt fields are materialized; private test bundles are not unpickled.
    style = LMStyle.CodeQwenInstruct
    prompts = [
        format_prompt_generation(SimpleNamespace(
            question_content=row["question_content"],
            starter_code=row.get("starter_code", ""),
        ), style)
        for row in rows
    ]
    engine_kwargs = dict(
        model=args.model, tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len, dtype="bfloat16",
        gpu_memory_utilization=0.85, enable_prefix_caching=True, seed=42,
    )
    engine_kwargs.update(json.loads(args.llm_kwargs))
    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()
    outputs = llm.generate(prompts, SamplingParams(
        n=4, temperature=0.6, top_p=0.95, top_k=-1,
        max_tokens=40960, seed=42, stop=[],
    ))
    generations = []
    for row, output in zip(rows, outputs, strict=True):
        texts = [sample.text for sample in output.outputs]
        if len(texts) != 4:
            raise ValueError("Expected four responses per problem")
        generations.append({
            "question_id": row["question_id"],
            "output_list": texts,
            "code_list": [extract_code(text, style) for text in texts],
            "response_tokens": [
                len(tokenizer.encode(text, add_special_tokens=False)) for text in texts
            ],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(generations, ensure_ascii=False) + "\n")
    args.output.with_suffix(".protocol.json").write_text(json.dumps({
        "model": args.model, "release": lock["release_version"],
        "problem_count": len(rows), "samples_per_problem": 4,
        "temperature": 0.6, "top_p": 0.95, "max_tokens": 40960,
        "model_style": "CodeQwenInstruct", "llm_kwargs": engine_kwargs,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
