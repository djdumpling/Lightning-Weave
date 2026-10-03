#!/usr/bin/env python3
"""Reasoning-only DOPD scores for cached responses: s = log p_post(z | h) − log p_pre(z | h).

z is a response's reasoning: the text after its opening <think> (or from its start, when the prompt itself opened the
block) through the closing </think>, or to the end of the text if the block never closed. h is the recipient's exact
rendered prompt text plus the response up to and including its opening <think>. Context and reasoning are encoded separately with the donors' shared tokenizer (they meet at the
<think> tag), with no chat template; the BOS token is prepended when the tokenizer defines one. Log-probabilities are
normalized over the tokenizer's vocabulary, not the padded embedding rows. Decision tokens after </think> are not
scored, so s is not the complete-response log-ratio.

Each donor model is loaded once and scores every row of this rank's share; the output joins on ``sample_id``
(``data_curation/decision_projection.py --scores``). Both donors are loaded at their recorded revisions: a local
directory must be that revision's Hugging Face snapshot, and a Hub id is fetched at it.

    python data_curation/score_reasoning.py --rollouts DIR --post POST --pre PRE --output scores.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import write_parquet
from data_curation.decision_projection import THINK_CLOSE, THINK_OPEN, reasoning_opened

SCHEMA = "reasoning_only_dopd_scores_v1"


@dataclass(frozen=True)
class Span:
    context: str
    target: str
    terminated: bool  # whether the reasoning ends with </think>


def reasoning_span(prompt: str, response: str) -> Span:
    """Split prompt + response into the scored reasoning and everything before it."""
    open_at, close_at = response.find(THINK_OPEN), response.find(THINK_CLOSE)
    if open_at != -1 and (close_at == -1 or open_at < close_at):
        start = open_at + len(THINK_OPEN)
    else:
        start = 0  # the prompt already opened the block (e.g. Qwen3-2507 templates), or there is no block
        if close_at == -1 and not reasoning_opened(prompt):
            return Span(prompt + response, "", False)  # no reasoning at all
    if close_at == -1:
        return Span(prompt + response[:start], response[start:], False)
    end = close_at + len(THINK_CLOSE)
    return Span(prompt + response[:start], response[start:end], True)


def encode_span(tokenizer, span: Span) -> tuple[list[int], list[int]]:
    context = tokenizer.encode(span.context, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        context = [tokenizer.bos_token_id, *context]
    return context, tokenizer.encode(span.target, add_special_tokens=False)


def target_log_prob(model, context_ids: list[int], target_ids: list[int], *, vocab_size: int, device) -> float:
    """Summed log-probability of ``target_ids`` after ``context_ids``; only the needed logits are materialized."""
    import torch

    if not target_ids:
        return 0.0
    ids = torch.tensor([context_ids + target_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        try:  # only the logits that predict the target
            logits = model(input_ids=ids, logits_to_keep=len(target_ids) + 1).logits[0, :-1, :vocab_size].float()
        except TypeError:  # a transformers release without logits_to_keep
            logits = model(input_ids=ids).logits[0, len(context_ids) - 1 : -1, :vocab_size].float()
    targets = torch.tensor(target_ids, dtype=torch.long, device=device).unsqueeze(-1)
    return float(logits.log_softmax(dim=-1).gather(dim=-1, index=targets).sum())


def model_source(name: str, revision: str) -> dict:
    """Loading arguments that pin ``revision``: a local directory must be its snapshot; a Hub id is fetched at it."""
    path = Path(name)
    if path.is_dir():
        resolved = path.resolve()
        if resolved.name != revision or resolved.parent.name != "snapshots":
            raise ValueError(f"{name} is not the Hugging Face snapshot of revision {revision}")
        return {"pretrained_model_name_or_path": str(path)}
    return {"pretrained_model_name_or_path": name, "revision": revision}


def tokenizer_digest(tokenizer) -> str:
    return hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()


def load_rows(paths: list[Path], rank: int, world_size: int) -> list[dict]:
    import pyarrow.parquet as pq

    files = sorted(file for path in paths for file in ([path] if path.is_file() else path.glob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet shards under {paths}")
    rows = []
    for file in files:
        table = pq.read_table(file, columns=["prompt", "metadata.sample_id", "metadata.prompt_id", "metadata.response"])
        rows.extend(
            {"prompt": prompt, "sample_id": str(sample_id), "prompt_id": str(prompt_id), "response": response}
            for prompt, sample_id, prompt_id, response in zip(
                *(table.column(index).to_pylist() for index in range(4)), strict=True
            )
        )
    return rows[rank::world_size]


def score_rows(model, tokenizer, rows: list[dict], *, device) -> list[float]:
    vocab_size = len(tokenizer)
    scores = []
    for row in rows:
        context, target = encode_span(tokenizer, reasoning_span(row["prompt"], row["response"]))
        scores.append(target_log_prob(model, context, target, vocab_size=vocab_size, device=device))
    return scores


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=Path, nargs="+", required=True)
    parser.add_argument("--post", required=True, help="the donor after efficiency training")
    parser.add_argument("--pre", required=True, help="the model it was trained from")
    parser.add_argument("--post-revision", required=True)
    parser.add_argument("--pre-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import pyarrow as pa
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sources = {role: model_source(getattr(args, role), getattr(args, f"{role}_revision")) for role in ("post", "pre")}
    tokenizers = {role: AutoTokenizer.from_pretrained(**source) for role, source in sources.items()}
    digests = {role: tokenizer_digest(tokenizer) for role, tokenizer in tokenizers.items()}
    if digests["post"] != digests["pre"]:
        raise ValueError(f"post and pre tokenizers differ ({digests}); their log-ratio would compare different texts")
    tokenizer = tokenizers["post"]
    rows = load_rows(args.rollouts, args.rank, args.world_size)
    columns = {"sample_id": [row["sample_id"] for row in rows], "prompt_id": [row["prompt_id"] for row in rows]}
    spans = [reasoning_span(row["prompt"], row["response"]) for row in rows]
    columns["terminated"] = [span.terminated for span in spans]
    columns["target_tokens"] = [len(encode_span(tokenizer, span)[1]) for span in spans]
    for role in ("post", "pre"):
        model = AutoModelForCausalLM.from_pretrained(
            **sources[role], torch_dtype=getattr(torch, args.dtype), attn_implementation=args.attn_implementation
        ).to(args.device).eval()
        columns[f"{role}_logprob"] = score_rows(model, tokenizer, rows, device=args.device)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    columns["score"] = [post - pre for post, pre in zip(columns["post_logprob"], columns["pre_logprob"], strict=True)]
    provenance = {
        "schema": SCHEMA,
        "post": {"model": args.post, "revision": args.post_revision},
        "pre": {"model": args.pre, "revision": args.pre_revision},
        "tokenizer_sha256": digests["post"],
        "dtype": args.dtype,
        "rank": args.rank,
        "world_size": args.world_size,
        "conditioning": "BOS + recipient prompt text + response through <think>; target = reasoning through </think>",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(columns).replace_schema_metadata({"provenance": json.dumps(provenance, sort_keys=True)})
    write_parquet(table, args.output)  # atomic, so an interrupted job leaves no partial file
    print(f"scored {len(rows)} responses -> {args.output}")


if __name__ == "__main__":
    main()
