"""Compare base and step-0 vLLM generation and forced-token sampling scores."""
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    rows = [r for f in sorted(args.input.glob("*.parquet")) for r in pq.read_table(f).to_pylist()][:4]
    llm = LLM(model=args.model, dtype="bfloat16", language_model_only=True, gdn_prefill_backend="triton", generation_config="vllm", max_model_len=24576, gpu_memory_utilization=.7, max_num_seqs=4, max_num_batched_tokens=2048, compilation_config={"mode": 0, "cudagraph_mode": "NONE"}, seed=42)
    generated = llm.generate([{"prompt_token_ids": r["metadata"]["prompt_tokens"]} for r in rows], SamplingParams(temperature=0, top_p=1, top_k=-1, presence_penalty=0, max_tokens=256, logprobs=1, seed=42), use_tqdm=False)
    # Forced contexts cover complete smoke responses, including their last action.
    forced = llm.generate([{"prompt_token_ids": r["metadata"]["prompt_tokens"] + r["metadata"]["response_tokens"]} for r in rows], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1, seed=42), use_tqdm=False)
    output = []
    for row, gen, scored in zip(rows, generated, forced):
        m = row["metadata"]; begin = len(m["prompt_tokens"])
        lp = [scored.prompt_logprobs[begin+i][token].logprob for i, token in enumerate(m["response_tokens"])]
        output.append({"sample_id": m["sample_id"], "generated_tokens": list(gen.outputs[0].token_ids), "generation_logprobs": [x[t].logprob for t, x in zip(gen.outputs[0].token_ids, gen.outputs[0].logprobs)], "forced_logprobs": lp})
    args.output.write_text(json.dumps(output) + "\n")

if __name__ == "__main__": main()
