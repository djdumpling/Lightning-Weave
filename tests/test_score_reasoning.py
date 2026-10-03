import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_curation import score_reasoning as scorer

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

PROMPT = "<|im_start|>user\nMove the file.<|im_end|>\n<|im_start|>assistant\n"


def test_spans_score_the_reasoning_through_its_closing_tag():
    span = scorer.reasoning_span(PROMPT, "<think>\nplan\n</think>\n\n<tool_call>{}</tool_call>")
    assert span == scorer.Span(PROMPT + "<think>", "\nplan\n</think>", True)
    opened_by_prompt = scorer.reasoning_span(PROMPT + "<think>\n", "plan\n</think>\n\nDone.")
    assert opened_by_prompt == scorer.Span(PROMPT + "<think>\n", "plan\n</think>", True)
    unfinished = scorer.reasoning_span(PROMPT, "<think>\nstill going")
    assert unfinished == scorer.Span(PROMPT + "<think>", "\nstill going", False)
    assert scorer.reasoning_span(PROMPT, "plain answer") == scorer.Span(PROMPT + "plain answer", "", False)
    # A prompt that opened the block: an unclosed response is all reasoning, not an empty target.
    opened = PROMPT + "<think>\n"
    assert scorer.reasoning_span(opened, "still going") == scorer.Span(opened, "still going", False)


def test_donors_load_at_their_recorded_revisions(tmp_path):
    snapshot = tmp_path / "models--org--donor" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    assert scorer.model_source(str(snapshot), "abc123") == {"pretrained_model_name_or_path": str(snapshot)}
    with pytest.raises(ValueError, match="snapshot of revision"):
        scorer.model_source(str(snapshot), "def456")
    with pytest.raises(ValueError, match="snapshot of revision"):
        scorer.model_source(str(tmp_path), "abc123")
    assert scorer.model_source("org/donor", "abc123") == {"pretrained_model_name_or_path": "org/donor",
                                                          "revision": "abc123"}


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("the Qwen3 tokenizer is not in the local Hugging Face cache")


def tiny_model(tokenizer, seed):
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(
        vocab_size=len(tokenizer) + 7,  # padded embedding rows, as real checkpoints have
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=512,
    )
    return transformers.Qwen3ForCausalLM(config).eval()


def test_target_log_prob_matches_full_logits_over_the_real_vocabulary(tokenizer):
    model = tiny_model(tokenizer, 0)
    context, target = scorer.encode_span(tokenizer, scorer.reasoning_span(PROMPT, "<think>\nplan it\n</think>\nok"))
    assert tokenizer.decode(target) == "\nplan it\n</think>"
    assert target[-1] == tokenizer.convert_tokens_to_ids("</think>")
    vocab = len(tokenizer)
    expected = 0.0
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([context + target])).logits[0].float()
    for offset, token in enumerate(target):
        expected += float(logits[len(context) + offset - 1, :vocab].log_softmax(-1)[token])
    assert scorer.target_log_prob(model, context, target, vocab_size=vocab, device="cpu") == pytest.approx(expected)
    assert scorer.target_log_prob(model, context, [], vocab_size=vocab, device="cpu") == 0.0


def snapshot_dir(root, role):
    return root / f"models--test--{role}" / "snapshots" / f"rev-{role}"


def donor_arguments(root):
    return ["--post", str(snapshot_dir(root, "post")), "--post-revision", "rev-post",
            "--pre", str(snapshot_dir(root, "pre")), "--pre-revision", "rev-pre"]


def test_main_writes_post_minus_pre_scores(tmp_path, tokenizer):
    for role, seed in (("post", 1), ("pre", 2)):
        tiny_model(tokenizer, seed).save_pretrained(snapshot_dir(tmp_path, role))
        tokenizer.save_pretrained(snapshot_dir(tmp_path, role))
    responses = ["<think>\nshort\n</think>\nA", "<think>\na longer plan\n</think>\nB", "<think>\nunfinished"]
    metadata = [
        {"sample_id": f"s{index}", "prompt_id": "p", "response": response} for index, response in enumerate(responses)
    ]
    pq.write_table(pa.table({"prompt": [PROMPT] * 3, "metadata": metadata}), tmp_path / "rollouts.parquet")
    output = tmp_path / "scores.parquet"
    scorer.main(["--rollouts", str(tmp_path / "rollouts.parquet"), *donor_arguments(tmp_path), "--output", str(output),
                 "--device", "cpu"])
    table = pq.read_table(output)
    rows = table.to_pydict()
    assert rows["sample_id"] == ["s0", "s1", "s2"] and rows["terminated"] == [True, True, False]
    for score, post, pre in zip(rows["score"], rows["post_logprob"], rows["pre_logprob"], strict=True):
        assert score == pytest.approx(post - pre) and post < 0 and pre < 0
    provenance = json.loads(table.schema.metadata[b"provenance"])
    assert provenance["schema"] == scorer.SCHEMA and provenance["tokenizer_sha256"] == scorer.tokenizer_digest(tokenizer)
    assert provenance["post"]["revision"] == "rev-post" and provenance["pre"]["revision"] == "rev-pre"


def test_donors_must_share_a_tokenizer(tmp_path, tokenizer):
    tiny_model(tokenizer, 1).save_pretrained(snapshot_dir(tmp_path, "post"))
    tokenizer.save_pretrained(snapshot_dir(tmp_path, "post"))
    other = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    other.add_tokens(["<extra_token>"])
    tiny_model(other, 2).save_pretrained(snapshot_dir(tmp_path, "pre"))
    other.save_pretrained(snapshot_dir(tmp_path, "pre"))
    pq.write_table(pa.table({"prompt": [PROMPT], "metadata": [{"sample_id": "s", "prompt_id": "p", "response": "x"}]}),
                   tmp_path / "rollouts.parquet")
    with pytest.raises(ValueError, match="tokenizers differ"):
        scorer.main(["--rollouts", str(tmp_path / "rollouts.parquet"), *donor_arguments(tmp_path), "--output",
                     str(tmp_path / "s.parquet"), "--device", "cpu"])
