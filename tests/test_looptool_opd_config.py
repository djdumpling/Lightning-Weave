from configs.looptool_opd import config as CONFIG
from data_curation import prepare_direct_opd_looptool as PREPARE


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is True
        users = " ".join(message["content"] for message in messages)
        names = " ".join(tool["function"]["name"] for tool in tools)
        return f"{users} {names} assistant"

    def __call__(self, prompts, **kwargs):
        assert kwargs["add_special_tokens"] is False
        assert kwargs["return_length"] is True
        return {"length": [len(prompt.split()) for prompt in prompts]}


def canonical_row(row_id, source_index, text, *, target_kind="single_call"):
    return {
        "id": row_id,
        "messages": [{"role": "user", "content": text}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup a value",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "target": {"role": "assistant", "content": "reference must not leak"},
        "metadata": {
            "source_index": source_index,
            "conversation_kind": "single_turn",
            "target_kind": target_kind,
        },
    }


def test_locked_recipe_matches_single_anchor_direct_opd_budget():
    recipe = CONFIG.RECIPE

    assert CONFIG.STUDENT_MODEL == "Qwen/Qwen3-4B"
    assert CONFIG.PRE_TEACHER_MODEL == "Qwen/Qwen3-4B-Base"
    assert CONFIG.POST_TEACHER_MODEL == "Qwen/Qwen3-4B-Thinking-2507"
    assert recipe.learning_rate == 1e-6
    assert recipe.alpha == 2.0
    assert recipe.global_batch_size == 64
    assert recipe.max_prompt_tokens == 8_192
    assert recipe.max_response_tokens == 2_048
    assert recipe.sequence_tokens == 10_240
    assert recipe.cached_trajectories == 12_800
    assert recipe.consumed_trajectories == 12_800
    assert recipe.optimizer_updates_per_round == 4
    assert recipe.optimizer_updates == 200


def test_runtime_artifacts_are_commit_or_digest_pinned():
    assert len(CONFIG.STUDENT_REVISION) == 40
    assert len(CONFIG.PRE_TEACHER_REVISION) == 40
    assert len(CONFIG.POST_TEACHER_REVISION) == 40
    assert "@sha256:" in CONFIG.RUNTIME_IMAGE
    assert "@sha256:" in CONFIG.ROLLOUT_IMAGE


def test_prompt_preparation_uses_full_pool_before_seeded_selection_and_drops_targets():
    source = [
        canonical_row("a", 0, "one two"),
        canonical_row("b", 1, "three four", target_kind="parallel_call"),
        canonical_row("c", 2, "five six", target_kind="text_no_call"),
        canonical_row("d", 3, "seven eight"),
    ]

    selected, summary = PREPARE.prepare_rows(
        source,
        FakeTokenizer(),
        max_prompt_length=8,
        num_prompts=3,
        seed=42,
        tokenizer_batch_size=2,
    )
    selected_again, _ = PREPARE.prepare_rows(
        source,
        FakeTokenizer(),
        max_prompt_length=8,
        num_prompts=3,
        seed=42,
        tokenizer_batch_size=3,
    )

    assert selected == selected_again
    assert summary["eligible_unique_prompts"] == 4
    assert summary["selected_prompts"] == 3
    assert all("target" not in row and "messages" not in row and "tools" not in row for row in selected)
    assert all("reference must not leak" not in row["prompt"] for row in selected)


def test_exact_rendered_prompt_duplicates_are_not_oversampled():
    source = [
        canonical_row("a", 0, "same prompt"),
        canonical_row("b", 1, "same prompt"),
        canonical_row("c", 2, "different prompt"),
    ]

    selected, summary = PREPARE.prepare_rows(
        source,
        FakeTokenizer(),
        max_prompt_length=8,
        num_prompts=2,
        seed=42,
        tokenizer_batch_size=2,
    )

    assert len(selected) == 2
    assert summary["eligible_unique_prompts"] == 2
    assert summary["exact_rendered_prompt_duplicates_removed"] == 1
    assert summary["duplicate_prompt_ids"] == [{"kept": "a", "removed": "b"}]
