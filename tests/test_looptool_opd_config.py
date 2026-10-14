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


def test_pools_name_their_student_and_only_the_2507_pool_changes_it():
    pools = CONFIG.STATE_POOLS
    for name in ("looptool", "tau2-fresh"):
        pool = pools[name]
        assert pool.student == CONFIG.QWEN3_4B and pool.model_type == "qwen3-4B" and pool.max_response_tokens == 2_048
        assert pool.generation_prompt == "<|im_start|>assistant\n"
        assert pool.checkpoint_root == "/checkpoints/looptool-offline-dopd-qwen3-4b-v1"
    strong = pools["tau2-fresh-2507"]
    assert strong.student == (CONFIG.POST_TEACHER_MODEL, CONFIG.POST_TEACHER_REVISION)
    assert strong.model_type == "qwen3-4B-Thinking-2507" and strong.max_response_tokens == 8_192
    assert strong.generation_prompt == "<|im_start|>assistant\n<think>\n"
    assert strong.checkpoint_root != pools["looptool"].checkpoint_root and strong.root != pools["tau2-fresh"].root
    # the same AReaL tasks and selection size as the pilot, with the mentor's decision states drawn first
    pilot = pools["tau2-fresh"].collection
    assert strong.collection["run_id"].startswith("collect-") and strong.collection["collectors"] == ["thinking2507"]
    assert strong.collection["prioritize_decisions"] and "prioritize_decisions" not in pilot
    assert {key: strong.collection[key] for key in ("domains", "tasks", "states_per_episode")} == {
        key: pilot[key] for key in ("domains", "tasks", "states_per_episode")
    }


def test_the_default_student_pools_keep_their_sealed_lock():
    resolved = CONFIG.RECIPE.resolved()
    assert CONFIG.DATA_PROFILE == "looptool" and "student_model_type" not in resolved
    assert resolved["student_behavior_and_trainable_policy"] == {"repo": "Qwen/Qwen3-4B", "revision": CONFIG.QWEN3_4B[1]}
    assert resolved["recipe"]["max_response_tokens"] == 2_048


def test_only_pools_longer_than_the_pilots_train_on_h200s():
    assert CONFIG.RECIPE.sequence_tokens == 8_192 + 2_048 and CONFIG.TRAINING_GPU == "H100:8"
    lengths = {name: pool.max_prompt_tokens + pool.max_response_tokens for name, pool in CONFIG.STATE_POOLS.items()}
    assert lengths == {
        "looptool": 10_240, "tau2-fresh": 18_432, "tau2-fresh-2507": 24_576, "tau2-retail-2507": 24_576,
        "tau2-retail-2507-cap2048": 24_576, "tau2-retail-2507-uniform": 24_576,
    }
    retail, mixed = CONFIG.STATE_POOLS["tau2-retail-2507"], CONFIG.STATE_POOLS["tau2-fresh-2507"]
    assert retail.collection["domains"] == ["tau2_retail"] and retail.collection["states_per_episode"] == 8
    assert retail.collection["run_id"] == mixed.collection["run_id"] and retail.root != mixed.root
    assert (retail.student, retail.checkpoint_root) == (mixed.student, mixed.checkpoint_root)


def test_cap_and_uniform_pools_change_only_state_selection():
    pools = CONFIG.STATE_POOLS
    retail, cap, uniform = pools["tau2-retail-2507"], pools["tau2-retail-2507-cap2048"], pools["tau2-retail-2507-uniform"]
    for pool in (cap, uniform):
        assert (pool.student, pool.model_type, pool.generation_prompt, pool.max_response_tokens, pool.checkpoint_root) == (
            retail.student, retail.model_type, retail.generation_prompt, retail.max_response_tokens, retail.checkpoint_root
        )
        assert pool.max_prompt_tokens == retail.max_prompt_tokens and pool.root != retail.root
        for key in ("results_volume", "run_id", "collectors", "domains", "tasks"):
            assert pool.collection[key] == retail.collection[key]
    assert len({retail.root, cap.root, uniform.root}) == 3
    # the mentor's cumulative budget, every in-budget state, no decision priority, the quota filled by repeats
    assert cap.collection["episode_token_budget"] == 2_048 and cap.collection["repeat_to_quota"]
    assert not cap.collection["prioritize_decisions"] and cap.collection["states_per_episode"] >= 8
    # uncapped and unprioritized; everything else as the decision-first pool
    assert {k: v for k, v in uniform.collection.items() if k != "prioritize_decisions"} == {
        k: v for k, v in retail.collection.items() if k != "prioritize_decisions"
    }
    assert retail.collection["prioritize_decisions"] and not uniform.collection["prioritize_decisions"]
    assert "episode_token_budget" not in retail.collection and "episode_token_budget" not in uniform.collection
