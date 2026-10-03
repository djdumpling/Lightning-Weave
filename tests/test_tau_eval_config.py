import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from configs.bfcl_eval import config as BFCL
from configs.looptool_opd import config as OPD
from configs.tau_bench_eval import config as CONFIG
from configs.tau_bench_eval import harness as HARNESS

YARN_64K = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32_768}


def test_models_and_serving_image_match_the_bfcl_eval():
    assert CONFIG.BASE_MODEL == OPD.STUDENT_MODEL == BFCL.BASE_MODEL
    assert CONFIG.BASE_REVISION == OPD.STUDENT_REVISION
    assert {tag: CONFIG.MODELS[tag] for tag in BFCL.MODELS} == BFCL.MODELS
    assert CONFIG.THINKING_2507_REVISION == OPD.POST_TEACHER_REVISION
    assert CONFIG.MODELS["thinking2507"].endswith(f"models--Qwen--Qwen3-4B-Thinking-2507/snapshots/{OPD.POST_TEACHER_REVISION}")
    assert CONFIG.SERVING_IMAGE == BFCL.SERVING_IMAGE == OPD.ROLLOUT_IMAGE
    assert CONFIG.MODAL_MODEL_VOLUME == OPD.MODAL_MODEL_VOLUME
    assert CONFIG.MODAL_CHECKPOINT_VOLUME == OPD.MODAL_CHECKPOINT_VOLUME


def test_domains_are_the_five_qwen_tau_rows():
    names = [domain.name for domain in CONFIG.PROTOCOL.domains]
    assert names == ["tau1_retail", "tau1_airline", "tau2_retail", "tau2_airline", "tau2_telecom"]
    assert set(CONFIG.QWEN_REPORTED) == set(names)
    assert {d.user_model for d in CONFIG.PROTOCOL.domains if d.suite == "tau1"} == {"gpt-4o-2024-08-06"}
    assert {d.user_model for d in CONFIG.PROTOCOL.domains if d.suite == "tau2"} == {"gpt-4.1-2025-04-14"}


def test_every_user_snapshot_has_a_prime_alias_and_price():
    users = {domain.user_model for domain in CONFIG.PROTOCOL.domains}
    assert users == set(CONFIG.USER_ALIASES) == set(CONFIG.USER_PRICES)
    assert CONFIG.USER_ALIASES == {"gpt-4o-2024-08-06": "openai/gpt-4o", "gpt-4.1-2025-04-14": "openai/gpt-4.1"}
    assert CONFIG.USER_API_BASE == "https://api.pinference.ai/api/v1" and CONFIG.USER_SECRET == "prime-secret"


def test_qwen_reported_numbers_are_single_trial_counts_over_the_pinned_task_sets():
    # A one-decimal percentage of k successes out of n tasks: k = value * n / 100 to within rounding.
    for name, value in CONFIG.QWEN_REPORTED.items():
        tasks = CONFIG.PROTOCOL.domain(name).tasks
        successes = value * tasks / 100
        assert abs(successes - round(successes)) <= 0.05 * tasks / 100 + 1e-9, name


def test_protocol_is_a_32k_turn_budget_in_the_native_window():
    protocol = CONFIG.PROTOCOL
    protocol.validate()
    assert protocol.max_model_len == CONFIG.NATIVE_MAX_MODEL_LEN == 40_960
    assert protocol.rope_scaling is None
    assert protocol.max_new_tokens == 32_768
    assert protocol.expected_generation_defaults == {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "max_tokens": 32_768}
    with pytest.raises(ValueError, match="native"):
        CONFIG.check_agent_window(replace(protocol, max_model_len=65_536), "base")
    CONFIG.check_agent_window(replace(protocol, max_model_len=65_536), "thinking2507")
    CONFIG.check_agent_window(replace(protocol, max_model_len=65_536, rope_scaling=YARN_64K), "base")
    replace(protocol, max_model_len=65_536, rope_scaling=YARN_64K).validate()
    with pytest.raises(ValueError, match="YaRN"):
        replace(protocol, max_model_len=131_072, rope_scaling=YARN_64K).validate()
    with pytest.raises(ValueError, match="max_new_tokens"):
        replace(protocol, max_new_tokens=40_960).validate()
    with pytest.raises(ValueError, match="max_tokens"):
        replace(protocol, max_new_tokens=16_384).validate()
    with pytest.raises(ValueError, match="temperature"):
        replace(protocol, temperature=0.0).validate()


def test_protocol_keeps_each_harness_default():
    protocol = CONFIG.PROTOCOL
    assert (protocol.tau1_agent_strategy, protocol.tau1_task_split, protocol.tau1_max_num_steps) == ("tool-calling", "test", 30)
    assert (protocol.tau2_max_steps, protocol.tau2_max_errors, protocol.tau2_seed) == (200, 10, 300)
    assert protocol.tau2_user_temperature == 0.0 and protocol.tau2_evaluation_type == "all"


def test_run_id_tracks_protocol_but_not_throughput_or_trials():
    full, smoke = CONFIG.run_id(), CONFIG.run_id(3)
    assert full.startswith("tau-") and full.endswith("-full") and smoke.endswith("-smoke3")
    assert CONFIG.PROTOCOL.digest() in full
    assert replace(CONFIG.PROTOCOL, max_model_len=65_536, rope_scaling=YARN_64K).digest() != CONFIG.PROTOCOL.digest()
    resolved = CONFIG.PROTOCOL.resolved()
    assert resolved["tau1"] == CONFIG.TAU1_COMMIT and resolved["tau2"] == CONFIG.TAU2_COMMIT
    assert "trials" not in resolved and "concurrency" not in resolved and "max_num_seqs" not in resolved


def test_vllm_command_caps_turns_server_side_without_yarn():
    command = CONFIG.SERVING.vllm_command("/models/x", "secret", CONFIG.PROTOCOL)
    joined = " ".join(command)
    for flag in (
        "--enable-auto-tool-choice",
        "--tool-call-parser hermes",
        "--reasoning-parser qwen3",
        "--generation-config auto",
        '--override-generation-config {"max_new_tokens": 32768}',
        "--api-key secret",
        "--served-model-name model",
        "--max-model-len 40960",
        "--enable-prefix-caching",
        "--disable-cascade-attn",
    ):
        assert flag in joined
    assert "--hf-overrides" not in command
    assert "--enforce-eager" not in command and "--tensor-parallel-size" not in command
    yarn = CONFIG.SERVING.vllm_command("/m", "k", replace(CONFIG.PROTOCOL, max_model_len=65_536, rope_scaling=YARN_64K))
    assert '"rope_type": "yarn"' in yarn[yarn.index("--hf-overrides") + 1]


def test_lane_concurrency_fits_the_server():
    names = {domain.name for domain in CONFIG.PROTOCOL.domains}
    assert set(CONFIG.LANES.concurrency) <= names
    total = sum(CONFIG.LANES.concurrency_for(name) for name in names)
    assert total <= CONFIG.SERVING.max_num_seqs * CONFIG.SERVING.data_parallel_size
    assert CONFIG.SERVING.gpu == ["H100:4", "H200:4"]


def test_trial_seeds_match_tau2_run_tasks_and_do_not_depend_on_the_trial_count():
    random.seed(300)
    expected = [random.randint(0, 1_000_000) for _ in range(4)]
    assert HARNESS.trial_seeds(300, 4) == expected
    assert HARNESS.trial_seeds(300, 1) == expected[:1]


def test_pass_hat_k_is_the_tau_bench_estimator():
    values = HARNESS.pass_hat_k({"a": [True, True, False, False], "b": [True] * 4})
    assert values[1] == pytest.approx(0.75)
    assert values[2] == pytest.approx((1 / 6 + 1) / 2)
    assert values[4] == pytest.approx(0.5)
    assert set(HARNESS.pass_hat_k({"a": [True, False], "b": [True]})) == {1}


class APIError(Exception):
    pass


class BadRequestError(APIError):
    pass


class ContextWindowExceededError(BadRequestError):
    pass


class RateLimitError(APIError):
    pass


class AuthenticationError(APIError):
    pass


def test_errors_are_sorted_like_tau2_bench_v1():
    assert HARNESS.classify_error(ContextWindowExceededError()) == "context_window_exceeded"
    assert HARNESS.classify_error(BadRequestError()) == "bad_request"
    assert HARNESS.classify_error(RateLimitError()) == "infrastructure"
    assert HARNESS.classify_error(APIError()) == "infrastructure"
    assert HARNESS.classify_error(AuthenticationError()) == "configuration"
    assert HARNESS.classify_error(ValueError("AssistantMessage must have either content or tool calls")) == "agent_error"


def response(prompt=10, completion=5, finish="stop", cost=None):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
        choices=[SimpleNamespace(finish_reason=finish)],
        _hidden_params={"response_cost": cost},
    )


def make_router(attempts=3):
    return HARNESS.Router(
        "model",
        HARNESS.Endpoint("https://tunnel/v1", "key"),
        HARNESS.Endpoint("https://prime/v1", "prime-key", {"X-Prime-Team-ID": "team"}),
        CONFIG.USER_ALIASES,
        CONFIG.USER_PRICES,
        HARNESS.Retry(attempts=attempts, backoff_s=0, max_backoff_s=0),
    )


def test_router_sends_the_agent_to_vllm_and_customers_to_prime():
    seen, failures = [], iter([RateLimitError(), RateLimitError()])

    def completion(**kwargs):
        seen.append(kwargs)
        if kwargs["model"] == "openai/model":
            error = next(failures, None)
            if error:
                raise error
            return response(completion=7, finish="length")
        return response(prompt=1_000_000, completion=100_000)

    router = make_router()
    module = SimpleNamespace(completion=completion)
    router.install(module)
    router.install(module)
    router.begin()
    # tau-bench names the provider separately; tau2-bench prefixes it.
    module.completion(model="model", custom_llm_provider="openai", messages=[])
    module.completion(model="gpt-4.1-2025-04-14", messages=[], temperature=0.0)
    module.completion(model="gpt-4o-2024-08-06", custom_llm_provider="openai", messages=[])
    agent, user = router.calls()
    assert [call["model"] for call in seen] == ["openai/model"] * 3 + ["openai/openai/gpt-4.1", "openai/openai/gpt-4o"]
    assert not any("custom_llm_provider" in call for call in seen)
    assert all(call["api_base"] == "https://tunnel/v1" and call["api_key"] == "key" for call in seen[:3])
    assert all("extra_headers" not in call for call in seen[:3])
    assert all(call["api_base"] == "https://prime/v1" and call["api_key"] == "prime-key" for call in seen[3:])
    assert all(call["extra_headers"] == {"X-Prime-Team-ID": "team"} for call in seen[3:])
    assert seen[3]["temperature"] == 0.0
    assert agent == [{"prompt_tokens": 10, "completion_tokens": 7, "finish_reason": "length", "seconds": agent[0]["seconds"]}]
    # 1M input + 0.1M output tokens at Prime's gpt-4.1 ($2/$8) and gpt-4o ($2.5/$10) prices.
    assert [entry["cost"] for entry in user] == [pytest.approx(2.8), pytest.approx(3.5)]


def test_router_refuses_unrouted_models_and_stops_on_bad_keys():
    router = make_router()

    def unauthorized(**kwargs):
        raise AuthenticationError("bad key")

    with pytest.raises(HARNESS.ConfigurationError, match="bad key"):
        router.wrap(unauthorized)(model="gpt-4.1-2025-04-14", messages=[])
    with pytest.raises(HARNESS.ConfigurationError, match="no endpoint"):
        router.wrap(lambda **kwargs: response())(model="gpt-4o-mini", messages=[])


def test_served_model_must_be_the_snapshot_or_its_alias():
    assert HARNESS.check_served_model("gpt-4.1-2025-04-14", "openai/gpt-4.1", "gpt-4.1-2025-04-14") == "verified"
    assert HARNESS.check_served_model("gpt-4o-2024-08-06", "openai/gpt-4o", "openai/gpt-4o") == "alias"
    with pytest.raises(HARNESS.ConfigurationError):
        HARNESS.check_served_model("gpt-4o-2024-08-06", "openai/gpt-4o", "gpt-4o-2024-11-20")


def test_lane_scores_model_failures_and_never_scores_infrastructure(monkeypatch):
    outcomes = {
        "ok": lambda: {"reward": 1.0, "termination": "user_stop", "result": {}},
        "empty": lambda: (_ for _ in ()).throw(ValueError("AssistantMessage must have either content or tool calls")),
        "overflow": lambda: (_ for _ in ()).throw(ContextWindowExceededError("maximum context length")),
        "down": lambda: (_ for _ in ()).throw(RateLimitError("still down")),
    }
    monkeypatch.setattr(HARNESS, "run_tau2", lambda domain, task, trial, seed, user_model, protocol: outcomes[task]())
    domain = CONFIG.PROTOCOL.domain("tau2_airline")
    router = make_router(attempts=1)
    jobs = [HARNESS.Job(name, name, 0, 1) for name in outcomes]
    written = []
    failures = HARNESS.run_lane(
        domain, jobs, CONFIG.PROTOCOL, router, concurrency=4, attempts=2, retry_delay_s=0, write=written.append, log=lambda _: None
    )
    by_task = {record["task_id"]: record for record in written}
    assert set(by_task) == {"ok", "empty", "overflow"}
    assert by_task["ok"]["success"] and by_task["ok"]["termination"] == "user_stop"
    assert by_task["empty"]["reward"] == 0.0 and by_task["empty"]["termination"] == "agent_error"
    assert by_task["overflow"]["termination"] == "context_window_exceeded"
    assert [(failure["task_id"], failure["kind"]) for failure in failures] == [("down", "infrastructure")]

    summary = HARNESS.summarize(written, [job.task_id for job in jobs], trials=1)
    assert not summary["complete"] and summary["conversations"] == 3 and summary["expected"] == 4
    assert summary["pass_hat_k"]["1"] == pytest.approx(1 / 3)
    assert summary["terminations"] == {"user_stop": 1, "agent_error": 1, "context_window_exceeded": 1}


def test_manifest_accepts_its_own_protocol_on_resume(tmp_path):
    manifest = {"tag": "base", "protocol": CONFIG.PROTOCOL.resolved(), "smoke_samples": 0}
    path = tmp_path / "run" / "base" / "manifest.json"
    HARNESS.write_or_check_manifest(path, manifest)
    # The protocol holds tuples (its domains), which come back from JSON as lists.
    HARNESS.write_or_check_manifest(path, {"tag": "base", "protocol": CONFIG.PROTOCOL.resolved(), "smoke_samples": 0})
    changed = replace(CONFIG.PROTOCOL, max_model_len=65_536, rope_scaling=YARN_64K).resolved()
    with pytest.raises(RuntimeError, match="different model or protocol"):
        HARNESS.write_or_check_manifest(path, {"tag": "base", "protocol": changed, "smoke_samples": 0})


def test_aggregate_and_paired_delta():
    lanes = [{"domain": name, "pass_hat_k": {"1": value}} for name, value in (("tau1_retail", 0.4), ("tau1_airline", 0.2), ("tau2_airline", 0.3))]
    aggregate = HARNESS.aggregate(lanes)
    assert aggregate["tau1_mean"] == pytest.approx(0.3) and aggregate["tau2_mean"] == pytest.approx(0.3)
    assert aggregate["overall_mean"] == pytest.approx(0.3)
    same = HARNESS.paired_delta({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0}, draws=200)
    assert (same["delta"], same["low"], same["high"]) == (0.0, 0.0, 0.0)
    better = HARNESS.paired_delta({"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 0.5}, draws=200)
    assert better["delta"] == pytest.approx(75.0) and better["low"] >= 50.0


def test_prime_protocol_keeps_the_run_id_of_the_existing_results():
    # tau-b712b7c60edc-full holds the TAU1 avg@8 and TAU2 results; resuming or comparing needs this digest.
    assert CONFIG.PROFILES["prime"].digest() == "b712b7c60edc"
    assert CONFIG.PROFILE == "prime" and CONFIG.PROTOCOL is CONFIG.PROFILES["prime"]
    assert "user_server" not in CONFIG.PROTOCOL.resolved()


def test_self_hosted_user_profile():
    protocol = CONFIG.PROFILES["user30b"]
    protocol.validate()
    user = protocol.user_server
    assert user.model == "Qwen/Qwen3-30B-A3B-Thinking-2507" and user.revision.startswith("144afc2f")
    assert [d.name for d in protocol.domains] == ["tau2_retail", "tau2_airline", "tau2_telecom"]
    assert {d.user_model for d in protocol.domains} == {user.model}
    assert protocol.tau2_user_temperature == 0.6 and protocol.reasoning_parser == "deepseek_r1"
    assert protocol.digest() != CONFIG.PROFILES["prime"].digest()
    assert protocol.resolved()["user_server"]["tensor_parallel_size"] == 4
    CONFIG.check_agent_window(protocol, "thinking2507")
    with pytest.raises(ValueError, match="native"):
        CONFIG.check_agent_window(protocol, "base")
    with pytest.raises(ValueError, match="user_model"):
        replace(protocol, domains=CONFIG.PROFILES["prime"].domains).validate()
    command = " ".join(CONFIG.SERVING.user_vllm_command(user, "k"))
    for flag in ("--port 8001", "--served-model-name user", "--tensor-parallel-size 4", "--reasoning-parser deepseek_r1",
                 "--tool-call-parser hermes", "--max-model-len 65536", '{"max_new_tokens": 32768}', "--disable-cascade-attn"):
        assert flag in command
    assert CONFIG.SERVING.gpu_for(user) == ["H100:8", "H200:8"]
    aliases, prices = CONFIG.user_routes(protocol)
    assert aliases == {user.model: "user"} and prices == {user.model: (0.0, 0.0)}
    assert CONFIG.user_routes(CONFIG.PROFILES["prime"]) == (CONFIG.USER_ALIASES, CONFIG.USER_PRICES)


def test_an_empty_user_turn_is_the_simulators_error_not_the_agents():
    user = ValueError("AssistantMessage must have either content or tool calls. Got UserMessage\ntimestamp: x")
    agent = ValueError("AssistantMessage must have either content or tool calls. Got AssistantMessage\ntimestamp: x")
    assert HARNESS.classify_error(user) == "user_error"
    assert HARNESS.classify_error(agent) == "agent_error"


def test_prime_profile_for_2507_agents_differs_from_user30b_only_in_the_user():
    prime2507, user30b = CONFIG.PROFILES["prime2507"], CONFIG.PROFILES["user30b"]
    prime2507.validate()
    assert prime2507.user_server is None and prime2507.domains == CONFIG.PROFILES["prime"].domains
    assert (prime2507.max_model_len, prime2507.reasoning_parser) == (user30b.max_model_len, user30b.reasoning_parser)
    assert prime2507.digest() not in (CONFIG.PROFILES["prime"].digest(), user30b.digest())
    CONFIG.check_agent_window(prime2507, "thinking2507")
    with pytest.raises(ValueError, match="native"):
        CONFIG.check_agent_window(prime2507, "base")


def test_existing_profiles_keep_their_run_ids():
    # Results already on the Volume: user30b (Thinking-2507 avg@5) and prime2507 (Thinking-2507 with gpt-4.1).
    assert CONFIG.PROFILES["user30b"].digest() == "e8e86892bbf3"
    assert CONFIG.PROFILES["prime2507"].digest() == "dae7d8ed6e47"


def test_instruct_235b_user_profile():
    protocol = CONFIG.PROFILES["user235b"]
    protocol.validate()
    user = protocol.user_server
    assert user.model == "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" and user.reasoning_parser is None
    # tau2's own user temperature, as with gpt-4.1: greedy suits a non-thinking user.
    assert protocol.tau2_user_temperature == 0.0
    assert user.expected_generation_defaults == {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "max_tokens": 8_192}
    command = CONFIG.SERVING.user_vllm_command(user, "k")
    assert "--reasoning-parser" not in command and "--tensor-parallel-size" in command
    assert CONFIG.SERVING.gpu_for(user) == ["H200:8"]
    assert (protocol.max_model_len, protocol.reasoning_parser) == (65_536, "deepseek_r1")
    assert protocol.digest() not in {CONFIG.PROFILES[name].digest() for name in ("prime", "user30b", "prime2507")}
