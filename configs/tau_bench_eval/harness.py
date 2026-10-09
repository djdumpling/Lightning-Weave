"""Per-conversation drivers for tau-bench and tau2-bench, run inside the lane image.

Every conversation goes through the harness's own entry point (tau-bench's
``ToolCallingAgent.solve``, tau2-bench's ``run_task``), so prompts, user
simulation, and grading are the harness's. This module only routes the agent's
litellm calls to the vLLM tunnel, retries transient API failures, records
per-call usage, and sorts failures the way tau2-bench v1.0 does: agent-side
failures score 0, infrastructure failures are retried and never scored.

Nothing here imports tau packages or litellm at module import, so the metric
helpers run anywhere, including the local entrypoint.
"""

from __future__ import annotations

import array
import base64
import collections
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import random
import sys
import threading
import time
import traceback
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Callable

# Transient: retried per call with backoff, then the conversation is restarted.
INFRASTRUCTURE_ERRORS = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "Timeout",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "BadGatewayError",
        "APIError",
        "ConnectionError",
        "TimeoutError",
    }
)
# A wrong key, model name, or endpoint: retrying cannot help, so the lane stops.
CONFIGURATION_ERRORS = frozenset({"AuthenticationError", "PermissionDeniedError", "NotFoundError"})
SUCCESS_TOLERANCE = 1e-6


class ConfigurationError(RuntimeError):
    """An API error no retry can fix; the lane stops instead of scoring it."""


def classify_error(error: BaseException) -> str:
    """tau2-bench v1.0's taxonomy: what the model caused is scored, what the infrastructure caused is not.

    Order matters: litellm's ContextWindowExceededError and BadRequestError
    subclass openai.APIError, which on its own means an unclassified API failure.
    """

    names = {cls.__name__ for cls in type(error).__mro__}
    # tau2 words every empty-turn error as "AssistantMessage must have ...",
    # then prints the message itself; an empty user turn is the simulator's fault.
    if "must have either content or tool calls. Got UserMessage" in str(error):
        return "user_error"
    if "ContextWindowExceededError" in names:
        return "context_window_exceeded"
    if names & CONFIGURATION_ERRORS:
        return "configuration"
    if names & {"BadRequestError", "UnprocessableEntityError"}:
        return "bad_request"
    if names & INFRASTRUCTURE_ERRORS:
        return "infrastructure"
    # The harness raised on the model's output, e.g. tau2-bench's "AssistantMessage
    # must have either content or tool calls" when reasoning ends without an answer.
    return "agent_error"


def check_served_model(snapshot: str, alias: str, served: str | None) -> str:
    """"verified" if the API names the protocol's snapshot, "alias" if it only echoes the alias.

    Any other name (for example a newer gpt-4o snapshot) raises.
    """

    name = (served or "").rsplit("/", 1)[-1]
    if name == snapshot:
        return "verified"
    if name == alias.rsplit("/", 1)[-1]:
        return "alias"
    raise ConfigurationError(f"{alias} served {served!r}, not {snapshot}")


def write_or_check_manifest(path: Path, manifest: dict) -> None:
    """A results tree belongs to exactly one model and protocol; never mix them."""

    # Compare in stored form: JSON turns the protocol's tuples into lists.
    manifest = json.loads(json.dumps(manifest))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"{path} was written for a different model or protocol; use a new run id")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_success(reward: float) -> bool:
    return abs(reward - 1.0) <= SUCCESS_TOLERANCE


def trial_seeds(seed: int, trials: int) -> list[int]:
    """tau2-bench's ``run_tasks`` seeds: ``random.seed(seed)`` then one ``randint`` per trial.

    The sequence is fixed by ``seed``, so trial i's seed does not depend on the trial count.
    """

    rng = random.Random(seed)
    return [rng.randint(0, 1_000_000) for _ in range(trials)]


def pass_hat_k(successes: dict[str, list[bool]]) -> dict[int, float]:
    """tau-bench's pass^k: the mean over tasks of C(c, k) / C(n, k), for every k all tasks reach."""

    if not successes:
        return {}
    max_k = min(len(trials) for trials in successes.values())
    return {
        k: sum(math.comb(sum(trials), k) / math.comb(len(trials), k) for trials in successes.values()) / len(successes)
        for k in range(1, max_k + 1)
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(records: list[dict], task_ids: list[str], trials: int) -> dict:
    """Lane metrics from its conversation records."""

    by_task: dict[str, list[bool]] = collections.defaultdict(list)
    for record in records:
        by_task[record["task_id"]].append(record["success"])
    calls = [call for record in records for call in record["agent_calls"]]
    completion = [call["completion_tokens"] for call in calls if call.get("completion_tokens") is not None]
    prompt = [call["prompt_tokens"] for call in calls if call.get("prompt_tokens") is not None]
    per_conversation = [
        sum(call.get("completion_tokens") or 0 for call in record["agent_calls"]) for record in records
    ]
    user_cost = sum(call.get("cost") or 0.0 for record in records for call in record["user_calls"])
    seconds = [record["seconds"] for record in records]
    expected = len(task_ids) * trials
    return {
        "conversations": len(records),
        "expected": expected,
        "complete": len(records) == expected and set(by_task) == set(task_ids),
        "pass_hat_k": {str(k): value for k, value in pass_hat_k(by_task).items()},
        "avg_reward": sum(record["reward"] for record in records) / len(records) if records else None,
        # Mean success per task, for the paired base-vs-OPD comparison.
        "per_task": {task: sum(trials_) / len(trials_) for task, trials_ in sorted(by_task.items())},
        "terminations": dict(collections.Counter(record["termination"] for record in records)),
        "agent_turns_mean": len(calls) / len(records) if records else None,
        "completion_tokens_per_turn": {
            "mean": sum(completion) / len(completion) if completion else None,
            "p50": percentile(completion, 0.5),
            "p90": percentile(completion, 0.9),
            "p99": percentile(completion, 0.99),
            "max": max(completion, default=None),
        },
        "completion_tokens_per_conversation_mean": (
            sum(per_conversation) / len(per_conversation) if per_conversation else None
        ),
        # Turns that ended at the token cap, usually mid-thought.
        "truncated_turns": sum(call.get("finish_reason") == "length" for call in calls),
        "prompt_tokens_max": max(prompt, default=None),
        "user_cost_usd": round(user_cost, 4),
        "seconds_mean": sum(seconds) / len(seconds) if seconds else None,
        "seconds_max": max(seconds, default=None),
    }


def aggregate(lanes: list[dict]) -> dict:
    """pass^1 per domain and the unweighted means per suite and over all five domains."""

    pass1 = {lane["domain"]: lane["pass_hat_k"]["1"] for lane in lanes if lane.get("pass_hat_k")}
    means = {}
    for suite in ("tau1", "tau2", ""):
        values = [value for name, value in pass1.items() if name.startswith(suite)]
        if values:
            means[f"{suite or 'overall'}_mean"] = sum(values) / len(values)
    return {"pass^1": pass1, **means}


def paired_delta(base: dict[str, float], other: dict[str, float], draws: int = 10_000, seed: int = 0) -> dict:
    """other - base in points, with a 95% bootstrap interval that resamples tasks (paired by task)."""

    tasks = sorted(set(base) & set(other))
    if not tasks:
        return {"tasks": 0}
    diffs = [other[task] - base[task] for task in tasks]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(draws))
    return {
        "tasks": len(tasks),
        "delta": 100 * sum(diffs) / len(diffs),
        "low": 100 * means[int(0.025 * draws)],
        "high": 100 * means[int(0.975 * draws) - 1],
    }


@dataclass
class Retry:
    attempts: int
    backoff_s: float
    max_backoff_s: float

    def delay(self, attempt: int) -> float:
        return min(self.max_backoff_s, self.backoff_s * 2 ** (attempt - 1)) * (0.5 + random.random())


@dataclass(frozen=True)
class Endpoint:
    """An OpenAI-compatible API: where calls go and how they authenticate."""

    api_base: str
    api_key: str
    headers: dict[str, str] | None = None

    def route(self, kwargs: dict, model: str) -> dict:
        # "openai/<id>": litellm's OpenAI-compatible client, sending <id> as the model name.
        routed = {key: value for key, value in kwargs.items() if key != "custom_llm_provider"}
        routed.update(model=f"openai/{model}", api_base=self.api_base, api_key=self.api_key)
        if self.headers:
            routed["extra_headers"] = {**(kwargs.get("extra_headers") or {}), **self.headers}
        return routed


class Router:
    """Wraps a harness module's ``completion``: the agent goes to vLLM, customers to the user API.

    Customer models are named by snapshot in the protocol and sent as the user
    API's alias. Any other model is refused, so no call can fall through to an
    endpoint that is not configured. Calls run in the conversation's own
    thread, so a thread-local list collects one conversation's calls.
    """

    def __init__(
        self,
        model: str,
        agent: Endpoint,
        user: Endpoint,
        user_aliases: dict[str, str],
        user_prices: dict[str, tuple[float, float]],
        retry: Retry,
    ) -> None:
        self.model, self.agent, self.user = model, agent, user
        self.agent_models = {model, f"openai/{model}"}
        self.user_aliases, self.user_prices, self.retry = user_aliases, user_prices, retry
        self.local = threading.local()
        # Collection only: called with each agent call's routed kwargs and response; returns one request record.
        self.capture: Callable[[dict, Any], dict] | None = None

    def begin(self) -> None:
        self.local.agent, self.local.user, self.local.requests = [], [], []

    def calls(self) -> tuple[list[dict], list[dict]]:
        return getattr(self.local, "agent", []), getattr(self.local, "user", [])

    def requests(self) -> list[dict]:
        return getattr(self.local, "requests", [])

    def install(self, module: Any) -> None:
        if not getattr(module.completion, "_routed", False):
            module.completion = self.wrap(module.completion)

    def wrap(self, completion: Callable) -> Callable:
        def routed(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            if model in self.agent_models:
                kwargs = self.agent.route(kwargs, self.model)
            elif model in self.user_aliases:
                kwargs = self.user.route(kwargs, self.user_aliases[model])
            else:
                raise ConfigurationError(f"no endpoint is configured for model {model!r}")
            for attempt in range(1, self.retry.attempts + 1):
                started = time.time()
                try:
                    response = completion(*args, **kwargs)
                except Exception as error:
                    kind = classify_error(error)
                    if kind == "configuration":
                        raise ConfigurationError(f"{model}: {error}") from error
                    if kind != "infrastructure" or attempt == self.retry.attempts:
                        raise
                    time.sleep(self.retry.delay(attempt))
                    continue
                self.record(model, response, time.time() - started)
                if self.capture is not None and model in self.agent_models:
                    try:
                        getattr(self.local, "requests", []).append(self.capture(kwargs, response))
                    except Exception as error:  # noqa: BLE001 - a missing request record is counted, not fatal
                        getattr(self.local, "requests", []).append({"capture_error": repr(error)[:2_000]})
                return response
            raise AssertionError("unreachable")

        routed._routed = True  # type: ignore[attr-defined]
        return routed

    def record(self, model: str, response: Any, seconds: float) -> None:
        try:
            usage = getattr(response, "usage", None)
            entry = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "finish_reason": response.choices[0].finish_reason,
                "seconds": round(seconds, 2),
            }
            if model in self.user_aliases:
                # litellm cannot price the user API's aliases, so spend comes from its price list.
                entry["model"] = getattr(response, "model", None)
                input_price, output_price = self.user_prices[model]
                entry["cost"] = (entry["prompt_tokens"] * input_price + entry["completion_tokens"] * output_price) / 1e6
            getattr(self.local, "user" if model in self.user_aliases else "agent", []).append(entry)
        except Exception:  # noqa: BLE001 - recording is observation only and must never fail a turn
            pass


def configure_litellm(served_model_names: list[str], request_timeout_s: int) -> None:
    import litellm

    litellm.request_timeout = request_timeout_s
    # tau2-bench prices every response; a zero price for the self-hosted names
    # keeps its log free of "model not mapped" errors. It does not change any request.
    litellm.register_model(
        {
            name: {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0, "litellm_provider": "openai", "mode": "chat"}
            for served in served_model_names
            for name in (served, f"openai/{served}")
        }
    )


def configure_logging() -> None:
    """tau2-bench logs every message at DEBUG; with a hundred conversations per lane, keep warnings only."""

    import sys

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def install_router(suite: str, router: Router) -> None:
    if suite == "tau1":
        import tau_bench.agents.tool_calling_agent as agent_module
        import tau_bench.envs.user as user_module

        router.install(agent_module)
        router.install(user_module)
    else:
        import tau2.utils.llm_utils as llm_utils

        router.install(llm_utils)


def load_tasks(suite: str, domain: str, protocol: Any) -> list[tuple[str, Any]]:
    """(task id, harness handle) in the harness's order."""

    if suite == "tau1":
        from tau_bench.envs import get_env

        # The human user strategy makes no API call when the env is built.
        env = get_env(domain, user_strategy="human", user_model="none", task_split=protocol.tau1_task_split, task_index=0)
        return [(str(index), index) for index in range(len(env.tasks))]
    from tau2.run import get_tasks

    return [(task.id, task) for task in get_tasks(domain)]


def run_tau1(domain: str, task_index: int, trial: int, user_model: str, protocol: Any) -> dict:
    """One conversation exactly as tau-bench's ``run.py`` runs it (``_run`` with agent_factory)."""

    from tau_bench.agents.tool_calling_agent import ToolCallingAgent
    from tau_bench.envs import get_env
    from tau_bench.types import EnvRunResult

    env = get_env(
        domain,
        user_strategy=protocol.tau1_user_strategy,
        user_model=user_model,
        task_split=protocol.tau1_task_split,
        user_provider="openai",
        task_index=task_index,
    )
    agent = ToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=protocol.served_model_name,
        provider="openai",
        temperature=protocol.temperature,
    )
    result = agent.solve(env=env, task_index=task_index, max_num_steps=protocol.tau1_max_num_steps)
    native = EnvRunResult(task_id=task_index, reward=result.reward, info=result.info, traj=result.messages, trial=trial)
    return {
        "reward": result.reward,
        # The env grades only when the user stops or the agent transfers.
        "termination": "done" if result.info.get("reward_info") else "max_steps",
        "result": native.model_dump(mode="json"),
    }


def run_tau2(domain: str, task: Any, trial: int, seed: int, user_model: str, protocol: Any) -> dict:
    """One conversation exactly as tau2-bench's ``run_tasks`` runs it."""

    from tau2.evaluator.evaluator import EvaluationType
    from tau2.run import run_task

    simulation = run_task(
        domain=domain,
        task=task,
        agent=protocol.tau2_agent,
        user=protocol.tau2_user,
        llm_agent=f"openai/{protocol.served_model_name}",
        llm_args_agent={"temperature": protocol.temperature, "top_p": protocol.top_p},
        llm_user=user_model,
        llm_args_user={"temperature": protocol.tau2_user_temperature},
        max_steps=protocol.tau2_max_steps,
        max_errors=protocol.tau2_max_errors,
        evaluation_type=EvaluationType(protocol.tau2_evaluation_type),
        seed=seed,
    )
    simulation.trial = trial
    return {
        "reward": simulation.reward_info.reward,
        "termination": simulation.termination_reason.value,
        "result": simulation.model_dump(mode="json"),
    }


@dataclass(frozen=True)
class Job:
    task_id: str
    handle: Any
    trial: int
    seed: int | None


def run_conversation(domain: Any, job: Job, protocol: Any, router: Router, play: Callable[[], dict] | None = None) -> dict:
    """One record; raises only for infrastructure and configuration errors.

    ``play`` runs the conversation and returns its outcome (``reward``, ``termination``, ``result``, and optional
    extra record ``fields``); by default it is the harness's own scored conversation.
    """

    router.begin()
    started = time.time()
    if play is None:
        if domain.suite == "tau1":
            play = lambda: run_tau1(domain.domain, job.handle, job.trial, domain.user_model, protocol)  # noqa: E731
        else:
            play = lambda: run_tau2(domain.domain, job.handle, job.trial, job.seed, domain.user_model, protocol)  # noqa: E731
    try:
        outcome = play()
        error = None
    except ConfigurationError:
        raise
    except Exception as exc:  # noqa: BLE001 - sorted below
        kind = classify_error(exc)
        if kind == "infrastructure":
            raise
        # The model's failure: a zero reward, as both harnesses' later versions score it.
        outcome = {"reward": 0.0, "termination": kind, "result": None}
        error = {"type": type(exc).__name__, "message": str(exc)[:4_000], "traceback": traceback.format_exc()[-8_000:]}
    agent_calls, user_calls = router.calls()
    record = {
        "domain": domain.name,
        "task_id": job.task_id,
        "trial": job.trial,
        "seed": job.seed,
        "reward": outcome["reward"],
        "success": is_success(outcome["reward"]),
        "termination": outcome["termination"],
        "error": error,
        "agent_calls": agent_calls,
        "user_calls": user_calls,
        "seconds": round(time.time() - started, 1),
        "result": outcome["result"],
        **outcome.get("fields", {}),
    }
    if router.capture is not None:
        record["requests"] = router.requests()
    return record


def run_lane(
    domain: Any,
    jobs: list[Job],
    protocol: Any,
    router: Router,
    *,
    concurrency: int,
    attempts: int,
    retry_delay_s: float,
    write: Callable[[dict], None],
    log: Callable[[str], None],
    play: Callable[[Job], dict] | None = None,
) -> list[dict]:
    """Run every job; returns the jobs that still failed on infrastructure after all attempts.

    ``play`` replaces the scored conversation (see ``run_conversation``); collection runs its own episodes.
    """

    stop = threading.Event()
    failures: list[dict] = []
    lock = threading.Lock()
    finished = [0]

    def one(job: Job) -> None:
        for attempt in range(1, attempts + 1):
            if stop.is_set():
                return
            try:
                record = run_conversation(domain, job, protocol, router, play=(lambda: play(job)) if play else None)
            except ConfigurationError as error:
                stop.set()
                log(f"trial {job.trial} task {job.task_id}: configuration error, stopping the lane: {error}")
                with lock:
                    failures.append({"task_id": job.task_id, "trial": job.trial, "kind": "configuration", "error": str(error)})
                return
            except Exception as error:  # noqa: BLE001 - classified infrastructure by run_conversation
                log(f"trial {job.trial} task {job.task_id}: attempt {attempt}/{attempts} failed: {type(error).__name__}: {error}")
                if attempt == attempts:
                    with lock:
                        failures.append(
                            {"task_id": job.task_id, "trial": job.trial, "kind": "infrastructure", "error": repr(error)[:4_000]}
                        )
                    return
                time.sleep(retry_delay_s * attempt)
                continue
            record["attempt"] = attempt
            write(record)
            with lock:
                finished[0] += 1
                count = finished[0]
            log(
                f"trial {job.trial} task {job.task_id}: reward={record['reward']:.0f} {record['termination']} "
                f"turns={len(record['agent_calls'])} {record['seconds']:.0f}s ({count}/{len(jobs)})"
            )
            return

    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(jobs)))) as pool:
        list(pool.map(one, jobs))
    return failures


# Fresh-state collection: Qwen3-4B students on AReaL tau2 training tasks (data_curation/areal_tau2_tasks.py).
# Nothing is graded for the benchmark; the product is every agent request, exactly as it was served.


def pack_tokens(tokens: list[int]) -> str:
    """Token ids as base64 of zlib-compressed little-endian uint32; ``unpack_tokens`` inverts it."""
    values = array.array("I", tokens)
    if sys.byteorder != "little":
        values.byteswap()
    return base64.b64encode(zlib.compress(values.tobytes(), 6)).decode("ascii")


def unpack_tokens(packed: str) -> list[int]:
    values = array.array("I")
    values.frombytes(zlib.decompress(base64.b64decode(packed)))
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def tokenizer_client(url: str, api_key: str, served_model_name: str, attempts: int = 4) -> Callable[[dict], dict]:
    """vLLM's /tokenize for a chat request: the served prompt's token ids, rendered by the server's own template path."""

    def tokenize(body: dict) -> dict:
        request = {"model": served_model_name, "messages": body["messages"], "add_generation_prompt": True}
        if body.get("tools"):
            request["tools"] = body["tools"]
        data = json.dumps(request).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=120) as response:
                    tokens = json.loads(response.read())["tokens"]
                return {"count": len(tokens), "tokens": pack_tokens(tokens)}
            except Exception:  # noqa: BLE001 - retried, then recorded as missing by the capture
                if attempt == attempts:
                    raise
                time.sleep(2 * attempt)
        raise AssertionError("unreachable")

    return tokenize


def capture_agent_call(kwargs: dict, response: Any, tokenize: Callable[[dict], dict]) -> dict:
    """One served agent request: its chat messages and tools, the response, and the server's rendering of the prompt."""

    # litellm omits null fields from the request body; the rendering does not depend on them.
    messages = [{key: value for key, value in message.items() if value is not None} for message in kwargs["messages"]]
    tools = kwargs.get("tools")
    choice = response.choices[0]
    message = choice.message
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None:
        extra = getattr(message, "provider_specific_fields", None) or {}
        reasoning = extra.get("reasoning_content") or extra.get("reasoning")
    usage = getattr(response, "usage", None)
    return {
        "messages": messages,
        "tools": tools,
        "tool_choice": kwargs.get("tool_choice"),
        "sampling": {key: kwargs.get(key) for key in ("temperature", "top_p", "seed")},
        "response": {
            "content": message.content,
            "reasoning_content": reasoning,
            "tool_calls": [
                {"id": call.id, "name": call.function.name, "arguments": call.function.arguments}
                for call in (message.tool_calls or [])
            ],
            "finish_reason": choice.finish_reason,
        },
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "server_render": tokenize({"messages": messages, "tools": tools}),
    }


class TaskDatabases:
    """Each task database, loaded once per lane; every episode mutates its own deep copy."""

    CLASSES = {"airline": "FlightDB", "retail": "RetailDB"}

    def __init__(self, root: Path) -> None:
        self.root, self.cache, self.lock = root, {}, threading.Lock()

    def fresh(self, domain: str, db_path: str) -> Any:
        from importlib import import_module

        with self.lock:
            if db_path not in self.cache:
                db_class = getattr(import_module(f"tau2.domains.{domain}.data_model"), self.CLASSES[domain])
                self.cache[db_path] = db_class.load(str(self.root / db_path))
            db = self.cache[db_path]
        return db.model_copy(deep=True)


def action_reward(raw: dict, messages: list) -> float | None:
    """tau2's action check against the task's gold actions (database-independent); a diagnostic only."""

    try:
        from tau2.data_model.tasks import Task
        from tau2.evaluator.evaluator_action import ActionEvaluator

        record = {key: value for key, value in raw.items() if key != "db_path"}
        # AReaL stores the criteria as a JSON string.
        if isinstance(record.get("evaluation_criteria"), str):
            record["evaluation_criteria"] = json.loads(record["evaluation_criteria"])
        task = Task.model_validate(record)
        return float(ActionEvaluator.calculate_reward(task=task, full_trajectory=messages).reward)
    except Exception:  # noqa: BLE001 - diagnostic only
        return None


def run_areal_episode(raw: dict, databases: TaskDatabases, seed: int, user_model: str, protocol: Any) -> dict:
    """tau2's ``run_task`` for an AReaL task: the same agent, user and orchestrator, on the task's own database."""

    from importlib import import_module

    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.tasks import Task
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.user.user_simulator import UserSimulator

    domain = raw["user_scenario"]["instructions"]["domain"]
    # The grader's criteria never reach the runtime task.
    task = Task.model_validate({key: raw[key] for key in ("id", "user_scenario", "initial_state") if key in raw})
    environment = import_module(f"tau2.domains.{domain}.environment").get_environment(
        db=databases.fresh(domain, raw["db_path"])
    )
    agent = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=f"openai/{protocol.served_model_name}",
        llm_args={"temperature": protocol.temperature, "top_p": protocol.top_p},
    )
    try:
        user_tools = environment.get_user_tools()
    except Exception:  # noqa: BLE001 - as run_task: domains without user tools raise
        user_tools = None
    user = UserSimulator(
        tools=user_tools,
        instructions=str(task.user_scenario),
        llm=user_model,
        llm_args={"temperature": protocol.tau2_user_temperature},
    )
    simulation = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=protocol.tau2_max_steps,
        max_errors=protocol.tau2_max_errors,
        seed=seed,
    ).run()
    reward = action_reward(raw, simulation.messages)
    return {
        # The action-match diagnostic, or -1 when it cannot be computed (never a benchmark score).
        "reward": -1.0 if reward is None else reward,
        "termination": simulation.termination_reason.value,
        "result": simulation.model_dump(mode="json"),
        "fields": {"db_path": raw["db_path"]},
    }


def collection_seed(task_id: str) -> int:
    return int(hashlib.sha256(f"fresh-collect:{task_id}".encode()).hexdigest()[:8], 16) % 1_000_000
