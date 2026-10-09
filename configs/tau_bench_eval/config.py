"""Immutable protocol constants for tau-bench evaluation on Modal.

The benchmarks are consumed, not reimplemented: sierra-research/tau-bench
(TAU1) and tau2-bench (TAU2) supply the tasks, environments, user simulators,
and graders at the versions that were current when Qwen published its Qwen3-4B
tau numbers (July-August 2025). Only serving, scheduling, and error bookkeeping
are Modal-specific.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os


# tau-bench's task data has not changed since 2025-01-22 (14bf0ef); this pin
# only adds `response_cost or 0`, without which a self-hosted model crashes the
# agent loop (litellm has no price for it).
TAU1_REPO = "https://github.com/sierra-research/tau-bench.git"
TAU1_COMMIT = "4754e6b406507dbcbce8e8b3855dcf80aaec18ac"
# v0.1.3. Tasks, policies, and graders are identical to v0.1.0-v0.1.2 for
# airline/retail/telecom; v0.1.3 fixes `--agent-llm-args` parsing and stops
# calling an unused gpt-4o-mini NL-assertion judge. v1.0.0 (2026-03) rewrote 75+
# airline/retail tasks, so later scores are not comparable with Qwen's.
TAU2_REPO = "https://github.com/sierra-research/tau2-bench.git"
TAU2_COMMIT = "5ba9e3e56db57c5e4114bf7f901291f09b2c5619"
# Unpinned dependencies (litellm, openai, ...) resolve as of the day after the
# later pin, as the harnesses were run then.
DEPENDENCY_CUTOFF = "2025-08-28T00:00:00Z"
LANE_PYTHON = "3.11"

# Same digest-pinned official vLLM 0.11.0 image as the BFCL eval and the
# frozen student's LoopTool rollouts.
SERVING_IMAGE = "vllm/vllm-openai@sha256:d8d39b59e909d2378ac4feeb191f7e7b6f1342477dc66b7c47cec89e9985ad8a"

MODAL_APP_NAME = "lightning-weave-tau-eval"
MODAL_RESULTS_VOLUME = "lightning-weave-tau-eval"
MODAL_MODEL_VOLUME = "lightning-weave-hf-models"
MODAL_CHECKPOINT_VOLUME = "lightning-weave-checkpoints"
MODAL_VLLM_CACHE_VOLUME = "vllm-cache"
REMOTE_RESULTS_ROOT = "/results"
REMOTE_MODEL_ROOT = "/models"
REMOTE_CHECKPOINT_ROOT = "/checkpoints"

BASE_MODEL = "Qwen/Qwen3-4B"
BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

# The LoopTool OPD run's post anchor, at the same revision.
THINKING_2507_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

# Every model is served under one name so harness requests are identical.
MODELS = {
    "base": f"{REMOTE_MODEL_ROOT}/models--Qwen--Qwen3-4B/snapshots/{BASE_REVISION}",
    "opd": f"{REMOTE_CHECKPOINT_ROOT}/looptool-offline-dopd-qwen3-4b-v1/hf",
    "thinking2507": f"{REMOTE_MODEL_ROOT}/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/{THINKING_2507_REVISION}",
}

# Agent-efficiency students (configs/agent_eff), addressed as in the BFCL eval:
# ``ae.<arm>.<variant>[.s<training seed>]``. All are trained from Qwen3-4B.
AGENT_EFF_PREFIX = "ae."
AGENT_EFF_ROOT = f"{REMOTE_CHECKPOINT_ROOT}/agent-eff"
AGENT_EFF_DEFAULT_SEED = 1234

# Qwen3-4B's config.json max_position_embeddings: 32,768 output + 8,192 prompt.
NATIVE_MAX_MODEL_LEN = 40_960
# Each agent's native window (config.json max_position_embeddings); without
# YaRN, the serving window must fit inside it.
MODEL_MAX_POSITIONS = {"base": NATIVE_MAX_MODEL_LEN, "opd": NATIVE_MAX_MODEL_LEN, "thinking2507": 262_144}


def agent_eff_path(tag: str) -> str | None:
    """The export of an ``ae.<arm>.<variant>[.s<seed>]`` tag, or None for any other tag."""

    if not tag.startswith(AGENT_EFF_PREFIX):
        return None
    parts = tag[len(AGENT_EFF_PREFIX) :].split(".")
    seed = AGENT_EFF_DEFAULT_SEED
    if len(parts) == 3 and parts[2].startswith("s") and parts[2][1:].isdigit():
        seed = int(parts.pop()[1:])
    if len(parts) != 2 or not all(parts) or "/" in tag:
        raise ValueError(f"malformed agent-efficiency tag {tag!r}; expected ae.<arm>.<variant>[.s<seed>]")
    arm, variant = parts
    return f"{AGENT_EFF_ROOT}/{arm}/{variant}/seed{seed}/hf"


def model_path(tag: str) -> str:
    path = MODELS.get(tag) or agent_eff_path(tag)
    if path is None:
        raise ValueError(f"unknown model tag {tag!r}; choose from {sorted(MODELS)} or ae.<arm>.<variant>[.s<seed>]")
    return path


def max_positions(tag: str) -> int:
    model_path(tag)
    return MODEL_MAX_POSITIONS.get(tag, NATIVE_MAX_MODEL_LEN)

# Fresh-state collection runs Qwen3-4B students on AReaL's tau2 training tasks (each with its own database),
# audited against the evaluation tasks above by data_curation/areal_tau2_tasks.py.
AREAL_REPO = "inclusionAI/AReaL-tau2-data"
AREAL_REVISION = "86971dc03da6e7c1a7933295e05b84aab8215386"
AREAL_TASKS_FILE = "tau2_rl_train.jsonl"
AREAL_TASKS_SHA256 = "f7cd4c53c279819cf7c96da3986295f94bd613c0ee078715e8ac6238ad1e5cfb"
AREAL_TASK_COUNT = 1_982

# Customers are simulated through Prime Intellect's OpenAI-compatible inference
# API, which serves OpenAI models under undated aliases at OpenAI's list prices.
# The protocol names snapshots; check_user_models confirms before any GPU
# starts that each alias answers, calls tools, and is not another snapshot.
USER_API_BASE = "https://api.pinference.ai/api/v1"
# Must hold PRIME_API_KEY; PRIME_TEAM_ID, if present, bills a team account.
USER_SECRET = "prime-secret"
USER_API_KEY_ENV = "PRIME_API_KEY"
USER_TEAM_ID_ENV = "PRIME_TEAM_ID"
USER_ALIASES = {"gpt-4o-2024-08-06": "openai/gpt-4o", "gpt-4.1-2025-04-14": "openai/gpt-4.1"}
# USD per million input and output tokens, from Prime's models API on
# 2026-09-24 (no cache discount listed). Used only to report spend.
USER_PRICES = {"gpt-4o-2024-08-06": (2.5, 10.0), "gpt-4.1-2025-04-14": (2.0, 8.0)}

# Qwen3-4B (thinking) as reported in the Qwen3-4B-Thinking-2507 model card. Each
# value is an exact multiple of 100/tasks (for example 32.0 = 16/50), so Qwen
# ran one trial per task. Reference only; not part of the protocol.
QWEN_REPORTED = {
    "tau1_retail": 33.9,
    "tau1_airline": 32.0,
    "tau2_retail": 38.6,
    "tau2_airline": 28.0,
    "tau2_telecom": 17.5,
}
# Qwen3-4B-Thinking-2507 in its own model card (gpt-4.1 users, presumably).
QWEN_REPORTED_2507 = {
    "tau1_retail": 66.1,
    "tau1_airline": 48.0,
    "tau2_retail": 53.5,
    "tau2_airline": 58.0,
    "tau2_telecom": 27.2,
}
REFERENCE = {"base": QWEN_REPORTED, "opd": QWEN_REPORTED, "thinking2507": QWEN_REPORTED_2507}


@dataclass(frozen=True)
class Domain:
    suite: str
    domain: str
    tasks: int
    user_model: str

    @property
    def name(self) -> str:
        return f"{self.suite}_{self.domain}"


@dataclass(frozen=True)
class UserServer:
    """A self-hosted user simulator, served on its own GPUs beside the agent. Score-affecting."""

    model: str
    revision: str
    served_model_name: str = "user"
    tensor_parallel_size: int = 4
    max_model_len: int = 65_536
    # Thinking users can reason at length before a one-line reply; a turn cut
    # off mid-thought has no reply, and tau2 fails the conversation.
    max_new_tokens: int = 32_768
    tool_call_parser: str = "hermes"
    # Qwen 2507 thinking templates open <think> in the prompt, so the output
    # has only </think>; vLLM 0.11's qwen3 parser would leave it all in content.
    # None for non-thinking (instruct) users.
    reasoning_parser: str | None = "deepseek_r1"
    dtype: str = "bfloat16"
    seed: int = 0

    @property
    def path(self) -> str:
        return f"{REMOTE_MODEL_ROOT}/models--{self.model.replace('/', '--')}/snapshots/{self.revision}"

    @property
    def expected_generation_defaults(self) -> dict[str, float]:
        # The model's generation_config.json plus the cap. tau2 sends its own user
        # temperature on every request; top-p/top-k come from these defaults.
        return {**USER_GENERATION_DEFAULTS[self.model], "max_tokens": self.max_new_tokens}


USER_30B = UserServer(model="Qwen/Qwen3-30B-A3B-Thinking-2507", revision="144afc2f379b542fdd4e85a1fcd5e1f79112d95d")
# Non-thinking, so it replies in one line; 8,192 tokens bounds a degenerate greedy reply.
USER_235B = UserServer(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    revision="e156cb4efae43fbee1a1ab073f946a1377e6b969",
    max_new_tokens=8_192,
    reasoning_parser=None,
)
USER_GENERATION_DEFAULTS = {
    USER_30B.model: {"temperature": 0.6, "top_k": 20, "top_p": 0.95},
    USER_235B.model: {"temperature": 0.7, "top_k": 20, "top_p": 0.8},
}
# 236 GB of FP8 weights leave too little KV cache on 4 H100s (TP=8 would break
# FP8's 128-wide blocks), so this user needs H200s. Throughput only; not hashed.
USER_GPU_TYPES = {USER_235B.model: ("H200",)}


@dataclass(frozen=True)
class Protocol:
    """Score-affecting settings. Changing any of these starts a new results tree."""

    # The five TAU rows of the Qwen model cards. Users are the harness defaults at
    # the pins (tau-bench: gpt-4o, tau2-bench: gpt-4.1), pinned to the snapshots
    # those aliases resolved to in mid-2025. Qwen's own tau2 leaderboard
    # submissions (Qwen3-Max-Thinking, v0.1.3) and tau2-bench's reference runs
    # used gpt-4.1-2025-04-14.
    domains: tuple[Domain, ...] = (
        Domain("tau1", "retail", 115, "gpt-4o-2024-08-06"),
        Domain("tau1", "airline", 50, "gpt-4o-2024-08-06"),
        Domain("tau2", "retail", 114, "gpt-4.1-2025-04-14"),
        Domain("tau2", "airline", 50, "gpt-4.1-2025-04-14"),
        Domain("tau2", "telecom", 114, "gpt-4.1-2025-04-14"),
    )
    # tau-bench `run.py` defaults. Its user simulator sends no temperature, so
    # OpenAI's default (1.0) applies.
    tau1_agent_strategy: str = "tool-calling"
    tau1_task_split: str = "test"
    tau1_user_strategy: str = "llm"
    tau1_max_num_steps: int = 30
    # tau2-bench `tau2 run` defaults at v0.1.3.
    tau2_agent: str = "llm_agent"
    tau2_user: str = "user_simulator"
    tau2_user_temperature: float = 0.0
    tau2_max_steps: int = 200
    tau2_max_errors: int = 10
    tau2_seed: int = 300
    tau2_evaluation_type: str = "all"
    # Qwen3 thinking-mode sampling (model card best practice, as in the BFCL
    # eval). Both harnesses default to temperature 0.0, which Qwen advises
    # against for thinking mode. top_k=20 comes from generation_config.json.
    temperature: float = 0.6
    top_p: float = 0.95
    # The Qwen card's 32,768-token output for all non-reasoning tasks, applied
    # server-wide; the harnesses send no max_tokens, so vLLM gives each turn
    # min(max_new_tokens, max_model_len - prompt). Agent prompts peak at about
    # 24k tokens on reference trajectories (median final prompt 6-10k).
    max_new_tokens: int = 32_768
    expected_generation_defaults: dict[str, float] = field(
        default_factory=lambda: {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "max_tokens": 32_768}
    )
    # The native window, without YaRN: the Qwen3-4B card advises against static
    # YaRN unless contexts exceed 32,768 tokens, since it can degrade shorter
    # texts. For the BFCL-style 64k window set max_model_len=65_536 and
    # rope_scaling={"rope_type": "yarn", "factor": 2.0,
    # "original_max_position_embeddings": 32_768}.
    max_model_len: int = NATIVE_MAX_MODEL_LEN
    rope_scaling: dict[str, object] | None = None
    served_model_name: str = "model"
    tool_call_parser: str = "hermes"
    reasoning_parser: str = "qwen3"
    dtype: str = "bfloat16"
    seed: int = 0
    # None: the domains' users are API models reached through Prime.
    user_server: UserServer | None = None

    def domain(self, name: str) -> Domain:
        for domain in self.domains:
            if domain.name == name:
                return domain
        raise KeyError(f"unknown domain {name}; choose from {[d.name for d in self.domains]}")

    def validate(self) -> None:
        names = [domain.name for domain in self.domains]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate domains: {names}")
        if any(domain.suite not in ("tau1", "tau2") for domain in self.domains):
            raise ValueError("every domain belongs to tau1 or tau2")
        if not 0 < self.max_new_tokens < self.max_model_len:
            raise ValueError("max_new_tokens must fit inside max_model_len")
        if self.user_server is not None:
            if {domain.user_model for domain in self.domains} != {self.user_server.model}:
                raise ValueError("with a self-hosted user, every domain's user_model must be that server's model")
            if not 0 < self.user_server.max_new_tokens < self.user_server.max_model_len:
                raise ValueError("the user server's max_new_tokens must fit inside its max_model_len")
        if self.rope_scaling is not None:
            scaled = self.rope_scaling["factor"] * self.rope_scaling["original_max_position_embeddings"]
            if scaled != self.max_model_len:
                raise ValueError(f"YaRN covers {scaled:.0f} positions but max_model_len is {self.max_model_len}")
        expected = {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_new_tokens}
        for key, value in expected.items():
            if self.expected_generation_defaults[key] != value:
                raise ValueError(f"server default {key} must match the protocol's {value}")

    def resolved(self) -> dict[str, object]:
        self.validate()
        fields = asdict(self)
        # Absent for API users, so protocols from before self-hosted users keep their digest.
        if fields["user_server"] is None:
            del fields["user_server"]
        return {
            **fields,
            "tau1": TAU1_COMMIT,
            "tau2": TAU2_COMMIT,
            "dependency_cutoff": DEPENDENCY_CUTOFF,
            "serving_image": SERVING_IMAGE,
        }

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.resolved(), sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Serving:
    """Throughput settings. These do not change scores, so they are not hashed."""

    # One vLLM server per model, data-parallel over 4 H100s (8 per run), as in
    # the BFCL eval. Conversations are latency-bound; the GPUs give KV-cache room
    # for every lane's conversations at once.
    gpus: int = 4
    data_parallel_size: int = 4
    # Per data-parallel rank.
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 16_384
    gpu_memory_utilization: float = 0.92
    port: int = 8000
    # A self-hosted user simulator's server, on the GPUs after the agent's.
    user_port: int = 8001
    startup_timeout_s: int = 30 * 60
    # A 236 GB user loads from the Volume at roughly 150-200 MB/s.
    user_startup_timeout_s: int = 90 * 60
    metrics_interval_s: int = 60
    server_timeout_s: int = 24 * 60 * 60
    # vLLM 0.11 switches to cascade attention when 8+ running requests share a
    # 256+ token prefix, as every tau lane's requests share its policy prompt.
    # In the first TAU1 run both servers died within 22 s of each other with a
    # CUDA illegal memory access (Xid 31) once a lane's batch became uniform.
    # Cascade attention only changes speed, not outputs.
    disable_cascade_attn: bool = True
    # vLLM exits when its engine dies; the watchdog restarts it (about 45 s
    # warm), well inside the ~4 minutes each agent call is retried.
    max_restarts: int = 3
    watchdog_interval_s: int = 10

    # Modal tries these in order: H100s, else H200s (same Hopper kernels, more KV memory).
    gpu_types: tuple[str, ...] = ("H100", "H200")

    @property
    def gpu(self) -> list[str]:
        return self.gpu_for(None)

    def gpu_for(self, user_server: UserServer | None) -> list[str]:
        total = self.gpus + (user_server.tensor_parallel_size if user_server else 0)
        types = USER_GPU_TYPES.get(user_server.model, self.gpu_types) if user_server else self.gpu_types
        return [f"{gpu_type}:{total}" for gpu_type in types]

    def validate(self) -> None:
        if self.data_parallel_size != self.gpus:
            raise ValueError("each data-parallel rank serves one GPU")

    def vllm_command(self, model_path: str, api_key: str, protocol: Protocol) -> list[str]:
        command = [
            "vllm", "serve", model_path,
            "--served-model-name", protocol.served_model_name,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--api-key", api_key,
            "--dtype", protocol.dtype,
            "--seed", str(protocol.seed),
            "--max-model-len", str(protocol.max_model_len),
            "--data-parallel-size", str(self.data_parallel_size),
            "--max-num-seqs", str(self.max_num_seqs),
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--enable-prefix-caching",
            "--generation-config", "auto",
            # Merged over generation_config.json; a server-wide cap that each
            # request without max_tokens gets, clipped to the remaining window.
            "--override-generation-config", json.dumps({"max_new_tokens": protocol.max_new_tokens}),
            "--enable-auto-tool-choice",
            "--tool-call-parser", protocol.tool_call_parser,
            "--reasoning-parser", protocol.reasoning_parser,
            "--uvicorn-log-level", "warning",
        ]
        if protocol.rope_scaling is not None:
            command += ["--hf-overrides", json.dumps({"rope_scaling": protocol.rope_scaling})]
        if self.disable_cascade_attn:
            command.append("--disable-cascade-attn")
        return command

    def user_vllm_command(self, user_server: UserServer, api_key: str) -> list[str]:
        command = [
            "vllm", "serve", user_server.path,
            "--served-model-name", user_server.served_model_name,
            "--host", "0.0.0.0",
            "--port", str(self.user_port),
            "--api-key", api_key,
            "--dtype", user_server.dtype,
            "--seed", str(user_server.seed),
            "--max-model-len", str(user_server.max_model_len),
            "--tensor-parallel-size", str(user_server.tensor_parallel_size),
            "--max-num-seqs", str(self.max_num_seqs),
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--enable-prefix-caching",
            "--generation-config", "auto",
            "--override-generation-config", json.dumps({"max_new_tokens": user_server.max_new_tokens}),
            "--enable-auto-tool-choice",
            "--tool-call-parser", user_server.tool_call_parser,
            "--uvicorn-log-level", "warning",
        ]
        if user_server.reasoning_parser:
            command += ["--reasoning-parser", user_server.reasoning_parser]
        if self.disable_cascade_attn:
            command.append("--disable-cascade-attn")
        return command


@dataclass(frozen=True)
class Lanes:
    """One CPU container per domain; each runs its conversations on a thread pool."""

    # tau2-bench's published reference runs use 4 trials; pass^k is reported up
    # to the trial count. Trial i does not depend on the trial count (tau2 draws
    # its seeds from one fixed sequence), so a run can be extended later.
    default_trials: int = 4
    # Concurrent conversations per lane. The OpenAI user simulator's rate limit
    # is the usual ceiling; telecom's conversations are about twice as long.
    default_concurrency: int = 64
    concurrency: dict[str, int] = field(default_factory=lambda: {"tau2_telecom": 128})
    cpu: float = 4.0
    memory_mb: int = 8_192
    timeout_s: int = 16 * 60 * 60
    commit_interval_s: int = 120
    # A 32,768-token generation takes about 5-10 minutes under load.
    request_timeout_s: int = 3_600
    # Each API call that fails on infrastructure (429, 5xx, dropped connection)
    # is retried with jittered exponential backoff, about 4 minutes in total.
    # litellm's own retries fire without delay and its global num_retries resets
    # itself after the first failure, so they are not relied on.
    call_attempts: int = 8
    call_backoff_s: float = 2.0
    call_max_backoff_s: float = 120.0
    # A conversation that still fails on infrastructure is restarted from scratch
    # this many times; after that the lane reports it as missing, not as a 0.
    attempts: int = 3
    retry_delay_s: int = 60

    def concurrency_for(self, domain: str) -> int:
        return self.concurrency.get(domain, self.default_concurrency)


PROFILES = {
    # tau-bench's and tau2-bench's own users (gpt-4o / gpt-4.1) through Prime.
    "prime": Protocol(),
    # TAU2 with Qwen3-30B-A3B-Thinking-2507 as the user, self-hosted beside the
    # agent (no API). The user runs at its own recommended sampling: tau2's
    # temperature 0.0 default suits the non-reasoning gpt-4.1, and Qwen advises
    # against greedy decoding for thinking models. The agent window and parser
    # suit the 2507 thinking models (262,144 native; <think> opened by the template).
    "user30b": Protocol(
        domains=tuple(
            Domain("tau2", domain, tasks, USER_30B.model) for domain, tasks in (("retail", 114), ("airline", 50), ("telecom", 114))
        ),
        tau2_user_temperature=0.6,
        max_model_len=65_536,
        reasoning_parser="deepseek_r1",
        user_server=USER_30B,
    ),
    # The Prime users (gpt-4o / gpt-4.1), for the 2507 thinking agents: the same
    # agent window and parser as user30b, so the two differ only in the user.
    "prime2507": Protocol(max_model_len=65_536, reasoning_parser="deepseek_r1"),
    # TAU2 with Qwen3-235B-A22B-Instruct-2507-FP8 as the user, self-hosted (no
    # API): a non-thinking instruct model at tau2's default temperature 0.0, the
    # closest open analogue of gpt-4.1 at 0.0. Same agent settings as user30b.
    "user235b": Protocol(
        domains=tuple(
            Domain("tau2", domain, tasks, USER_235B.model) for domain, tasks in (("retail", 114), ("airline", 50), ("telecom", 114))
        ),
        max_model_len=65_536,
        reasoning_parser="deepseek_r1",
        user_server=USER_235B,
    ),
    # The same 235B user for Qwen3-4B agents (base, the LoopTool student, the
    # agent-efficiency students): prime's agent settings (native 40,960 window,
    # qwen3 parser, thinking-mode sampling), so only the user differs from prime.
    "user235b-4b": Protocol(
        domains=tuple(
            Domain("tau2", domain, tasks, USER_235B.model) for domain, tasks in (("retail", 114), ("airline", 50), ("telecom", 114))
        ),
        user_server=USER_235B,
    ),
}
# Chosen by the launcher (TAU_PROFILE) and passed to every container, so remote
# imports resolve the same protocol.
PROFILE = os.environ.get("TAU_PROFILE", "prime")
if PROFILE not in PROFILES:
    raise ValueError(f"TAU_PROFILE={PROFILE!r}; choose from {sorted(PROFILES)}")
PROTOCOL = PROFILES[PROFILE]
PROTOCOL.validate()
SERVING = Serving()
SERVING.validate()
LANES = Lanes()


def check_agent_window(protocol: Protocol, tag: str) -> None:
    """Without YaRN, the serving window must fit the agent's native positions."""

    if protocol.rope_scaling is None and protocol.max_model_len > max_positions(tag):
        raise ValueError(
            f"{tag} has {max_positions(tag)} native positions; max_model_len {protocol.max_model_len} needs YaRN"
        )


def user_routes(protocol: Protocol) -> tuple[dict[str, str], dict[str, tuple[float, float]]]:
    """The user models' names at their endpoint, and their prices (a self-hosted user costs no API credits)."""

    if protocol.user_server is None:
        return USER_ALIASES, USER_PRICES
    user = protocol.user_server
    return {user.model: user.served_model_name}, {user.model: (0.0, 0.0)}


def run_id(smoke_samples: int = 0) -> str:
    prefix = f"smoke{smoke_samples}" if smoke_samples else "full"
    return f"tau-{PROTOCOL.digest()}-{prefix}"


def main() -> None:
    print(
        json.dumps(
            {
                "profile": PROFILE,
                "run_id": run_id(),
                "protocol": PROTOCOL.resolved(),
                "serving": asdict(SERVING),
                "lanes": asdict(LANES),
                "models": MODELS,
                "user_api_base": USER_API_BASE if PROTOCOL.user_server is None else "self-hosted",
                "user_aliases": user_routes(PROTOCOL)[0],
                "qwen_reported": QWEN_REPORTED,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
