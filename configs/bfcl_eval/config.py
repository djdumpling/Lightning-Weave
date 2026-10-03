"""Immutable protocol constants for BFCL evaluation on Modal.

The benchmark code is consumed, not reimplemented: agentic-eval supplies the
category list, generation budget, lane planner, and leaderboard aggregators;
NeMo-Skills supplies the BFCL driver; gorilla supplies ``bfcl_eval``, the test
data, and the official checker. Only serving and scheduling are Modal-specific.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json


AGENTIC_EVAL_COMMIT = "36a183a054a5dceb42123feaf3df76cf2abed940"
# The repository is private, so its BFCL files are copied from a local checkout
# (verified clean at the pin before every launch) instead of cloned in the image.
AGENTIC_EVAL_BFCL_SUBDIR = "benchmarks/bfcl"
NEMO_SKILLS_REPO = "https://github.com/NVIDIA-NeMo/Skills.git"
NEMO_SKILLS_COMMIT = "8979a15fb460e804761bd3f71ebf831368efa9fe"
# The gorilla commit baked into NeMo-Skills' Dockerfile.nemo-skills at the pin.
GORILLA_REPO = "https://github.com/ShishirPatil/gorilla.git"
GORILLA_COMMIT = "86d0374d0db52623c5092a73f82c22b87b7e9a25"
# Unpinned Python dependencies resolve as of the day after the NeMo-Skills pin
# (committed 2026-06-01), matching what that code was tested against; e.g. mcp
# 2.0 (2026-07-28) removed an API the pinned driver imports.
DEPENDENCY_CUTOFF = "2026-06-02T00:00:00Z"
# Installed as in Dockerfile.nemo-skills. BFCL v3 has no web_search categories;
# v4's two need a search backend that is not set up here (DuckDuckGo rate-limits
# keyless ddgs to about one query per 15 s per IP), so they are never run.
DDGS_VERSION = "9.16.0"
LANE_PYTHON = "3.10"

# Same digest-pinned official vLLM 0.11.0 image that generated the frozen
# student's LoopTool rollouts, so eval renders the chat template identically.
SERVING_IMAGE = "vllm/vllm-openai@sha256:d8d39b59e909d2378ac4feeb191f7e7b6f1342477dc66b7c47cec89e9985ad8a"

MODAL_APP_NAME = "lightning-weave-bfcl-eval"
MODAL_RESULTS_VOLUME = "lightning-weave-bfcl-eval"
MODAL_MODEL_VOLUME = "lightning-weave-hf-models"
MODAL_CHECKPOINT_VOLUME = "lightning-weave-checkpoints"
MODAL_VLLM_CACHE_VOLUME = "vllm-cache"
REMOTE_RESULTS_ROOT = "/results"
REMOTE_MODEL_ROOT = "/models"
REMOTE_CHECKPOINT_ROOT = "/checkpoints"

BASE_MODEL = "Qwen/Qwen3-4B"
BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

# Every model is served under one name so NeMo-Skills requests are identical.
MODELS = {
    "base": f"{REMOTE_MODEL_ROOT}/models--Qwen--Qwen3-4B/snapshots/{BASE_REVISION}",
    "opd": f"{REMOTE_CHECKPOINT_ROOT}/looptool-offline-dopd-qwen3-4b-v1/hf",
}
THINKING_2507_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
# Agent-efficiency students are addressed as ``ae.<arm>.<variant>`` (see configs/agent_eff).
AGENT_EFF_PREFIX = "ae."
AGENT_EFF_ROOT = f"{REMOTE_CHECKPOINT_ROOT}/agent-eff"
AGENT_EFF_DEFAULT_SEED = 1234
CONCISE_SYSTEM_PROMPT = (
    "Reason only as much as the task requires. Once you know which tool calls or answer are needed, "
    "stop thinking and respond."
)
# Inference-time variants any agent-efficiency student takes as a last tag part, e.g. ``ae.joint.acc-legacy.concise``.
AGENT_EFF_DECODING = {"concise": {"system_prompt": CONCISE_SYSTEM_PROMPT}}


@dataclass(frozen=True)
class ModelSpec:
    """Per-model serving differences, recorded in each results manifest."""

    path: str
    # False renders the Qwen3 chat template with enable_thinking=false (the empty-think baseline).
    thinking: bool = True
    # Static YaRN over 32,768 original positions; not for checkpoints that are natively long-context.
    yarn: bool = True
    # None keeps Protocol.reasoning_parser; Thinking-2507's template opens <think> in the prompt.
    reasoning_parser: str | None = None
    # Inference-time efficiency baselines, applied by the serving proxy to every request:
    # a lower per-step generation cap, and a system instruction to keep reasoning short.
    max_completion_tokens: int | None = None
    system_prompt: str | None = None


MODEL_SPECS = {
    "base": ModelSpec(MODELS["base"]),
    "opd": ModelSpec(MODELS["opd"]),
    "base-nothink": ModelSpec(MODELS["base"], thinking=False),
    "opd-nothink": ModelSpec(MODELS["opd"], thinking=False),
    "base-cap4k": ModelSpec(MODELS["base"], max_completion_tokens=4_096),
    "opd-cap4k": ModelSpec(MODELS["opd"], max_completion_tokens=4_096),
    "base-concise": ModelSpec(MODELS["base"], system_prompt=CONCISE_SYSTEM_PROMPT),
    "opd-concise": ModelSpec(MODELS["opd"], system_prompt=CONCISE_SYSTEM_PROMPT),
    # V0 after 20 of its 50 replay rounds (80 of 200 updates), where its training loss flattens.
    "opd-r20": ModelSpec(f"{REMOTE_CHECKPOINT_ROOT}/looptool-offline-dopd-qwen3-4b-v1/hf-iter0000019"),
    "thinking2507": ModelSpec(
        f"{REMOTE_MODEL_ROOT}/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/{THINKING_2507_REVISION}",
        yarn=False,
        reasoning_parser="deepseek_r1",
    ),
}


def resolve_model(tag: str) -> ModelSpec:
    """A registered tag, or ``ae.<arm>.<variant>[.s<training seed>][.<decoding>]`` for an agent-efficiency student."""
    if tag in MODEL_SPECS:
        return MODEL_SPECS[tag]
    if tag.startswith(AGENT_EFF_PREFIX):
        parts = tag[len(AGENT_EFF_PREFIX) :].split(".")
        decoding = AGENT_EFF_DECODING[parts.pop()] if len(parts) > 2 and parts[-1] in AGENT_EFF_DECODING else {}
        seed = AGENT_EFF_DEFAULT_SEED
        if len(parts) == 3 and parts[2].startswith("s") and parts[2][1:].isdigit():
            seed = int(parts.pop()[1:])
        if len(parts) == 2 and all(parts) and "/" not in tag:
            arm, variant = parts
            return ModelSpec(f"{AGENT_EFF_ROOT}/{arm}/{variant}/seed{seed}/hf", **decoding)
    raise ValueError(
        f"unknown model tag {tag!r}; choose from {sorted(MODEL_SPECS)} or ae.<arm>.<variant>[.s<seed>][.concise]"
    )

# bfcl_eval hardcodes its result/score trees here and NeMo-Skills scores as
# this FC handler, so the directory name is fixed regardless of the model.
GORILLA_ROOT = "/opt/gorilla"
BFCL_PROJECT_ROOT = f"{GORILLA_ROOT}/berkeley-function-call-leaderboard"
SCORE_MODEL_DIR = "o4-mini-2025-04-16-FC"
MEMORY_PREFIX = "memory_"
WEB_PREFIX = "web_search_"
# agentic-eval's GEN_BUDGET, checked against lanes.py at plan time. Its v3 value
# assumes a 131,072-token serving window.
RECIPE_GEN_BUDGET = {"v3": 131_072, "v4": 8_192}
# NeMo-Skills' is_context_window_exceeded_error() substrings at the pin that vLLM's
# overflow errors can match; a match soft-fails the task as _ran_out_of_context_.
CONTEXT_OVERFLOW_MARKERS = ("'max_completion_tokens' is too large", "reduce the length of the input messages")


@dataclass(frozen=True)
class Protocol:
    """Score-affecting settings. Changing any of these starts a new results tree."""

    bfcl_version: str = "v3"
    # The Qwen3 report's BFCL setting (Section 4.6) and the mentor's runs: a
    # 32,768-token response within a 65,536-token total. agentic-eval's v3
    # recipe allows 131,072. A request whose prompt exceeds max_model_len -
    # tokens_to_generate (32,768 tokens) is rejected by vLLM and scored as
    # _ran_out_of_context_; lanes count these.
    tokens_to_generate: int = 32_768
    # The recipe's request-level sampling. top_k=20 is not sent: NeMo-Skills'
    # OpenAI client rejects top_k, so vLLM applies it from generation_config.json.
    temperature: float = 0.6
    top_p: float = 0.95
    expected_generation_defaults: dict[str, float] = field(
        default_factory=lambda: {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
    )
    # Static YaRN over Qwen3's original 32,768 positions: factor 2.0 gives the
    # 65,536-token total, as the Qwen3-4B model card recommends for 64k. It is
    # applied to every category, as in the mentor's runs; the Qwen3 report
    # enabled it for the multi-turn categories.
    max_model_len: int = 65_536
    rope_scaling: dict[str, object] = field(
        default_factory=lambda: {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32_768}
    )
    served_model_name: str = "model"
    tool_call_parser: str = "hermes"
    reasoning_parser: str = "qwen3"
    dtype: str = "bfloat16"
    seed: int = 0
    # The model sees its reasoning from earlier steps of the current turn (between tool calls), as the Qwen3 chat
    # template intends ("interleaved thinking"). NeMo-Skills sends that reasoning back in each assistant message's
    # reasoning_content, but vLLM 0.11.0 drops the field before templating (0.11.1+ passes it through), so the usage
    # proxy moves it into the message text as <think>...</think>, which the template renders identically. The
    # template still drops reasoning from earlier user turns. False reproduces the runs made before 2026-10-01.
    interleaved_thinking: bool = True

    def validate(self) -> None:
        if self.bfcl_version not in RECIPE_GEN_BUDGET:
            raise ValueError(f"bfcl_version must be one of {sorted(RECIPE_GEN_BUDGET)}")
        if not 0 < self.tokens_to_generate < self.max_model_len:
            raise ValueError("tokens_to_generate must fit inside max_model_len")
        scaled = self.rope_scaling["factor"] * self.rope_scaling["original_max_position_embeddings"]
        if scaled != self.max_model_len:
            raise ValueError(f"YaRN covers {scaled:.0f} positions but max_model_len is {self.max_model_len}")
        if self.expected_generation_defaults["temperature"] != self.temperature:
            raise ValueError("request temperature must match the checkpoint generation config")
        if self.expected_generation_defaults["top_p"] != self.top_p:
            raise ValueError("request top_p must match the checkpoint generation config")

    def resolved(self) -> dict[str, object]:
        self.validate()
        fields = asdict(self)
        if not self.interleaved_thinking:
            del fields["interleaved_thinking"]  # the original runs' digest, from before the field existed
        return {
            **fields,
            "recipe_tokens_to_generate": RECIPE_GEN_BUDGET[self.bfcl_version],
            "agentic_eval": AGENTIC_EVAL_COMMIT,
            "nemo_skills": NEMO_SKILLS_COMMIT,
            "gorilla": GORILLA_COMMIT,
            "ddgs": DDGS_VERSION,
            "serving_image": SERVING_IMAGE,
        }

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.resolved(), sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Serving:
    """Throughput settings. These do not change scores, so they are not hashed."""

    # One vLLM server per model, data-parallel over 4 H100s (8 per run, a full
    # node's worth). Multi-turn chains are latency-bound, so the extra GPUs buy
    # room to run every task at once rather than a faster single sequence.
    gpus: int = 4
    data_parallel_size: int = 4
    # Per data-parallel rank.
    max_num_seqs: int = 256
    # Chunked prefill (on by default in vLLM V1) keeps long multi-turn prompts
    # from stalling decodes of the serial memory and multi-turn chains.
    max_num_batched_tokens: int = 16_384
    gpu_memory_utilization: float = 0.92
    port: int = 8000
    startup_timeout_s: int = 30 * 60
    metrics_interval_s: int = 60
    server_timeout_s: int = 24 * 60 * 60
    # Model servers one launch runs at once; further models queue. Each launch is its own app, so the total across
    # concurrent launches is the sum of their caps.
    max_concurrent_models: int = 24

    # Modal tries these in order: H100s, else H200s (same Hopper kernels, more KV memory).
    gpu_types: tuple[str, ...] = ("H100", "H200")

    @property
    def gpu(self) -> list[str]:
        return [f"{gpu_type}:{self.gpus}" for gpu_type in self.gpu_types]

    def validate(self) -> None:
        if self.data_parallel_size != self.gpus:
            raise ValueError("each data-parallel rank serves one GPU")

    def vllm_command(
        self,
        model_path: str,
        api_key: str,
        protocol: Protocol,
        spec: ModelSpec | None = None,
        chat_template: str | None = None,
    ) -> list[str]:
        spec = spec or ModelSpec(model_path)
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
            "--enable-auto-tool-choice",
            "--tool-call-parser", protocol.tool_call_parser,
            "--reasoning-parser", spec.reasoning_parser or protocol.reasoning_parser,
            "--uvicorn-log-level", "warning",
        ]
        if spec.yarn:
            command += ["--hf-overrides", json.dumps({"rope_scaling": protocol.rope_scaling})]
        if chat_template is not None:
            command += ["--chat-template", chat_template]
        return command


@dataclass(frozen=True)
class Lanes:
    """One CPU container per category; memory prereqs are serial inside each."""

    # agentic-eval used 16 because its backends served one sequence each. Every
    # multi-turn task runs at once: those serial chains set the run time.
    default_concurrency: int = 64
    concurrency: dict[str, int] = field(
        default_factory=lambda: {
            "multi_turn_base": 200,
            "multi_turn_long_context": 200,
            "multi_turn_miss_func": 200,
            "multi_turn_miss_param": 200,
            "live_multiple": 256,
            "live_irrelevance": 256,
            "simple_python": 256,
        }
    )
    cpu: float = 2.0
    memory_mb: int = 8_192
    timeout_s: int = 8 * 60 * 60
    commit_interval_s: int = 120
    # NeMo-Skills waits 14,400 s per request by default; a response lost in the
    # tunnel then idles the GPUs for hours. A 32,768-token generation takes
    # about 5-20 minutes under load; on timeout the OpenAI client retries.
    request_timeout_s: int = 3_600

    def concurrency_for(self, category: str) -> int:
        return self.concurrency.get(category, self.default_concurrency)


PROTOCOL = Protocol()
PROTOCOL.validate()
SERVING = Serving()
SERVING.validate()
LANES = Lanes()


def run_id(smoke_samples: int = 0, protocol: Protocol = PROTOCOL) -> str:
    prefix = f"smoke{smoke_samples}" if smoke_samples else "full"
    return f"bfcl-{protocol.bfcl_version}-{protocol.digest()}-{prefix}"


def driver_command(
    *,
    category: str,
    input_file: str,
    output_file: str,
    base_url: str,
    concurrency: int,
    smoke_samples: int = 0,
    random_seed: int = 0,
) -> list[str]:
    """The agentic-eval lane driver's NeMo-Skills invocation, verbatim apart from scheduling.

    ``random_seed`` is the decoding seed. NeMo-Skills sends it as every request's sampling ``seed`` (its default is
    0), so a decoding-seed replicate must pass it here: vLLM's engine ``--seed`` alone does not change it.
    """

    command = [
        "python", "-m", "nemo_skills.inference.eval.bfcl",
        "++eval_type=bfcl",
        "++eval_config.split=test",
        f"++server.base_url={base_url}",
        f"++server.model={PROTOCOL.served_model_name}",
        "++server.server_type=openai",
        f"++inference.temperature={PROTOCOL.temperature}",
        f"++inference.top_p={PROTOCOL.top_p}",
        "++use_client_parsing=False",
        "++skip_filled=True",
        f"++input_file={input_file}",
        f"++output_file={output_file}",
        f"++max_concurrent_requests={concurrency}",
        f"++inference.tokens_to_generate={PROTOCOL.tokens_to_generate}",
        f"++inference.timeout={LANES.request_timeout_s}",
        f"++inference.random_seed={random_seed}",
    ]
    if smoke_samples:
        if category.startswith(MEMORY_PREFIX):
            raise ValueError("smoke runs cannot truncate memory categories: max_samples keeps only prereqs")
        command += [f"++max_samples={smoke_samples}", "++eval_config.partial_eval=True"]
    return command


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return "" if content is None else str(content)


def request_identity(body: dict) -> dict:
    """What joins a chat request to its BFCL entry (see ``evaluation/bfcl_efficiency.py``).

    ``users`` (digests of every user message, in order) must be a prefix of the
    entry's user messages; single-turn entries also match ``tools_digest``.
    ``body_digest`` identifies retries of the same request.
    """
    messages, tools = body.get("messages") or [], body.get("tools") or []
    return {
        "users": [_digest(message_text(m)) for m in messages if m.get("role") == "user"],
        "tools_digest": _digest(tools),
        "messages": len(messages),
        "body_digest": _digest(body),
    }


def history_reasoning(body: dict) -> dict:
    """How much earlier reasoning the harness sends back in assistant messages (a field, or ``<think>`` text).

    Whether that reasoning reaches the model depends on its tokenizer's chat template.
    """
    messages = chars = 0
    for message in body.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        messages += 1
        chars += len(message.get("reasoning_content") or message.get("reasoning") or "")
        text = message_text(message)
        end = text.find("</think>")
        if end != -1:
            start = text.find("<think>")
            chars += end - (start + len("<think>") if -1 < start < end else 0)
    return {"history_assistant_messages": messages, "history_reasoning_chars": chars}


def inline_history_reasoning(body: dict) -> dict:
    """Move each assistant message's reasoning field into its text as ``<think>...</think>``.

    vLLM 0.11.0 drops ``reasoning_content`` and ``reasoning`` from request messages. The Qwen3 chat template reads
    reasoning from either the field or a ``<think>`` block in the content and renders both identically, keeping it
    only for messages after the last user query.
    """
    messages = []
    for message in body.get("messages") or []:
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if message.get("role") == "assistant" and reasoning:
            message = {key: value for key, value in message.items() if key not in ("reasoning_content", "reasoning")}
            content = message_text(message)
            if "</think>" not in content:
                message["content"] = f"<think>\n{reasoning}\n</think>\n\n{content}"
        messages.append(message)
    return {**body, "messages": messages}


def rewrite_request(body: dict, spec: ModelSpec) -> dict:
    """Apply an inference-time efficiency baseline (a lower cap, a system instruction) to a chat request."""
    body = dict(body)
    if spec.max_completion_tokens is not None:
        for key in ("max_completion_tokens", "max_tokens"):
            if key in body:
                body[key] = min(int(body[key]), spec.max_completion_tokens)
        body.setdefault("max_completion_tokens", spec.max_completion_tokens)
    if spec.system_prompt is not None:
        messages = [dict(message) for message in body.get("messages") or []]
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{message_text(messages[0])}\n\n{spec.system_prompt}"
        else:
            messages.insert(0, {"role": "system", "content": spec.system_prompt})
        body["messages"] = messages
    return body


def main() -> None:
    print(
        json.dumps(
            {
                "run_id": run_id(),
                "protocol": PROTOCOL.resolved(),
                "serving": asdict(SERVING),
                "lanes": asdict(LANES),
                "models": {tag: asdict(spec) for tag, spec in MODEL_SPECS.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
