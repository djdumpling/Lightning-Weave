# /// script
# requires-python = "==3.12.*"
# dependencies = ["modal==1.5.5"]
# ///
"""Modal orchestration for tau-bench evaluation of the LoopTool OPD student.

Each model gets one vLLM server (four GPUs, data parallel) behind a
key-protected tunnel. Every domain runs in its own CPU container through the
pinned harness, which calls the OpenAI user simulator from there. Nothing runs
at import time.
"""

from __future__ import annotations

import ast
import collections
import contextlib
from dataclasses import dataclass
import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

# Modal copies the entrypoint to /root/modal_eval.py; config.py and harness.py sit beside it.
ENTRYPOINT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENTRYPOINT_DIR))

import harness  # noqa: E402 - after the sys.path setup above
from config import (  # noqa: E402
    DEPENDENCY_CUTOFF,
    LANE_PYTHON,
    LANES,
    MODAL_APP_NAME,
    MODAL_CHECKPOINT_VOLUME,
    MODAL_MODEL_VOLUME,
    MODAL_RESULTS_VOLUME,
    MODAL_VLLM_CACHE_VOLUME,
    MODELS,
    PROFILE,
    PROTOCOL,
    QWEN_REPORTED,
    REFERENCE,
    REMOTE_CHECKPOINT_ROOT,
    REMOTE_MODEL_ROOT,
    REMOTE_RESULTS_ROOT,
    SERVING,
    SERVING_IMAGE,
    TAU1_COMMIT,
    TAU1_REPO,
    TAU2_COMMIT,
    TAU2_REPO,
    USER_ALIASES,
    USER_API_BASE,
    USER_API_KEY_ENV,
    USER_SECRET,
    USER_TEAM_ID_ENV,
    check_agent_window,
    run_id,
    user_routes,
)

TAU1_ROOT = "/opt/tau-bench"
TAU2_ROOT = "/opt/tau2-bench"
UV_INSTALL = f"uv pip install --system --no-cache-dir --exclude-newer {DEPENDENCY_CUTOFF}"
CONFIG_FILE = str(ENTRYPOINT_DIR / "config.py")
HARNESS_FILE = str(ENTRYPOINT_DIR / "harness.py")
# vLLM's context-overflow message; litellm maps it to ContextWindowExceededError.
CONTEXT_OVERFLOW_MARKER = "maximum context length"

app = modal.App(MODAL_APP_NAME)
results_volume = modal.Volume.from_name(MODAL_RESULTS_VOLUME, create_if_missing=True)
model_volume = modal.Volume.from_name(MODAL_MODEL_VOLUME)
checkpoint_volume = modal.Volume.from_name(MODAL_CHECKPOINT_VOLUME)
vllm_cache_volume = modal.Volume.from_name(MODAL_VLLM_CACHE_VOLUME, create_if_missing=True)
user_secret = modal.Secret.from_name(USER_SECRET, required_keys=[USER_API_KEY_ENV])
# Every container imports config under the launcher's profile, so all agree on the protocol.
profile_secret = modal.Secret.from_dict({"TAU_PROFILE": PROFILE})
# A self-hosted user simulator (None: API users through Prime).
USER = PROTOCOL.user_server


def verify_lane_image() -> None:
    """Fail the image build instead of silently scoring a mismatched harness."""

    for root, commit in ((TAU1_ROOT, TAU1_COMMIT), (TAU2_ROOT, TAU2_COMMIT)):
        head = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        if head != commit:
            raise RuntimeError(f"{root} is at {head}, expected {commit}")
    import tau2
    import tau_bench

    for module, root in ((tau_bench, TAU1_ROOT), (tau2, TAU2_ROOT)):
        if not os.path.realpath(module.__file__).startswith(root):
            raise RuntimeError(f"{module.__name__} is not the pinned checkout: {module.__file__}")
    # The task sets Qwen's denominators imply (50/115 and 50/114/114).
    for domain in PROTOCOL.domains:
        count = len(harness.load_tasks(domain.suite, domain.domain, PROTOCOL))
        if count != domain.tasks:
            raise RuntimeError(f"{domain.name} has {count} tasks, expected {domain.tasks}")
        print(f"verified {domain.name}: {count} tasks", flush=True)


serving_image = (
    modal.Image.from_registry(SERVING_IMAGE)
    .entrypoint([])
    # The official vLLM image exposes /usr/bin/python3 but no `python`, which
    # Modal needs to detect the image's Python version.
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python")
    .env({"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1"})
    .add_local_file(CONFIG_FILE, "/root/config.py", copy=True)
    .add_local_file(HARNESS_FILE, "/root/harness.py", copy=True)
)

lane_image = (
    modal.Image.debian_slim(python_version=LANE_PYTHON)
    .apt_install("git")
    .run_commands(
        "pip install --no-cache-dir 'uv>=0.11.10'",
        f"git clone {TAU1_REPO} {TAU1_ROOT} && git -C {TAU1_ROOT} checkout {TAU1_COMMIT}",
        f"git clone {TAU2_REPO} {TAU2_ROOT} && git -C {TAU2_ROOT} checkout {TAU2_COMMIT}",
        # Editable, as both READMEs install them; tau2-bench reads tasks from its checkout.
        f"{UV_INSTALL} -e {TAU1_ROOT} -e {TAU2_ROOT}",
        "pip freeze > /opt/lane-requirements.txt",
    )
    .env({"PYTHONUNBUFFERED": "1", "TAU2_DATA_DIR": f"{TAU2_ROOT}/data"})
    .add_local_file(CONFIG_FILE, "/root/config.py", copy=True)
    .add_local_file(HARNESS_FILE, "/root/harness.py", copy=True)
    .run_function(verify_lane_image)
)


download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]<1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(CONFIG_FILE, "/root/config.py", copy=True)
    .add_local_file(HARNESS_FILE, "/root/harness.py", copy=True)
)


@app.function(
    image=download_image, secrets=[profile_secret], cpu=8, memory=16_384, timeout=7_200, volumes={REMOTE_MODEL_ROOT: model_volume}
)
def download_user_model() -> str:
    """Put the self-hosted user model in the models Volume at its pinned revision; a no-op once it is there."""

    from huggingface_hub import snapshot_download

    model_volume.reload()
    if not (Path(USER.path) / "config.json").exists():
        path = snapshot_download(USER.model, revision=USER.revision, cache_dir=REMOTE_MODEL_ROOT)
        if os.path.realpath(path) != os.path.realpath(USER.path):
            raise RuntimeError(f"downloaded to {path}, expected {USER.path}")
        model_volume.commit()
    shards = sorted(Path(USER.path).glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no weights under {USER.path}")
    return f"{USER.path} ({len(shards)} shards)"


def commit_periodically(volume: modal.Volume, stop: threading.Event, interval_s: int) -> threading.Thread:
    def loop() -> None:
        while not stop.wait(interval_s):
            try:
                volume.commit()
            except Exception as error:  # noqa: BLE001 - a missed commit is retried next interval
                print(f"volume commit failed: {error}", flush=True)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def record_name(task_id: str) -> str:
    """A readable, unique file name; telecom task ids contain brackets and pipes."""

    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)[:60]
    return f"{readable}-{hashlib.sha1(task_id.encode()).hexdigest()[:8]}.json"


def user_endpoint() -> harness.Endpoint:
    team = os.environ.get(USER_TEAM_ID_ENV)
    return harness.Endpoint(USER_API_BASE, os.environ[USER_API_KEY_ENV], {"X-Prime-Team-ID": team} if team else None)


@app.function(image=lane_image, secrets=[user_secret, profile_secret], cpu=1, memory=2_048, timeout=600)
def check_user_models(models: list[str]) -> dict:
    """Before any GPU starts: each customer model answers, calls tools (telecom's customer has tools), and is its snapshot."""

    import litellm

    tool = {
        "type": "function",
        "function": {
            "name": "check_signal",
            "description": "Check the phone's signal strength.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    checked = {}
    for model in models:
        alias = USER_ALIASES[model]
        request = {
            "messages": [{"role": "user", "content": "My phone has no signal. Check it with the tool."}],
            "tools": [tool],
            "tool_choice": "required",
            "max_tokens": 50,
        }
        response = litellm.completion(**user_endpoint().route(request, alias))
        calls = response.choices[0].message.tool_calls or []
        if not calls or calls[0].function.name != "check_signal":
            raise RuntimeError(f"{alias} at {USER_API_BASE} did not return a native tool call")
        checked[model] = {"alias": alias, "served": response.model, "snapshot": harness.check_served_model(model, alias, response.model)}
    return checked


@app.function(
    image=lane_image,
    secrets=[user_secret, profile_secret],
    cpu=LANES.cpu,
    memory=LANES.memory_mb,
    timeout=LANES.timeout_s,
    volumes={REMOTE_RESULTS_ROOT: results_volume},
)
def run_lane(job: dict) -> dict:
    """Run one domain's conversations against the tunnelled vLLM endpoint; one file per conversation."""

    domain = PROTOCOL.domain(job["domain"])
    tag, trials = job["tag"], job["trials"]
    lane_root = Path(REMOTE_RESULTS_ROOT) / job["run_id"] / tag / domain.name
    results_volume.reload()

    served_names = [PROTOCOL.served_model_name] + ([USER.served_model_name] if USER else [])
    harness.configure_litellm(served_names, LANES.request_timeout_s)
    aliases, prices = user_routes(PROTOCOL)
    router = harness.Router(
        PROTOCOL.served_model_name,
        harness.Endpoint(job["base_url"], job["api_key"]),
        harness.Endpoint(job["user_base_url"], job["user_api_key"]) if USER else user_endpoint(),
        aliases,
        prices,
        harness.Retry(LANES.call_attempts, LANES.call_backoff_s, LANES.call_max_backoff_s),
    )
    harness.install_router(domain.suite, router)
    harness.configure_logging()
    tasks = harness.load_tasks(domain.suite, domain.domain, PROTOCOL)
    if len(tasks) != domain.tasks:
        raise RuntimeError(f"{domain.name} has {len(tasks)} tasks, expected {domain.tasks}")
    if job["smoke_samples"]:
        tasks = tasks[: job["smoke_samples"]]
    seeds = harness.trial_seeds(PROTOCOL.tau2_seed, trials) if domain.suite == "tau2" else [None] * trials

    def path(task_id: str, trial: int) -> Path:
        return lane_root / f"trial{trial}" / record_name(task_id)

    def write(record: dict) -> None:
        target = path(record["task_id"], record["trial"])
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        partial.write_text(json.dumps(record) + "\n", encoding="utf-8")
        os.replace(partial, target)

    # Rerunning resumes: finished conversations are skipped, and a larger
    # --trials adds trials with the same seed sequence.
    jobs = [
        harness.Job(task_id, handle, trial, seeds[trial])
        for trial in range(trials)
        for task_id, handle in tasks
        if not path(task_id, trial).exists()
    ]
    print(
        f"[{tag}:{domain.name}] {len(tasks)} tasks x {trials} trials, {len(jobs)} to run, "
        f"concurrency={job['concurrency']}",
        flush=True,
    )
    stop = threading.Event()
    committer = commit_periodically(results_volume, stop, LANES.commit_interval_s)
    started = time.time()
    failures = harness.run_lane(
        domain,
        jobs,
        PROTOCOL,
        router,
        concurrency=job["concurrency"],
        attempts=LANES.attempts,
        retry_delay_s=LANES.retry_delay_s,
        write=write,
        log=lambda message: print(f"[{tag}:{domain.name}] {message}", flush=True),
    )
    stop.set()
    committer.join()
    failures_file = lane_root / "failures.json"
    if failures:
        failures_file.parent.mkdir(parents=True, exist_ok=True)
        failures_file.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    elif failures_file.exists():
        failures_file.unlink()
    results_volume.commit()

    records = []
    for trial in range(trials):
        for task_id, _ in tasks:
            if path(task_id, trial).exists():
                record = json.loads(path(task_id, trial).read_text(encoding="utf-8"))
                record.pop("result")
                records.append(record)
    summary = harness.summarize(records, [task_id for task_id, _ in tasks], trials)
    freeze = Path("/opt/lane-requirements.txt").read_bytes()
    return {
        "domain": domain.name,
        "seconds": round(time.time() - started, 1),
        "failures": failures,
        "lane_requirements_sha256": hashlib.sha256(freeze).hexdigest(),
        **summary,
    }


class VllmServer:
    """A local vLLM process whose log is teed to stdout and kept for startup checks."""

    def __init__(
        self,
        command: list[str],
        port: int,
        env: dict[str, str] | None = None,
        label: str = "",
        startup_timeout_s: int = SERVING.startup_timeout_s,
    ) -> None:
        self.command, self.port, self.label, self.startup_timeout_s = command, port, label, startup_timeout_s
        self.env = {**os.environ, **(env or {})}
        self.lines: collections.deque[str] = collections.deque(maxlen=20_000)
        # Held while stopping or restarting, so a restart cannot outlive the final stop.
        self.lock = threading.Lock()
        printable = [("<redacted>" if previous == "--api-key" else part) for previous, part in zip([""] + command, command)]
        print(f"{label}+", " ".join(printable), flush=True)
        self.start()

    def start(self) -> None:
        # Its own process group, so stop() also reaps the engine-core processes.
        self.process = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True, env=self.env
        )
        threading.Thread(target=self._tee, args=(self.process,), daemon=True).start()

    def _tee(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self.lines.append(line)
            print(f"{self.label}{line}", end="", flush=True)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def wait_ready(self) -> None:
        deadline = time.time() + self.startup_timeout_s
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("vLLM exited during startup:\n" + "".join(list(self.lines)[-50:]))
            try:
                with urllib.request.urlopen(self.url("/health"), timeout=5) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(5)
        raise TimeoutError(f"vLLM was not healthy within {self.startup_timeout_s}s")

    def stop(self) -> None:
        """Stop the server and every process it started; after an engine crash some may outlive it."""

        # SIGKILL always follows, for whatever a crashed engine left in the group.
        for sig, grace_s in ((signal.SIGTERM, 60), (signal.SIGKILL, 10)):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                pass
        self.process.wait()


def watch_server(server: VllmServer, tag: str, stop: threading.Event, restarts: list[float]) -> threading.Thread:
    """vLLM exits when its engine dies; restart it so the lanes' retried calls reach it again."""

    def loop() -> None:
        while not stop.wait(SERVING.watchdog_interval_s):
            if server.process.poll() is None:
                continue
            with server.lock:
                if stop.is_set():
                    return
                if len(restarts) >= SERVING.max_restarts:
                    print(f"[{tag}] vLLM exited with {server.process.returncode}; {len(restarts)} restarts used, giving up", flush=True)
                    return
                restarts.append(time.time())
                print(f"[{tag}] vLLM exited with {server.process.returncode}; restart {len(restarts)}/{SERVING.max_restarts}", flush=True)
                server.stop()
                server.start()
            try:
                server.wait_ready()
                print(f"[{tag}] vLLM is serving again", flush=True)
            except Exception as error:  # noqa: BLE001 - the next pass sees the dead process and retries
                print(f"[{tag}] restart {len(restarts)} failed: {error}", flush=True)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


@dataclass(frozen=True)
class Expected:
    """What a started server must report: its name, window, YaRN use, and sampling defaults."""

    served_model_name: str
    max_model_len: int
    generation_defaults: dict
    yarn: bool = False
    # Thinking models must return their reasoning separately from content.
    reasoning: bool = True


AGENT_EXPECTED = Expected(
    PROTOCOL.served_model_name,
    PROTOCOL.max_model_len,
    PROTOCOL.expected_generation_defaults,
    PROTOCOL.rope_scaling is not None,
)
USER_EXPECTED = (
    Expected(
        USER.served_model_name,
        USER.max_model_len,
        USER.expected_generation_defaults,
        reasoning=USER.reasoning_parser is not None,
    )
    if USER
    else None
)


def check_serving_window(lines: list[str], expected: Expected) -> None:
    """Confirm vLLM started with the expected window, with YaRN only if expected."""

    text = "".join(lines)
    if f"max_seq_len={expected.max_model_len}" not in text:
        raise RuntimeError(f"vLLM did not report max_seq_len={expected.max_model_len}")
    if expected.yarn and "yarn" not in text:
        raise RuntimeError("vLLM's startup log does not mention the yarn rope_scaling override")
    if SERVING.disable_cascade_attn and "'disable_cascade_attn': True" not in text:
        raise RuntimeError("vLLM's non-default args do not include disable_cascade_attn=True")


def check_generation_defaults(lines: list[str], expected: Expected) -> dict:
    """Confirm vLLM applies the checkpoint's sampling defaults plus the per-turn token cap."""

    for line in lines:
        if "default chat sampling params" not in line.lower():
            continue
        match = re.search(r"\{.*\}", line)
        defaults = ast.literal_eval(match.group(0)) if match else {}
        mismatched = {
            key: (defaults.get(key), value)
            for key, value in expected.generation_defaults.items()
            if defaults.get(key) != value
        }
        if mismatched:
            raise RuntimeError(f"vLLM default sampling params differ from the protocol: {mismatched}")
        return defaults
    raise RuntimeError("vLLM did not log its default chat sampling params; is --generation-config auto active?")


def request_json(url: str, body: dict | None, api_key: str | None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read())
        except ValueError:
            return error.code, {}


def filler_messages(words: int) -> list[dict]:
    return [{"role": "user", "content": "hello " * words}]


def prompt_of_length(server: VllmServer, api_key: str, target: int, served_model_name: str) -> list[dict]:
    """A chat prompt of exactly ``target`` tokens under the served chat template."""

    words = target
    for _ in range(5):
        status, body = request_json(
            server.url("/tokenize"),
            {"model": served_model_name, "messages": filler_messages(words)},
            api_key,
        )
        if status != 200:
            raise RuntimeError(f"/tokenize returned HTTP {status}: {body}")
        if body["count"] == target:
            return filler_messages(words)
        words += target - body["count"]
    raise RuntimeError(f"could not build a {target}-token prompt")


def probe_endpoint(server: VllmServer, api_key: str, expected: Expected) -> dict:
    """The serving contract the harnesses rely on: auth, clipped budgets, overflow errors, split reasoning, tool_calls."""

    status, _ = request_json(server.url("/v1/models"), None, None)
    if status != 401:
        raise RuntimeError(f"unauthenticated request returned {status}; the public tunnel must require the key")
    chat = server.url("/v1/chat/completions")
    # The harnesses send no max_tokens: a long prompt must get the rest of the
    # window as its budget instead of an error.
    headroom = 8
    name = expected.served_model_name
    near_full = prompt_of_length(server, api_key, expected.max_model_len - headroom, name)
    status, response = request_json(chat, {"model": name, "messages": near_full}, api_key)
    if status != 200:
        raise RuntimeError(f"a prompt {headroom} tokens short of the window returned HTTP {status}: {response}")
    clipped = response["usage"]["completion_tokens"]
    if not 0 < clipped <= headroom:
        raise RuntimeError(f"expected at most {headroom} completion tokens in the remaining window, got {clipped}")
    # A prompt past the window must fail in a way litellm maps to ContextWindowExceededError.
    over = filler_messages(expected.max_model_len + 64)
    status, error = request_json(chat, {"model": name, "messages": over}, api_key)
    if status != 400 or CONTEXT_OVERFLOW_MARKER not in json.dumps(error):
        raise RuntimeError(f"context overflow returned HTTP {status} without {CONTEXT_OVERFLOW_MARKER!r}: {error}")
    body = {
        "model": name,
        "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                },
            }
        ],
        "max_tokens": 2_048,
    }
    for attempt in range(1, 5):
        status, response = request_json(chat, body, api_key)
        if status != 200:
            raise RuntimeError(f"tool-call probe returned HTTP {status}")
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        if "<think>" in content or "</think>" in content:
            raise RuntimeError("reasoning leaked into content; the qwen3 reasoning parser is not active")
        if expected.reasoning and not (message.get("reasoning_content") or message.get("reasoning")):
            raise RuntimeError("no reasoning field in the response; thinking is not being split")
        calls = message.get("tool_calls") or []
        if calls:
            call = calls[0]["function"]
            json.loads(call["arguments"])
            if call["name"] != "get_weather":
                raise RuntimeError(f"probe called an unexpected tool: {call}")
            return {"attempts": attempt, "tool_call": call, "clipped_completion_tokens": clipped}
    raise RuntimeError("no native tool_calls in 4 probe attempts; check --tool-call-parser")


METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)


def scrape_metrics(server: VllmServer) -> dict[str, float]:
    with urllib.request.urlopen(server.url("/metrics"), timeout=10) as response:
        text = response.read().decode()
    values: dict[str, float] = {}
    samples: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in METRICS:
            values[name] = values.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            samples[name] = samples.get(name, 0) + 1
    # Counts add up across data-parallel engines; a usage fraction is their mean.
    if samples.get("vllm:kv_cache_usage_perc"):
        values["vllm:kv_cache_usage_perc"] /= samples["vllm:kv_cache_usage_perc"]
    return values


def monitor_metrics(server: VllmServer, tag: str, path: Path, stop: threading.Event) -> threading.Thread:
    """One compact line per interval: is the server client-bound or GPU-bound?"""

    def loop() -> None:
        previous: dict[str, float] = {}
        previous_time = time.time()
        while not stop.wait(SERVING.metrics_interval_s):
            try:
                values = scrape_metrics(server)
            except Exception as error:  # noqa: BLE001 - metrics are diagnostic only
                print(f"[metrics {tag}] scrape failed: {error}", flush=True)
                continue
            now = time.time()
            elapsed = max(now - previous_time, 1e-6)

            def rate(name: str) -> float:
                return (values.get(name, 0.0) - previous.get(name, 0.0)) / elapsed

            queries = values.get("vllm:prefix_cache_queries_total", 0.0) - previous.get("vllm:prefix_cache_queries_total", 0.0)
            hits = values.get("vllm:prefix_cache_hits_total", 0.0) - previous.get("vllm:prefix_cache_hits_total", 0.0)
            record = {
                "time": now,
                "running": values.get("vllm:num_requests_running", 0.0),
                "waiting": values.get("vllm:num_requests_waiting", 0.0),
                "kv_cache_usage": values.get("vllm:kv_cache_usage_perc", 0.0),
                "preemptions_total": values.get("vllm:num_preemptions_total", 0.0),
                "prefix_hit_rate": hits / queries if queries else None,
                "prompt_tokens_per_s": rate("vllm:prompt_tokens_total"),
                "generation_tokens_per_s": rate("vllm:generation_tokens_total"),
            }
            previous, previous_time = values, now
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            hit = "n/a" if record["prefix_hit_rate"] is None else f"{record['prefix_hit_rate']:.0%}"
            print(
                f"[metrics {tag}] running={record['running']:.0f} waiting={record['waiting']:.0f} "
                f"kv={record['kv_cache_usage']:.0%} preemptions={record['preemptions_total']:.0f} "
                f"prefix_hit={hit} gen_tok/s={record['generation_tokens_per_s']:.0f} "
                f"prompt_tok/s={record['prompt_tokens_per_s']:.0f}",
                flush=True,
            )

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def model_identity(model_path: str) -> str:
    digest = hashlib.sha256()
    root = Path(model_path)
    for name in ("config.json", "generation_config.json", "tokenizer_config.json", "model.safetensors.index.json"):
        path = root / name
        if path.exists():
            digest.update(name.encode() + b"\0" + path.read_bytes())
    for shard in sorted(root.glob("*.safetensors")):
        digest.update(f"{shard.name}:{shard.stat().st_size}".encode())
    return digest.hexdigest()


@app.function(
    image=serving_image,
    secrets=[profile_secret],
    # The agent's GPUs, then a self-hosted user's.
    gpu=SERVING.gpu_for(USER),
    cpu=8 * SERVING.gpus,
    memory=32_768 * SERVING.gpus + (65_536 if USER else 0),
    timeout=SERVING.server_timeout_s,
    volumes={
        REMOTE_MODEL_ROOT: model_volume.read_only(),
        REMOTE_CHECKPOINT_ROOT: checkpoint_volume.read_only(),
        "/root/.cache/vllm": vllm_cache_volume,
        REMOTE_RESULTS_ROOT: results_volume,
    },
)
def serve_and_evaluate(job: dict) -> dict:
    """Serve one model (and a self-hosted user, if any) for exactly as long as its domain lanes run."""

    tag, run, smoke_samples = job["tag"], job["run_id"], job["smoke_samples"]
    check_agent_window(PROTOCOL, tag)
    model_path = MODELS[tag]
    if not (Path(model_path) / "config.json").exists():
        raise FileNotFoundError(f"{tag} checkpoint not found at {model_path}")
    if USER and not (Path(USER.path) / "config.json").exists():
        raise FileNotFoundError(f"user model not found at {USER.path}; run download_user_model")
    results_volume.reload()
    root = Path(REMOTE_RESULTS_ROOT) / run / tag
    harness.write_or_check_manifest(
        root / "manifest.json",
        {
            "tag": tag,
            "model_path": model_path,
            "model_identity": model_identity(model_path),
            "protocol": PROTOCOL.resolved(),
            "smoke_samples": smoke_samples,
        },
    )
    results_volume.commit()

    def devices(first: int, count: int) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(first, first + count))}

    api_key = secrets.token_urlsafe(32)
    server = VllmServer(
        SERVING.vllm_command(model_path, api_key, PROTOCOL), SERVING.port, devices(0, SERVING.gpus) if USER else None
    )
    servers, user_server, user_key, user_probe = [server], None, None, None
    restarts: list[float] = []
    user_restarts: list[float] = []
    stop = threading.Event()
    try:
        server.wait_ready()
        check_serving_window(list(server.lines), AGENT_EXPECTED)
        defaults = check_generation_defaults(list(server.lines), AGENT_EXPECTED)
        probe = probe_endpoint(server, api_key, AGENT_EXPECTED)
        print(f"[{tag}] sampling defaults {defaults}; probe {probe}", flush=True)
        if USER:
            # Started after the agent is up, so the two servers' ports cannot race.
            user_key = secrets.token_urlsafe(32)
            user_server = VllmServer(
                SERVING.user_vllm_command(USER, user_key),
                SERVING.user_port,
                devices(SERVING.gpus, USER.tensor_parallel_size),
                label="[user-vllm] ",
                startup_timeout_s=SERVING.user_startup_timeout_s,
            )
            servers.append(user_server)
            user_server.wait_ready()
            check_serving_window(list(user_server.lines), USER_EXPECTED)
            user_defaults = check_generation_defaults(list(user_server.lines), USER_EXPECTED)
            user_probe = probe_endpoint(user_server, user_key, USER_EXPECTED)
            print(f"[{tag}:user] sampling defaults {user_defaults}; probe {user_probe}", flush=True)
        watch_server(server, tag, stop, restarts)
        if user_server:
            watch_server(user_server, f"{tag}:user", stop, user_restarts)
        with contextlib.ExitStack() as tunnels:
            base_url = f"{tunnels.enter_context(modal.forward(SERVING.port)).url}/v1"
            user_base_url = f"{tunnels.enter_context(modal.forward(SERVING.user_port)).url}/v1" if user_server else None
            invocations = root / "invocations"
            invocations.mkdir(parents=True, exist_ok=True)
            monitors = [monitor_metrics(server, tag, invocations / f"{job['invocation']}.vllm_metrics.jsonl", stop)]
            if user_server:
                user_metrics = invocations / f"{job['invocation']}.user.vllm_metrics.jsonl"
                monitors.append(monitor_metrics(user_server, f"{tag}:user", user_metrics, stop))
            committer = commit_periodically(results_volume, stop, 600)
            lanes = [
                {
                    "domain": name,
                    "tag": tag,
                    "run_id": run,
                    "base_url": base_url,
                    "api_key": api_key,
                    "user_base_url": user_base_url,
                    "user_api_key": user_key,
                    "trials": job["trials"],
                    "smoke_samples": smoke_samples,
                    "concurrency": job["concurrency"] or LANES.concurrency_for(name),
                }
                for name in job["domains"]
            ]
            calls = [run_lane.spawn(lane) for lane in lanes]
            outcomes = []
            for call in calls:
                try:
                    outcomes.append(call.get())
                except Exception as error:  # noqa: BLE001 - one failed lane must not hide the others
                    outcomes.append(error)
    finally:
        stop.set()
        for each in servers:
            with each.lock:
                each.stop()

    for monitor in monitors:
        monitor.join()
    committer.join()
    results, failures = [], []
    for lane, outcome in zip(lanes, outcomes):
        if isinstance(outcome, BaseException):
            failures.append({"domain": lane["domain"], "error": repr(outcome)})
        else:
            results.append(outcome)
            if outcome["failures"] or not outcome["complete"]:
                failures.append({"domain": lane["domain"], "missing": outcome["expected"] - outcome["conversations"]})
    gpus = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout.split("\n")
    invocation = {
        "tag": tag,
        "run_id": run,
        "domains": job["domains"],
        "trials": job["trials"],
        # Throughput only; not part of the protocol, so a resume may land on either type.
        "gpus": [name for name in gpus if name],
        "lanes": results,
        "failures": failures,
        "probe": probe,
        "server_restarts": len(restarts),
        "user_models": job["user_models"],
        "user_probe": user_probe,
        "user_server_restarts": len(user_restarts),
    }
    path = root / "invocations" / f"{job['invocation']}.json"
    path.write_text(json.dumps(invocation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_volume.commit()
    # Other invocations may have written other domains of this run, so the
    # summary is rebuilt from every record on disk.
    summary = summarize_run.remote(run, tag, smoke_samples)
    return {**summary, "invocation": {key: value for key, value in invocation.items() if key != "lanes"}}


@app.function(
    image=lane_image, secrets=[profile_secret], cpu=2, memory=8_192, timeout=1_800, volumes={REMOTE_RESULTS_ROOT: results_volume}
)
def summarize_run(run: str, tag: str, smoke_samples: int) -> dict:
    """One model's summary, rebuilt from its conversation records across every invocation that wrote them."""

    results_volume.reload()
    root = Path(REMOTE_RESULTS_ROOT) / run / tag
    lanes = []
    for domain in PROTOCOL.domains:
        lane_root = root / domain.name
        indices = [int(path.name[len("trial"):]) for path in lane_root.glob("trial*")] if lane_root.exists() else []
        if not indices:
            continue
        trials = max(indices) + 1
        tasks = harness.load_tasks(domain.suite, domain.domain, PROTOCOL)
        if smoke_samples:
            tasks = tasks[:smoke_samples]
        records = []
        for trial in range(trials):
            for task_id, _ in tasks:
                path = lane_root / f"trial{trial}" / record_name(task_id)
                if path.exists():
                    record = json.loads(path.read_text(encoding="utf-8"))
                    record.pop("result")
                    records.append(record)
        failures_file = lane_root / "failures.json"
        failures = json.loads(failures_file.read_text(encoding="utf-8")) if failures_file.exists() else []
        summary = harness.summarize(records, [task_id for task_id, _ in tasks], trials)
        lanes.append({"domain": domain.name, "trials": trials, "failures": failures, **summary})
    complete = bool(lanes) and all(lane["complete"] and not lane["failures"] for lane in lanes)
    invocations = root / "invocations"
    invocations.mkdir(parents=True, exist_ok=True)
    target = root / "summary.json"
    # Before summaries were rebuilt from records, each invocation wrote its own here; keep those.
    if target.exists() and "invocations" not in json.loads(target.read_text(encoding="utf-8")):
        target.rename(invocations / f"legacy-{int(time.time())}.json")
    summary = {
        "tag": tag,
        "run_id": run,
        "lanes": lanes,
        # Partial aggregates are still written; `complete` says whether to trust them.
        "aggregate": {"complete": complete, **harness.aggregate(lanes)},
        "invocations": sorted(path.name for path in invocations.glob("*.json")),
    }
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_volume.commit()
    return summary


@app.function(
    image=lane_image, secrets=[profile_secret], cpu=1, memory=2_048, timeout=600, volumes={REMOTE_RESULTS_ROOT: results_volume}
)
def write_comparison(run: str, comparison: dict) -> None:
    path = Path(REMOTE_RESULTS_ROOT) / run / "comparison.json"
    path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_volume.commit()


def compare(summaries: list[dict]) -> dict:
    """pass^1 per domain for every model, and each model's paired delta against base."""

    lanes = {summary["tag"]: {lane["domain"]: lane for lane in summary["lanes"]} for summary in summaries}
    # Qwen's own numbers for the first model compared (the 2507 card for Thinking-2507).
    reference = REFERENCE.get(summaries[0]["tag"], QWEN_REPORTED) if summaries else QWEN_REPORTED
    rows = {}
    for domain in PROTOCOL.domains:
        row = {"qwen_reported": reference.get(domain.name)}
        for tag, by_domain in lanes.items():
            lane = by_domain.get(domain.name)
            if lane is None:
                continue
            row[tag] = {
                "pass_hat_k": {k: 100 * v for k, v in lane["pass_hat_k"].items()},
                "complete": lane["complete"],
                "completion_tokens_per_turn_mean": lane["completion_tokens_per_turn"]["mean"],
                "completion_tokens_per_conversation_mean": lane["completion_tokens_per_conversation_mean"],
            }
            base_lane = lanes.get("base", {}).get(domain.name)
            if tag != "base" and base_lane is not None:
                row[tag]["delta_vs_base"] = harness.paired_delta(base_lane["per_task"], lane["per_task"])
        rows[domain.name] = row
    return {"domains": rows, "aggregate": {summary["tag"]: summary["aggregate"] for summary in summaries}}


def print_comparison(comparison: dict, tags: list[str]) -> None:
    header = "domain".ljust(14) + "qwen".rjust(7) + "".join(f"{tag:>9}" for tag in tags)
    others = [tag for tag in tags if tag != "base"] if "base" in tags else []
    header += "".join(f"{f'{tag}-base [95% CI]':>28}" for tag in others)
    print("\npass^1 (%), mean over trials\n" + header)
    for name, row in comparison["domains"].items():
        line = name.ljust(14) + f"{row['qwen_reported']:7.1f}"
        for tag in tags:
            value = row.get(tag, {}).get("pass_hat_k", {}).get("1")
            line += f"{value:9.1f}" if value is not None else " " * 9
        for tag in others:
            delta = row.get(tag, {}).get("delta_vs_base")
            line += f"{delta['delta']:+10.1f} [{delta['low']:+6.1f}, {delta['high']:+6.1f}]    " if delta else " " * 28
        print(line)
    for tag in tags:
        aggregate = comparison["aggregate"][tag]
        means = " ".join(f"{key}={100 * value:.1f}" for key, value in aggregate.items() if key.endswith("_mean"))
        print(f"{tag}: {means} complete={aggregate['complete']}")


@app.local_entrypoint()
def main(
    models: str = "base,opd",
    domains: str = "",
    trials: int = 0,
    smoke_samples: int = 0,
    concurrency: int = 0,
    compare_only: bool = False,
) -> None:
    """Evaluate each model on TAU1/TAU2, concurrently, four GPUs per model."""

    tags = [tag.strip() for tag in models.split(",") if tag.strip()]
    unknown = sorted(set(tags) - set(MODELS))
    if unknown:
        raise ValueError(f"unknown model tags {unknown}; choose from {sorted(MODELS)}")
    selected = [name.strip() for name in domains.split(",") if name.strip()] or [d.name for d in PROTOCOL.domains]
    for name in selected:
        PROTOCOL.domain(name)
    for tag in tags:
        check_agent_window(PROTOCOL, tag)
    trials = trials or (1 if smoke_samples else LANES.default_trials)
    run = run_id(smoke_samples)
    if compare_only:
        summaries = [summarize_run.remote(run, tag, smoke_samples) for tag in tags]
    else:
        print(f"run {run}: models={tags} domains={selected} trials={trials} smoke_samples={smoke_samples}", flush=True)
        if USER:
            print(f"user simulator {USER.model}@{USER.revision[:8]}, self-hosted: {download_user_model.remote()}", flush=True)
            users = {USER.model: {"self_hosted": True, "revision": USER.revision}}
        else:
            users = check_user_models.remote(sorted({PROTOCOL.domain(name).user_model for name in selected}))
            print(f"user simulators via {USER_API_BASE}: {users}", flush=True)
        invocation = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + "+".join(sorted(selected))
        calls = [
            serve_and_evaluate.spawn(
                {
                    "tag": tag,
                    "run_id": run,
                    "domains": selected,
                    "trials": trials,
                    "smoke_samples": smoke_samples,
                    "concurrency": concurrency,
                    "user_models": users,
                    "invocation": invocation,
                }
            )
            for tag in tags
        ]
        summaries = [call.get() for call in calls]
        print(json.dumps([summary["invocation"] for summary in summaries], indent=2, sort_keys=True))
    comparison = compare(summaries)
    write_comparison.remote(run, comparison)
    print_comparison(comparison, tags)
