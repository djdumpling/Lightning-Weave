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
    AREAL_REPO,
    AREAL_REVISION,
    AREAL_TASKS_FILE,
    AREAL_TASKS_SHA256,
    DEPENDENCY_CUTOFF,
    DeepSeekUserServer,
    Qwen35Protocol,
    Qwen35MentorProtocol,
    LANE_PYTHON,
    LANES,
    MODAL_APP_NAME,
    MODAL_CHECKPOINT_VOLUME,
    MODAL_MODEL_VOLUME,
    MODAL_RESULTS_VOLUME,
    MODAL_VLLM_CACHE_VOLUME,
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
    CORRECTED_TASKS_ROOT,
    TAU2_REPO,
    USER_ALIASES,
    USER_API_BASE,
    USER_API_KEY_ENV,
    USER_SECRET,
    USER_TEAM_ID_ENV,
    check_agent_window,
    model_path as resolve_model_path,
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
EXTERNAL_USER = isinstance(USER, DeepSeekUserServer)
MODERN_AGENT = isinstance(PROTOCOL, Qwen35Protocol)
AGENT_SERVING_IMAGE = getattr(PROTOCOL, "serving_image", SERVING_IMAGE)


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
    modal.Image.from_registry(AGENT_SERVING_IMAGE)
    .entrypoint([])
    # The official vLLM image exposes /usr/bin/python3 but no `python`, which
    # Modal needs to detect the image's Python version.
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python")
    .env({"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1"})
    .add_local_file(CONFIG_FILE, "/root/config.py", copy=True)
    .add_local_file(HARNESS_FILE, "/root/harness.py", copy=True)
    .add_local_dir(str(ENTRYPOINT_DIR / "tasks"), CORRECTED_TASKS_ROOT, copy=True)
)

# DeepSeek has its own image; each agent uses the image pinned by its protocol.
external_user_image = (
    modal.Image.from_registry(USER.serving_image)
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python")
    .env({"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "VLLM_ENGINE_READY_TIMEOUT_S": "5400"})
    .add_local_file(CONFIG_FILE, "/root/config.py", copy=True)
    .add_local_file(HARNESS_FILE, "/root/harness.py", copy=True)
    .add_local_dir(str(ENTRYPOINT_DIR / "tasks"), CORRECTED_TASKS_ROOT, copy=True)
    if EXTERNAL_USER else serving_image
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
    .add_local_dir(str(ENTRYPOINT_DIR / "tasks"), CORRECTED_TASKS_ROOT, copy=True)
    .run_function(verify_lane_image)
)


AREAL_ROOT = "/opt/areal-tau2-data"


def verify_areal_data() -> None:
    data = (Path(AREAL_ROOT) / AREAL_TASKS_FILE).read_bytes()
    if hashlib.sha256(data).hexdigest() != AREAL_TASKS_SHA256:
        raise RuntimeError(f"{AREAL_REPO} tasks differ from revision {AREAL_REVISION}")


# Fresh-state collection: the lane image plus the AReaL tau2 training tasks and their databases.
collect_lane_image = (
    lane_image.run_commands(
        f"{UV_INSTALL} 'huggingface_hub<1'",
        f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{AREAL_REPO}', "
        f"repo_type='dataset', revision='{AREAL_REVISION}', local_dir='{AREAL_ROOT}', "
        f"allow_patterns=['{AREAL_TASKS_FILE}', 'tau2_rl_database/tau2_airline*', 'tau2_rl_database/tau2_retail*'])\"",
    ).run_function(verify_areal_data)
)


download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]<1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "HF_XET_HIGH_PERFORMANCE": "1"}
         if EXTERNAL_USER else {"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(CONFIG_FILE, "/root/config.py", copy=True)
    .add_local_file(HARNESS_FILE, "/root/harness.py", copy=True)
    .add_local_dir(str(ENTRYPOINT_DIR / "tasks"), CORRECTED_TASKS_ROOT, copy=True)
)


@app.function(
    image=download_image, secrets=[profile_secret], cpu=8, memory=16_384, timeout=7_200, volumes={REMOTE_MODEL_ROOT: model_volume}
)
def download_user_model() -> str:
    """Put the self-hosted user model in the models Volume at its pinned revision; a no-op once it is there."""

    from huggingface_hub import snapshot_download

    model_volume.reload()
    started = time.monotonic()
    if EXTERNAL_USER or not (Path(USER.path) / "config.json").exists():
        path = snapshot_download(USER.model, revision=USER.revision, cache_dir=REMOTE_MODEL_ROOT)
        if os.path.realpath(path) != os.path.realpath(USER.path):
            raise RuntimeError(f"downloaded to {path}, expected {USER.path}")
        model_volume.commit()
    shards = sorted(Path(USER.path).glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no weights under {USER.path}")
    if EXTERNAL_USER:
        index = json.loads((Path(USER.path) / "model.safetensors.index.json").read_text())
        missing = [name for name in set(index["weight_map"].values()) if not (Path(USER.path) / name).is_file()]
        if missing:
            raise RuntimeError(f"incomplete pinned user snapshot: {missing}")
    size = sum(shard.stat().st_size for shard in shards)
    return f"{USER.path} ({len(shards)} shards, {size / 1e9:.3f} GB; download/verification {time.monotonic() - started:.1f}s)"



@app.function(image=download_image, secrets=[profile_secret], cpu=8, memory=16_384,
              timeout=7_200, volumes={REMOTE_MODEL_ROOT: model_volume})
def download_agent_model(tag: str) -> str:
    from huggingface_hub import snapshot_download

    model_volume.reload()
    path = Path(resolve_model_path(tag))
    if not str(path).startswith(REMOTE_MODEL_ROOT + "/models--"):
        raise ValueError("agent download is only for pinned Hugging Face baselines")
    model = path.parent.parent.name[len("models--"):].replace("--", "/")
    result = snapshot_download(model, revision=path.name, cache_dir=REMOTE_MODEL_ROOT)
    if os.path.realpath(result) != os.path.realpath(path):
        raise RuntimeError(f"wrong pinned snapshot: {result}")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    if any(not (path / name).is_file() for name in set(index["weight_map"].values())):
        raise RuntimeError("incomplete agent weights")
    model_volume.commit()
    return str(path)

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


def lane_router(job: dict, domain) -> harness.Router:
    """A lane's router, installed in the harness: the agent at its tunnel, the user at its tunnel or the user API."""

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
    router.reasoning_history = isinstance(PROTOCOL, Qwen35MentorProtocol)
    harness.install_router(domain.suite, router)
    harness.configure_logging()
    if job.get("invocation"):
        timing_path = (Path(REMOTE_RESULTS_ROOT) / job["run_id"] / job["tag"] / "invocations" /
                       f"{job['invocation']}.{domain.name}.user_latency.jsonl")
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_lock, lane_started = threading.Lock(), time.time()
        window, last_log = [], [lane_started]

        def timing_sink(entry: dict) -> None:
            entry["lane_started_at_unix_s"] = lane_started
            entry["elapsed_from_lane_start_s"] = round(entry["finished_at_unix_s"] - lane_started, 3)
            with timing_lock:
                with timing_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
                window.append(entry["seconds"])
                if time.time() - last_log[0] >= 60:
                    print(f"[user latency {job['tag']}:{domain.name}] elapsed={entry['elapsed_from_lane_start_s']:.0f}s "
                          f"completed={len(window)} p50={harness.percentile(window, .5):.2f}s "
                          f"p90={harness.percentile(window, .9):.2f}s", flush=True)
                    window.clear()
                    last_log[0] = time.time()

        router.user_timing_sink = timing_sink
    return router


def run_jobs(job: dict, domain, router: harness.Router, jobs: list, path, lane_root: Path, play=None) -> list[dict]:
    """Run the jobs, one atomically written file each (``path(record)``), committing periodically.

    Returns the infrastructure failures, which are also kept in the lane's failures.json until a rerun clears them.
    """

    def write(record: dict) -> None:
        target = path(record)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        partial.write_text(json.dumps(record) + "\n", encoding="utf-8")
        os.replace(partial, target)

    stop = threading.Event()
    committer = commit_periodically(results_volume, stop, LANES.commit_interval_s)
    failures = harness.run_lane(
        domain,
        jobs,
        PROTOCOL,
        router,
        concurrency=job["concurrency"],
        attempts=LANES.attempts,
        retry_delay_s=LANES.retry_delay_s,
        write=write,
        log=lambda message: print(f"[{job['tag']}:{domain.name}] {message}", flush=True),
        play=play,
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
    return failures


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
    router = lane_router(job, domain)
    if isinstance(PROTOCOL, Qwen35MentorProtocol):
        tokenizer_url = job["base_url"].removesuffix("/v1") + "/tokenize"
        tokenize = harness.tokenizer_client(tokenizer_url, job["api_key"], PROTOCOL.served_model_name)
        def capture_check(kwargs, response):
            if router.local.trial == 0 and router.local.task_id in {"0", "5"}:
                return harness.capture_agent_call(kwargs, response, tokenize)
            return {"sampling": {key: kwargs.get(key) for key in (
                "temperature", "top_p", "presence_penalty", "max_tokens", "extra_body", "seed")}}
        router.capture = capture_check
    tasks = harness.load_tasks(domain.suite, domain.domain, PROTOCOL)
    if len(tasks) != domain.tasks:
        raise RuntimeError(f"{domain.name} has {len(tasks)} tasks, expected {domain.tasks}")
    if job["smoke_samples"]:
        tasks = tasks[: job["smoke_samples"]]
    seeds = harness.trial_seeds(PROTOCOL.tau2_seed, trials) if domain.suite in ("tau2", "tau2c") else [None] * trials

    def path(task_id: str, trial: int) -> Path:
        return lane_root / f"trial{trial}" / record_name(task_id)

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
    started = time.time()
    failures = run_jobs(job, domain, router, jobs, lambda record: path(record["task_id"], record["trial"]), lane_root)

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


@app.function(
    image=collect_lane_image,
    secrets=[profile_secret],
    cpu=8,
    memory=65_536,
    timeout=LANES.timeout_s,
    volumes={REMOTE_RESULTS_ROOT: results_volume},
)
def collect_lane(job: dict) -> dict:
    """Run one domain's AReaL training episodes with the tunnelled agent; one record per episode, every request kept."""

    domain = PROTOCOL.domain(job["domain"])
    lane_root = Path(REMOTE_RESULTS_ROOT) / job["run_id"] / job["tag"] / domain.name
    results_volume.reload()
    router = lane_router(job, domain)
    tokenize = harness.tokenizer_client(job["tokenize_url"], job["api_key"], PROTOCOL.served_model_name)
    router.capture = lambda kwargs, response: harness.capture_agent_call(kwargs, response, tokenize)
    lines = (Path(AREAL_ROOT) / AREAL_TASKS_FILE).read_text(encoding="utf-8").splitlines()
    raw = {task["id"]: task for task in map(json.loads, lines)}
    task_ids = job["task_ids"]
    if any(task_id not in raw or raw[task_id]["user_scenario"]["instructions"]["domain"] != domain.domain for task_id in task_ids):
        raise RuntimeError(f"{domain.name}: unknown or other-domain AReaL task ids")
    databases = harness.TaskDatabases(Path(AREAL_ROOT))

    def path(task_id: str) -> Path:
        return lane_root / record_name(task_id)

    jobs = [harness.Job(task_id, raw[task_id], 0, harness.collection_seed(task_id)) for task_id in task_ids if not path(task_id).exists()]
    print(f"[{job['tag']}:{domain.name}] {len(task_ids)} episodes, {len(jobs)} to run, concurrency={job['concurrency']}", flush=True)
    started = time.time()
    failures = run_jobs(
        job, domain, router, jobs, lambda record: path(record["task_id"]), lane_root,
        play=lambda item: harness.run_areal_episode(item.handle, databases, item.seed, domain.user_model, PROTOCOL),
    )
    written = sum(path(task_id).exists() for task_id in task_ids)
    # The keys serve() reads from every lane.
    return {"domain": domain.name, "expected": len(task_ids), "conversations": written,
            "complete": written == len(task_ids), "failures": failures, "seconds": round(time.time() - started, 1)}


@app.function(
    image=collect_lane_image,
    secrets=[profile_secret],
    cpu=8,
    memory=65_536,
    timeout=LANES.timeout_s,
    volumes={REMOTE_RESULTS_ROOT: results_volume},
)
def branch_lane(job: dict) -> dict:
    """Run one domain's executed branches (data_curation/branch_outcomes.py) with the tunnelled agent and user.

    Each branch resumes its episode's recorded history with a cached response as the agent's message, then plays
    on; its record (named by branch id) holds tau2's database reward on the task's own database.
    """

    domain = PROTOCOL.domain(job["domain"])
    lane_root = Path(REMOTE_RESULTS_ROOT) / job["run_id"] / job["tag"] / domain.name
    results_volume.reload()
    router = lane_router(job, domain)
    lines = (Path(AREAL_ROOT) / AREAL_TASKS_FILE).read_text(encoding="utf-8").splitlines()
    raw = {task["id"]: task for task in map(json.loads, lines)}
    wanted = set(job["branch_ids"])
    plan = (Path(REMOTE_RESULTS_ROOT) / job["plan"]).read_text(encoding="utf-8").splitlines()
    branches = [item for item in map(json.loads, plan) if item["branch_id"] in wanted]
    if len(branches) != len(wanted) or any(item["domain"] != domain.name or item["task_id"] not in raw for item in branches):
        raise RuntimeError(f"{domain.name}: the plan does not hold these branches for this domain")
    databases = harness.TaskDatabases(Path(AREAL_ROOT))

    def path(branch_id: str) -> Path:
        return lane_root / f"{branch_id}.json"

    jobs = [harness.Job(item["branch_id"], item, 0, item["seed"]) for item in branches if not path(item["branch_id"]).exists()]
    print(f"[{job['tag']}:{domain.name}] {len(wanted)} branches, {len(jobs)} to run, concurrency={job['concurrency']}", flush=True)
    started = time.time()
    failures = run_jobs(
        job, domain, router, jobs, lambda record: path(record["task_id"]), lane_root,
        play=lambda item: harness.run_areal_branch(raw[item.handle["task_id"]], databases, item.handle, domain.user_model, PROTOCOL),
    )
    written = sum(path(branch_id).exists() for branch_id in wanted)
    return {"domain": domain.name, "expected": len(wanted), "conversations": written,
            "complete": written == len(wanted), "failures": failures, "seconds": round(time.time() - started, 1)}


@app.function(image=lane_image, cpu=2, memory=8_192, timeout=1_800, volumes={REMOTE_RESULTS_ROOT: results_volume})
def read_branch_plan(plan: str) -> dict:
    """{domain: [branch ids]} of a branch plan on the results volume, and the plan's hash."""

    import hashlib

    results_volume.reload()
    data = (Path(REMOTE_RESULTS_ROOT) / plan).read_bytes()
    by_domain: dict[str, list[str]] = collections.defaultdict(list)
    for line in data.decode("utf-8").splitlines():
        item = json.loads(line)
        by_domain[item["domain"]].append(item["branch_id"])
    return {"sha256": hashlib.sha256(data).hexdigest(), "branches": dict(by_domain)}


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
        reasoning=USER.reasoning_parser is not None and not EXTERNAL_USER,
    )
    if USER
    else None
)


def check_serving_window(lines: list[str], expected: Expected, *, legacy_log: bool = True) -> None:
    """Confirm vLLM started with the expected window, with YaRN only if expected."""

    text = "".join(lines)
    if f"max_seq_len={expected.max_model_len}" not in text:
        raise RuntimeError(f"vLLM did not report max_seq_len={expected.max_model_len}")
    if expected.yarn and "yarn" not in text:
        raise RuntimeError("vLLM's startup log does not mention the yarn rope_scaling override")
    if legacy_log and SERVING.disable_cascade_attn and "'disable_cascade_attn': True" not in text:
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



def check_modern_agent_sampling(server: VllmServer) -> dict:
    text = "".join(server.lines)
    match = re.search(r"'override_generation_config': (\{[^{}]+\})", text)
    overrides = ast.literal_eval(match.group(1)) if match else {}
    defaults = {**overrides}
    defaults["max_tokens"] = defaults.pop("max_new_tokens", None)
    if defaults != PROTOCOL.expected_generation_defaults:
        raise RuntimeError(f"modern agent sampling differs from protocol: {defaults}")
    if "'default_chat_template_kwargs': {'enable_thinking': True}" not in text:
        raise RuntimeError("Qwen3.5 thinking was not explicitly enabled")
    return defaults

def probe_current_step_history(server: VllmServer, api_key: str, model_path: str) -> dict:
    """Decode actual generation prompts, proving vLLM's field mapping and native history gate."""
    from transformers import AutoTokenizer
    import vllm

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tool = {"type": "function", "function": {"name": "get_weather", "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
    def assistant(call_id, city, reasoning):
        return {"role": "assistant", "content": None, "reasoning": reasoning,
                "tool_calls": [{"id": call_id, "type": "function", "function": {
                    "name": "get_weather", "arguments": json.dumps({"city": city})}}]}
    old, current1, current2 = "OLD_STEP_REASONING_7a43", "CURRENT_STEP_REASONING_1_7a43", "CURRENT_STEP_REASONING_2_7a43"
    messages = [
        {"role": "user", "content": "Check Paris."}, assistant("old", "Paris", old),
        {"role": "tool", "tool_call_id": "old", "content": "Paris: sunny."},
        {"role": "assistant", "content": "Paris is sunny.", "reasoning": "OLD_FINAL_REASONING_7a43"},
        {"role": "user", "content": "Now compare London and Rome."}, assistant("new1", "London", current1),
        {"role": "tool", "tool_call_id": "new1", "content": "London: rainy."},
        assistant("new2", "Rome", current2),
        {"role": "tool", "tool_call_id": "new2", "content": "Rome: sunny."},
    ]
    args = harness.agent_llm_args(PROTOCOL)
    body = {"model": PROTOCOL.served_model_name, "messages": messages, "tools": [tool], "tool_choice": "auto",
            **{k: v for k, v in args.items() if k != "extra_body"}, **args["extra_body"], "max_tokens": 8}
    evidence = []
    for next_step in (False, True):
        request = {**body, "messages": messages + ([{"role": "user", "content": "Thanks. Summarize."}] if next_step else [])}
        status, response = request_json(server.url("/v1/chat/completions"), request, api_key)
        if status != 200 or not response.get("prompt_token_ids"):
            raise RuntimeError(f"history probe lacks actual prompt IDs: HTTP {status}")
        ids = response["prompt_token_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if old in decoded or "OLD_FINAL_REASONING_7a43" in decoded:
            raise RuntimeError("the native template retained reasoning from an earlier user turn")
        if any((sentinel in decoded) != (not next_step) for sentinel in (current1, current2)):
            raise RuntimeError("the current-step reasoning boundary differs from the mentor's protocol")
        evidence.append({"after_new_user": next_step, "request": request,
                         "prompt_token_ids": harness.pack_tokens(ids), "decoded_prompt": decoded,
                         "prompt_tokens": response["usage"]["prompt_tokens"]})
    return {"policy": PROTOCOL.reasoning_history, "vllm_version": vllm.__version__,
            "wire_field": "reasoning", "native_template_field": "reasoning_content", "checks": evidence}


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


# The agent's GPUs, then a self-hosted user's; shared by every function that serves a model.
SERVER_FUNCTION = dict(
    image=serving_image,
    secrets=[profile_secret],
    gpu=SERVING.gpu_for(None if EXTERNAL_USER else USER),
    cpu=8 * SERVING.gpus,
    memory=32_768 * SERVING.gpus + (65_536 if USER and not EXTERNAL_USER else 0),
    timeout=SERVING.server_timeout_s,
    volumes={
        REMOTE_MODEL_ROOT: model_volume.read_only(),
        REMOTE_CHECKPOINT_ROOT: checkpoint_volume.read_only(),
        "/root/.cache/vllm": vllm_cache_volume,
        REMOTE_RESULTS_ROOT: results_volume,
    },
)


def serve(job: dict, lane: modal.Function, *, manifest: dict | None = None, lane_fields=None) -> dict:
    """Serve one model (and a self-hosted user, if any) for exactly as long as its domain lanes run.

    ``lane`` runs each domain against the tunnels; ``lane_fields(domain, base_url)`` adds its own job fields.
    Returns the invocation record (also written under the run's ``invocations/``).
    """

    tag, run, smoke_samples = job["tag"], job["run_id"], job["smoke_samples"]
    check_agent_window(PROTOCOL, tag)
    model_path = resolve_model_path(tag)
    if model_path.startswith(REMOTE_MODEL_ROOT + "/"):
        model_volume.reload()
    # A warm container's checkpoint mount can predate an export that finished after it started; refresh it first.
    for attempt in range(6):
        if (Path(model_path) / "config.json").exists():
            break
        try:
            checkpoint_volume.reload()
        except Exception as error:  # noqa: BLE001 - a failed refresh only delays the check
            print(f"[{tag}] checkpoint volume reload failed: {error}", flush=True)
        time.sleep(10 * attempt)
    else:
        raise FileNotFoundError(f"{tag} checkpoint not found at {model_path}")
    if USER and not EXTERNAL_USER and not (Path(USER.path) / "config.json").exists():
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
            **(manifest or {}),
        },
    )
    results_volume.commit()

    def devices(first: int, count: int) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(first, first + count))}

    # token_urlsafe can start with '-', which vllm's argument parser reads as a flag.
    api_key = "sk-" + secrets.token_urlsafe(32)
    server = VllmServer(
        SERVING.vllm_command(model_path, api_key, PROTOCOL), SERVING.port, devices(0, SERVING.gpus) if USER else None
    )
    external = job.get("external_user")
    if EXTERNAL_USER and external is None:
        server.stop()
        raise RuntimeError("DeepSeek requires its separate user container; use evaluate_with_external_user")
    servers, user_server = [server], None
    user_key = external["api_key"] if external else None
    user_probe = external["metadata"]["probe"] if external else None
    restarts: list[float] = []
    user_restarts: list[float] = []
    stop = threading.Event()
    try:
        server.wait_ready()
        check_serving_window(list(server.lines), AGENT_EXPECTED, legacy_log=not MODERN_AGENT)
        defaults = (check_modern_agent_sampling(server) if MODERN_AGENT else
                    check_generation_defaults(list(server.lines), AGENT_EXPECTED))
        probe = probe_endpoint(server, api_key, AGENT_EXPECTED)
        if isinstance(PROTOCOL, Qwen35MentorProtocol):
            probe["reasoning_history"] = probe_current_step_history(server, api_key, model_path)
        print(f"[{tag}] sampling defaults {defaults}; probe {probe}", flush=True)
        if USER and not EXTERNAL_USER:
            # Started after the agent is up, so the two servers' ports cannot race.
            user_key = "sk-" + secrets.token_urlsafe(32)
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
            user_base_url = (
                external["base_url"] if external else
                f"{tunnels.enter_context(modal.forward(SERVING.user_port)).url}/v1" if user_server else None
            )
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
                    "invocation": job["invocation"],
                    "trials": job["trials"],
                    "smoke_samples": smoke_samples,
                    "concurrency": job["concurrency"] or LANES.concurrency_for(name),
                    **(lane_fields(name, base_url) if lane_fields else {}),
                }
                for name in job["domains"]
            ]
            calls = [lane.spawn(item) for item in lanes]
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
    for item, outcome in zip(lanes, outcomes):
        if isinstance(outcome, BaseException):
            failures.append({"domain": item["domain"], "error": repr(outcome)})
        else:
            results.append(outcome)
            if outcome["failures"] or not outcome["complete"]:
                failures.append({"domain": item["domain"], "missing": outcome["expected"] - outcome["conversations"]})
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
        **({"external_user": external["metadata"]} if external else {}),
    }
    path = root / "invocations" / f"{job['invocation']}.json"
    path.write_text(json.dumps(invocation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_volume.commit()
    return invocation


@app.function(**SERVER_FUNCTION)
def serve_and_evaluate(job: dict) -> dict:
    """Evaluate one model on its domain lanes, then summarize the run's records."""

    invocation = serve(job, run_lane)
    # Other invocations may have written other domains of this run, so the
    # summary is rebuilt from every record on disk.
    summary = summarize_run.remote(job["run_id"], job["tag"], job["smoke_samples"])
    return {**summary, "invocation": {key: value for key, value in invocation.items() if key != "lanes"}}



def check_deepseek_sampling(server: VllmServer) -> dict:
    """vLLM 0.28 no longer logs default sampling; verify the pinned config and live overrides."""

    text = "".join(server.lines)
    required = (
        f"'override_generation_config': {{'max_new_tokens': {USER.max_new_tokens}}}",
        "'default_chat_template_kwargs': {'thinking': False}",
    )
    if any(value not in text for value in required):
        raise RuntimeError("DeepSeek startup did not confirm the token cap and thinking=false")
    config = json.loads((Path(USER.path) / "generation_config.json").read_text())
    defaults = {key: config[key] for key in ("temperature", "top_p")}
    defaults["max_tokens"] = USER.max_new_tokens
    if defaults != USER.expected_generation_defaults:
        raise RuntimeError(f"DeepSeek pinned sampling defaults changed: {defaults}")
    return defaults


def probe_nonthinking_user(server: VllmServer, key: str) -> dict:
    """Check auth, response visibility, non-thinking mode, and native tools without a long prefill."""

    status, _ = request_json(server.url("/v1/models"), None, None)
    if status != 401:
        raise RuntimeError(f"user endpoint is not authenticated: HTTP {status}")
    messages = [
        {"role": "system", "content": "You are a retail customer. You want to return your delivered blue jacket. Your order is #12345. Answer the assistant naturally in one short sentence."},
        {"role": "user", "content": "How can I help you today?"},
    ]
    started = time.monotonic()
    status, response = request_json(server.url("/v1/chat/completions"), {
        "model": USER.served_model_name, "messages": messages, "temperature": 0,
    }, key)
    elapsed = time.monotonic() - started
    if status != 200:
        raise RuntimeError(f"non-thinking probe failed: HTTP {status}: {response}")
    message = response["choices"][0]["message"]
    content = message.get("content") or ""
    if not content.strip() or message.get("reasoning") or message.get("reasoning_content") or any(
        marker in content for marker in ("<think>", "</think>", "[think]", "[end_think]")
    ):
        raise RuntimeError(f"non-thinking user has empty content or reasoning: {message}")
    if response["choices"][0]["finish_reason"] != "stop":
        raise RuntimeError(f"user probe did not end normally: {response}")
    # Telecom users have tools; validate the parser even though this smoke is retail-only.
    status, tool_response = request_json(server.url("/v1/chat/completions"), {
        "model": USER.served_model_name, "temperature": 0, "max_tokens": 512,
        "messages": [{"role": "user", "content": "Call get_weather for Paris."}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    }, key)
    if status != 200 or not tool_response["choices"][0]["message"].get("tool_calls"):
        raise RuntimeError(f"DeepSeek tool parser probe failed: HTTP {status}: {tool_response}")
    call = tool_response["choices"][0]["message"]["tool_calls"][0]["function"]
    if call["name"] != "get_weather" or json.loads(call["arguments"]).get("city") != "Paris":
        raise RuntimeError(f"unexpected probe tool: {call}")
    return {"content": content, "reasoning_absent": True, "tool_call": call,
            "completion_tokens": response["usage"]["completion_tokens"], "seconds": elapsed,
            "tokens_per_second": response["usage"]["completion_tokens"] / elapsed}


@app.function(
    image=external_user_image, secrets=[profile_secret], gpu="H200:4", cpu=32, memory=196_608,
    timeout=SERVING.server_timeout_s,
    volumes={REMOTE_MODEL_ROOT: model_volume.read_only(), REMOTE_RESULTS_ROOT: results_volume,
             "/root/.cache/vllm": vllm_cache_volume},
)
def evaluate_with_external_user(job: dict) -> dict:
    """Keep the separate H200 user server alive until the agent's smoke/evaluation finishes."""

    if not EXTERNAL_USER:
        raise RuntimeError("select TAU_PROFILE=userdeepseek")
    collection = job.get("work") == "collection"
    if not collection:
        summary = summarize_run.remote(job["run_id"], job["tag"], job["smoke_samples"])
        by_domain = {lane["domain"]: lane for lane in summary["lanes"]}
        if all(name in by_domain and by_domain[name]["complete"] and not by_domain[name]["failures"]
               and by_domain[name]["trials"] >= job["trials"] for name in job["domains"]):
            print("requested conversations already complete; skipping user/agent reload", flush=True)
            return {**summary, "invocation": {"recovered_complete": True, "domains": job["domains"], "trials": job["trials"]}}
    root = Path(REMOTE_RESULTS_ROOT) / job["run_id"] / job["tag"] / "invocations"
    root.mkdir(parents=True, exist_ok=True)
    key = "sk-" + secrets.token_urlsafe(32)
    started = time.monotonic()
    server = VllmServer(SERVING.user_vllm_command(USER, key), SERVING.user_port,
                        label="[deepseek-user] ", startup_timeout_s=SERVING.user_startup_timeout_s)
    stop, restarts = threading.Event(), []
    metadata = {"image": USER.serving_image, "model": USER.model, "revision": USER.revision}
    try:
        server.wait_ready()
        metadata["ready_seconds"] = time.monotonic() - started
        metadata["gpus"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()
        check_serving_window(list(server.lines), USER_EXPECTED, legacy_log=False)
        metadata["generation_defaults"] = check_deepseek_sampling(server)
        metadata["sampling_validation"] = "pinned generation config and live server overrides"
        metadata["probe"] = probe_nonthinking_user(server, key)
        print(f"DeepSeek user ready: {json.dumps(metadata)}", flush=True)
        (root / f"{job['invocation']}.user.json").write_text(json.dumps(metadata, indent=2) + "\n")
        results_volume.commit()
        watch_server(server, "deepseek-user", stop, restarts)
        monitor = monitor_metrics(server, "deepseek-user", root / f"{job['invocation']}.user.vllm_metrics.jsonl", stop)
        committer = commit_periodically(results_volume, stop, 120)
        with modal.forward(SERVING.user_port) as tunnel:
            worker = serve_and_collect if collection else serve_and_evaluate
            result = worker.remote({**job, "external_user": {
                "base_url": f"{tunnel.url}/v1", "api_key": key, "metadata": metadata,
            }})
        result["invocation"]["user_server_restarts"] = len(restarts)
        return result
    except BaseException as error:
        metadata["error"] = repr(error)
        raise
    finally:
        stop.set()
        with server.lock:
            server.stop()
        if "monitor" in locals():
            monitor.join()
            committer.join()
        metadata["server_restarts"] = len(restarts)
        metadata["total_seconds"] = time.monotonic() - started
        (root / f"{job['invocation']}.user.json").write_text(json.dumps(metadata, indent=2) + "\n")
        log = "".join(server.lines).replace(key, "<redacted>")
        (root / f"{job['invocation']}.user.log").write_text(log)
        results_volume.commit()

@app.function(**SERVER_FUNCTION)
def serve_and_collect(job: dict) -> dict:
    """Run one collector model on AReaL training tasks (``job["task_ids"]``: {domain: [ids]}), keeping every request."""

    invocation = serve(
        job,
        collect_lane,
        manifest={"collection": {"areal": {"repo": AREAL_REPO, "revision": AREAL_REVISION, "tasks_sha256": AREAL_TASKS_SHA256}}},
        lane_fields=lambda name, base_url: {
            "task_ids": job["task_ids"][name],
            # vLLM's /tokenize renders a chat exactly as the server does.
            "tokenize_url": base_url[: -len("/v1")] + "/tokenize",
        },
    )
    return {"tag": job["tag"], "invocation": invocation}


@app.function(**SERVER_FUNCTION)
def serve_and_branch(job: dict) -> dict:
    """Run one shard of a branch plan (``job["branch_ids"]``: {domain: [ids]}) with the student and the user."""

    invocation = serve(
        job,
        branch_lane,
        manifest={
            "branch": {"plan": job["plan"], "plan_sha256": job["plan_sha256"], "shard": job["shard"]},
            "collection": {"areal": {"repo": AREAL_REPO, "revision": AREAL_REVISION, "tasks_sha256": AREAL_TASKS_SHA256}},
        },
        lane_fields=lambda name, base_url: {"branch_ids": job["branch_ids"][name], "plan": job["plan"]},
    )
    return {"tag": job["tag"], "invocation": invocation}


@app.function(
    image=lane_image,
    secrets=[profile_secret],
    cpu=1,
    memory=2_048,
    timeout=SERVING.server_timeout_s,
    volumes={REMOTE_CHECKPOINT_ROOT: checkpoint_volume.read_only()},
)
def wait_and_evaluate(job: dict, wait_hours: float) -> dict:
    """Start ``serve_and_evaluate`` once the checkpoint is complete, waiting on a CPU instead of on held GPUs.

    An agent-efficiency export is complete once its provenance file exists (written after the weights).
    """

    path = Path(resolve_model_path(job["tag"]))
    marker = path / ("agent_eff_provenance.json" if job["tag"].startswith("ae.") else "config.json")
    deadline = time.time() + wait_hours * 3600
    while not marker.exists():
        if time.time() > deadline:
            raise TimeoutError(f"{marker} did not appear within {wait_hours} h")
        time.sleep(300)
        checkpoint_volume.reload()
    return serve_and_evaluate.remote(job)


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
        reference = row["qwen_reported"]
        line = name.ljust(14) + (f"{reference:7.1f}" if reference is not None else " " * 7)
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
    wait_hours: float = 0.0,
) -> None:
    """Evaluate each model on TAU1/TAU2, concurrently, four GPUs per model.

    ``--wait-hours``: each model starts once its checkpoint is complete (for example, students still training).
    """

    tags = [tag.strip() for tag in models.split(",") if tag.strip()]
    for tag in tags:
        resolve_model_path(tag)
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
        if "qwen35" in tags:
            print(f"pinned Qwen3.5 baseline: {download_agent_model.remote('qwen35')}", flush=True)
        if USER:
            print(f"user simulator {USER.model}@{USER.revision[:8]}, self-hosted: {download_user_model.remote()}", flush=True)
            users = {USER.model: {"self_hosted": True, "revision": USER.revision}}
        else:
            users = check_user_models.remote(sorted({PROTOCOL.domain(name).user_model for name in selected}))
            print(f"user simulators via {USER_API_BASE}: {users}", flush=True)
        invocation = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + "+".join(sorted(selected))
        jobs = [
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
            for tag in tags
        ]
        if EXTERNAL_USER and wait_hours:
            raise ValueError("external user evaluation does not support checkpoint waiters")
        evaluator = evaluate_with_external_user if EXTERNAL_USER else serve_and_evaluate
        calls = [wait_and_evaluate.spawn(job, wait_hours) if wait_hours else evaluator.spawn(job) for job in jobs]
        summaries = [call.get() for call in calls]
        print(json.dumps([summary["invocation"] for summary in summaries], indent=2, sort_keys=True))
    comparison = compare(summaries)
    write_comparison.remote(run, comparison)
    print_comparison(comparison, tags)


@app.local_entrypoint()
def branch(plan: str, run_prefix: str, shards: int = 1, tag: str = "thinking2507", concurrency: int = 0, limit: int = 0) -> None:
    """Executed branches: the plan (a JSONL on the results volume, from data_curation/branch_outcomes.py) is split by
    hash into ``shards`` runs ``<run_prefix>-s<k>``, each served on its own node. Finished branches are skipped, so
    a rerun resumes. ``limit`` keeps the first branches of each domain (a smoke test).
    """

    import hashlib

    if not run_prefix.startswith("branch-"):
        raise ValueError("branch run ids start with branch-")
    resolve_model_path(tag)
    check_agent_window(PROTOCOL, tag)
    read = read_branch_plan.remote(plan)
    print(f"plan {plan} ({read['sha256'][:12]}): " + ", ".join(f"{name} {len(ids)}" for name, ids in read["branches"].items()), flush=True)
    print(f"user simulator {USER.model}@{USER.revision[:8]}, self-hosted: {download_user_model.remote()}", flush=True)
    invocation = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-branch"
    jobs = []
    for shard in range(shards):
        ids = {
            name: [
                branch_id for branch_id in (sorted(branch_ids)[:limit] if limit else branch_ids)
                if int(hashlib.sha256(branch_id.encode()).hexdigest(), 16) % shards == shard
            ]
            for name, branch_ids in read["branches"].items()
        }
        ids = {name: items for name, items in ids.items() if items}
        if ids:
            jobs.append(
                {
                    "tag": tag, "run_id": f"{run_prefix}-s{shard}", "shard": shard, "domains": sorted(ids),
                    "branch_ids": ids, "plan": plan, "plan_sha256": read["sha256"], "trials": 1, "smoke_samples": 0,
                    "concurrency": concurrency,
                    "user_models": {USER.model: {"self_hosted": True, "revision": USER.revision}},
                    "invocation": invocation,
                }
            )
    calls = [serve_and_branch.spawn(job) for job in jobs]
    print(json.dumps([call.get()["invocation"] for call in calls], indent=2, sort_keys=True))


@app.local_entrypoint()
def collect(plan: str, concurrency: int = 0) -> None:
    """Fresh-state collection: each collector model plays its AReaL training tasks (configs/tau_bench_eval/harness.py).

    ``plan.json`` is {"run_id": "collect-...", "assignments": {tag: {domain: [task ids]}}}. Rerunning a plan skips
    finished episodes; data_curation/fresh_states.py extracts training states from the records.
    """

    spec = json.loads(Path(plan).read_text(encoding="utf-8"))
    if not spec["run_id"].startswith("collect-"):
        raise ValueError("collection run ids start with collect-")
    for tag, by_domain in spec["assignments"].items():
        resolve_model_path(tag)
        check_agent_window(PROTOCOL, tag)
        for name in by_domain:
            PROTOCOL.domain(name)
    print(f"user simulator {USER.model}@{USER.revision[:8]}, self-hosted: {download_user_model.remote()}", flush=True)
    # Downstream stages wait for every collector's "*-collect.json" invocation record.
    invocation = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-collect"
    calls = [
        (evaluate_with_external_user if EXTERNAL_USER else serve_and_collect).spawn(
            {
                "tag": tag,
                "run_id": spec["run_id"],
                "domains": sorted(by_domain),
                "task_ids": by_domain,
                "trials": 1,
                "smoke_samples": 0,
                "concurrency": concurrency,
                "user_models": {USER.model: {"self_hosted": True, "revision": USER.revision}},
                "invocation": invocation,
                "work": "collection",
            }
        )
        for tag, by_domain in spec["assignments"].items()
    ]
    print(json.dumps([call.get()["invocation"] for call in calls], indent=2, sort_keys=True))
