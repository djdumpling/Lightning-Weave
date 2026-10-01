# /// script
# requires-python = "==3.12.*"
# dependencies = ["modal==1.5.5"]
# ///
"""Modal orchestration for BFCL evaluation of the LoopTool OPD student.

Each model gets one H100 running vLLM behind a key-protected tunnel. Every BFCL
category runs in its own CPU container through the pinned NeMo-Skills driver,
because ``bfcl_eval`` hardcodes its result and score trees under /opt/gorilla.
Nothing runs at import time.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import modal

# Modal copies the entrypoint to /root/modal_eval.py and config.py beside it.
ENTRYPOINT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENTRYPOINT_DIR))

from config import (  # noqa: E402 - after the sys.path setup above
    AGENTIC_EVAL_BFCL_SUBDIR,
    AGENTIC_EVAL_COMMIT,
    AGENT_EFF_PREFIX,
    BFCL_PROJECT_ROOT,
    CONTEXT_OVERFLOW_MARKERS,
    DDGS_VERSION,
    DEPENDENCY_CUTOFF,
    GORILLA_COMMIT,
    GORILLA_REPO,
    GORILLA_ROOT,
    LANE_PYTHON,
    LANES,
    MEMORY_PREFIX,
    MODAL_APP_NAME,
    MODAL_CHECKPOINT_VOLUME,
    MODAL_MODEL_VOLUME,
    MODAL_RESULTS_VOLUME,
    MODAL_VLLM_CACHE_VOLUME,
    NEMO_SKILLS_COMMIT,
    NEMO_SKILLS_REPO,
    PROTOCOL,
    RECIPE_GEN_BUDGET,
    REMOTE_CHECKPOINT_ROOT,
    REMOTE_MODEL_ROOT,
    REMOTE_RESULTS_ROOT,
    SCORE_MODEL_DIR,
    SERVING,
    SERVING_IMAGE,
    WEB_PREFIX,
    ModelSpec,
    driver_command,
    resolve_model,
    run_id,
)

NEMO_SKILLS_ROOT = "/opt/NeMo-Skills"
AGENTIC_EVAL_ROOT = "/opt/agentic-eval"
AGENTIC_EVAL_BFCL = f"{AGENTIC_EVAL_ROOT}/{AGENTIC_EVAL_BFCL_SUBDIR}"
AGENTIC_EVAL_LOCAL = Path(os.environ.get("AGENTIC_EVAL_DIR", str(Path.home() / "agentic-eval")))
DATASET_ROOTS = {version: f"{NEMO_SKILLS_ROOT}/nemo_skills/dataset/bfcl_{version}" for version in RECIPE_GEN_BUDGET}
DATASET_ROOT = DATASET_ROOTS[PROTOCOL.bfcl_version]
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
UV_INSTALL = f"uv pip install --system --no-cache-dir --exclude-newer {DEPENDENCY_CUTOFF}"
CONFIG_FILE = str(ENTRYPOINT_DIR / "config.py")
GORILLA_CONSTRAINTS = "/opt/gorilla-constraints.txt"
# gorilla's exact `==` dependency pins, e.g. tree_sitter==0.21.3 for the Java and
# JavaScript checkers; later installs must not move them.
WRITE_GORILLA_CONSTRAINTS = (
    "python -c 'import pathlib, re; "
    "deps = pathlib.Path(\"pyproject.toml\").read_text().split(\"dependencies = [\", 1)[1].split(\"]\", 1)[0]; "
    "pins = re.findall(r\"\\\"([A-Za-z0-9_.-]+==[^\\\"]+)\\\"\", deps); "
    f"pathlib.Path(\"{GORILLA_CONSTRAINTS}\").write_text(chr(10).join(pins) + chr(10)); print(pins)'"
)


def check_agentic_eval_checkout(checkout: Path) -> None:
    """The copied BFCL files must be exactly agentic-eval at the pin."""

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()

    head = git("rev-parse", "HEAD")
    if head != AGENTIC_EVAL_COMMIT:
        raise RuntimeError(f"{checkout} is at {head}; check out {AGENTIC_EVAL_COMMIT} or set AGENTIC_EVAL_DIR")
    dirty = git("status", "--porcelain", "--", AGENTIC_EVAL_BFCL_SUBDIR)
    if dirty:
        raise RuntimeError(f"{checkout}/{AGENTIC_EVAL_BFCL_SUBDIR} has local changes:\n{dirty}")


if modal.is_local():
    check_agentic_eval_checkout(AGENTIC_EVAL_LOCAL)

app = modal.App(MODAL_APP_NAME)
results_volume = modal.Volume.from_name(MODAL_RESULTS_VOLUME, create_if_missing=True)
model_volume = modal.Volume.from_name(MODAL_MODEL_VOLUME)
checkpoint_volume = modal.Volume.from_name(MODAL_CHECKPOINT_VOLUME)
vllm_cache_volume = modal.Volume.from_name(MODAL_VLLM_CACHE_VOLUME, create_if_missing=True)


def load_lanes():
    """agentic-eval's lane planner: the single source of categories and budgets."""

    sys.path.insert(0, AGENTIC_EVAL_BFCL)
    import lanes

    return lanes


def load_scorer(version: str):
    """agentic-eval's vendored aggregators, with its README's v3 import shim for v4."""

    sys.path.insert(0, f"{AGENTIC_EVAL_BFCL}/vendor")
    import bfcl_score_v3

    if version == "v3":
        return bfcl_score_v3
    sys.modules["nemo_skills.dataset.bfcl_v3.bfcl_score"] = bfcl_score_v3
    import bfcl_score_v4

    return bfcl_score_v4


def version_categories(lanes, version: str) -> list[str]:
    return list(lanes.V3_CATEGORIES if version == "v3" else lanes.V4_CATEGORIES)


def verify_lane_image() -> None:
    """Fail the image build instead of silently scoring a mismatched environment."""

    import importlib
    import importlib.metadata

    for pin in Path(GORILLA_CONSTRAINTS).read_text(encoding="utf-8").split():
        name, version = pin.split("==")
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"gorilla pins {pin} but {importlib.metadata.version(name)} is installed")
    for module in (
        # The first import of `python -m bfcl_eval evaluate`; loads the Java/JS tree-sitter parsers.
        "bfcl_eval.constants.model_config",
        "nemo_skills.inference.eval.bfcl",
        "nemo_skills.inference.eval.bfcl_web_search",
        "bfcl_eval",
        "ddgs",
        # Loads the baked all-MiniLM-L6-v2 encoder used by memory_vector.
        "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_vector",
    ):
        importlib.import_module(module)
    import nemo_skills

    if not os.path.realpath(nemo_skills.__file__).startswith(NEMO_SKILLS_ROOT):
        raise RuntimeError(f"pinned NeMo-Skills is not the active install: {nemo_skills.__file__}")
    for root, commit in ((GORILLA_ROOT, GORILLA_COMMIT), (NEMO_SKILLS_ROOT, NEMO_SKILLS_COMMIT)):
        head = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        if head != commit:
            raise RuntimeError(f"{root} is at {head}, expected {commit}")

    # Each test.jsonl must come from gorilla@pin, the same data the checker grades.
    lanes = load_lanes()
    source_root = Path(BFCL_PROJECT_ROOT) / "bfcl_eval/data"
    for version, dataset_root in DATASET_ROOTS.items():
        for category in version_categories(lanes, version):
            prepared = Path(dataset_root) / category / "test.jsonl"
            ids = [json.loads(line)["id"] for line in prepared.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not ids:
                raise RuntimeError(f"{prepared} is empty")
            # gorilla names its data files BFCL_v4_* for both versions.
            source = source_root / f"BFCL_v4_{category}.json"
            if source.exists():
                expected = [json.loads(line)["id"] for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
                if sorted(ids) != sorted(expected):
                    raise RuntimeError(f"{version} {category}: prepared ids differ from gorilla@{GORILLA_COMMIT[:8]}")
            print(f"verified {version} {category}: {len(ids)} rows", flush=True)


serving_image = (
    modal.Image.from_registry(SERVING_IMAGE)
    .entrypoint([])
    # The official vLLM image exposes /usr/bin/python3 but no `python`, which
    # Modal needs to detect the image's Python version.
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python")
    .env({"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1"})
    .add_local_file(CONFIG_FILE, "/root/config.py", copy=True)
)

lane_image = (
    modal.Image.debian_slim(python_version=LANE_PYTHON)
    # ffmpeg as in Dockerfile.nemo-skills: torchcodec (a NeMo-Skills core dependency) links against it.
    .apt_install("git", "curl", "ffmpeg")
    .run_commands(
        # bfcl_eval is installed editable in place, as in Dockerfile.nemo-skills.
        f"git clone {GORILLA_REPO} {GORILLA_ROOT}",
        # A branch at the pin makes NeMo-Skills' shallow clone of this checkout
        # resolve to the pin; upstream prepare.py would otherwise clone gorilla HEAD.
        f"git -C {GORILLA_ROOT} checkout -B bfcl-pin {GORILLA_COMMIT}",
        "pip install --no-cache-dir 'uv>=0.11.10'",
        # CPU torch first, so sentence-transformers does not pull the CUDA build.
        f"pip install --no-cache-dir torch --index-url {CPU_TORCH_INDEX}",
        f"cd {BFCL_PROJECT_ROOT} && {UV_INSTALL} -e . && {WRITE_GORILLA_CONSTRAINTS}",
        f"git clone {NEMO_SKILLS_REPO} {NEMO_SKILLS_ROOT} && git -C {NEMO_SKILLS_ROOT} checkout {NEMO_SKILLS_COMMIT}",
        # uv honors pyproject's [tool.uv] overrides, which the upstream Dockerfile relies on.
        # compute-eval (an unrelated CUDA benchmark, imported only for eval_type=compute-eval)
        # requires tree-sitter>=0.25.2, which conflicts with gorilla's checker pin.
        f"cd {NEMO_SKILLS_ROOT} && grep -v '^compute-eval' core/requirements.txt > /opt/nemo-skills-requirements.txt"
        f" && {UV_INSTALL} -r /opt/nemo-skills-requirements.txt soundfile --constraint {GORILLA_CONSTRAINTS}",
        # As in Dockerfile.nemo-skills; imported by the driver's web-search module.
        f"uv pip install --system --no-cache-dir ddgs=={DDGS_VERSION} --constraint {GORILLA_CONSTRAINTS}",
        # gorilla depends on the py2 'pathlib' backport, which shadows the stdlib module.
        "pip uninstall -y pathlib || true",
        f"pip install --no-cache-dir --no-deps -e {NEMO_SKILLS_ROOT}",
    )
    .env({"HF_HOME": "/opt/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', device='cpu')\"",
        # file:// avoids git's hardlinked local clone, whose safety check fails across
        # image layers. prepare.py only logs clone errors; verify_lane_image catches them.
        f"cd {NEMO_SKILLS_ROOT} && python -c \"import argparse, nemo_skills.dataset.bfcl_v3.prepare as p; "
        f"p.REPO_URL = 'file://{GORILLA_ROOT}'; p.main(argparse.Namespace(model_type=None))\"",
        f"cd {NEMO_SKILLS_ROOT} && python -c \"import nemo_skills.dataset.bfcl_v4.prepare as p; "
        f"p.REPO_URL = 'file://{GORILLA_ROOT}'; p.main()\"",
        "pip freeze > /opt/lane-requirements.txt",
    )
    .add_local_dir(
        str(AGENTIC_EVAL_LOCAL / AGENTIC_EVAL_BFCL_SUBDIR), AGENTIC_EVAL_BFCL, copy=True, ignore=["**/__pycache__/**"]
    )
    .add_local_file(CONFIG_FILE, "/root/config.py", copy=True)
    .run_function(verify_lane_image)
    .env({"HF_HUB_OFFLINE": "1"})
)


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


def first_json_line(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.loads(handle.readline())


@app.function(image=lane_image, cpu=1, memory=2_048, timeout=600)
def plan_categories(categories: list[str], smoke_samples: int) -> list[dict]:
    """One category per lane, memory first: agentic-eval's plan() with n_lanes = len(categories)."""

    lanes = load_lanes()
    version = PROTOCOL.bfcl_version
    if dict(lanes.GEN_BUDGET) != RECIPE_GEN_BUDGET:
        raise RuntimeError(f"config.RECIPE_GEN_BUDGET differs from agentic-eval's GEN_BUDGET {lanes.GEN_BUDGET}")
    available = version_categories(lanes, version)
    unknown = sorted(set(LANES.concurrency) - set(available))
    if unknown:
        raise RuntimeError(f"concurrency overrides name categories outside bfcl_{version}: {unknown}")
    web = [category for category in categories if category.startswith(WEB_PREFIX)]
    if web:
        raise RuntimeError(f"{web} need a web-search backend, which this recipe does not set up")
    selected = list(categories) or [category for category in available if not category.startswith(WEB_PREFIX)]
    if smoke_samples and not categories:
        # max_samples truncates before prereq filtering, so memory smoke runs are meaningless.
        selected = [category for category in selected if not category.startswith(MEMORY_PREFIX)]
    plan = lanes.plan(selected, version, n_lanes=len(selected))
    if any(len(lane) != 1 for lane in plan):
        raise RuntimeError(f"expected one category per lane, got {plan}")
    return [{"category": lane[0], "concurrency": LANES.concurrency_for(lane[0])} for lane in plan]


@app.function(
    image=lane_image,
    cpu=LANES.cpu,
    memory=LANES.memory_mb,
    timeout=LANES.timeout_s,
    volumes={REMOTE_RESULTS_ROOT: results_volume},
)
def run_category(job: dict) -> dict:
    """Generate and score one category against the tunnelled vLLM endpoint."""

    category, tag = job["category"], job["tag"]
    model_root = Path(REMOTE_RESULTS_ROOT) / job["run_id"] / tag
    output_file = model_root / f"bfcl_{PROTOCOL.bfcl_version}.{category}" / "output.jsonl"
    # bfcl_eval names score files BFCL_v4_* for both versions.
    score_copy = model_root / "scores" / f"BFCL_v4_{category}_score.json"
    results_volume.reload()
    if score_copy.exists():
        # Memory categories rerun all 37 serial prereqs before skip_filled applies.
        return {"category": category, "status": "cached", **first_json_line(score_copy)}

    environment = {**os.environ, "OPENAI_API_KEY": job["api_key"], "NGC_API_KEY": "dummy"}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    command = driver_command(
        category=category,
        input_file=f"{DATASET_ROOT}/{category}/test.jsonl",
        output_file=str(output_file),
        base_url=job["base_url"],
        concurrency=job["concurrency"],
        smoke_samples=job["smoke_samples"],
        random_seed=job["seed"],
    )
    print(f"[{tag}:{category}] concurrency={job['concurrency']}", flush=True)

    stop = threading.Event()
    committer = commit_periodically(results_volume, stop, LANES.commit_interval_s)
    started = time.time()
    last_progress = 0.0
    process = subprocess.Popen(
        command, cwd=NEMO_SKILLS_ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert process.stdout is not None
    for line in process.stdout:
        if "Remaining generations" in line:
            # tqdm redraws constantly; keep one progress line per minute.
            if time.time() - last_progress < 60:
                continue
            last_progress = time.time()
        print(f"[{tag}:{category}] {line}", end="", flush=True)
    returncode = process.wait()
    stop.set()
    committer.join()
    results_volume.commit()
    if returncode:
        raise RuntimeError(f"{tag}:{category} driver exited with {returncode}")

    scores = list((Path(BFCL_PROJECT_ROOT) / "score" / SCORE_MODEL_DIR).rglob(f"BFCL_v4_{category}_score.json"))
    if len(scores) != 1:
        raise RuntimeError(f"{tag}:{category}: expected one score file, found {scores}")
    score_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scores[0], score_copy)
    results_volume.commit()
    # Prompts longer than max_model_len - tokens_to_generate are rejected and fail.
    out_of_context = sum(
        '"_ran_out_of_context_"' in line for line in output_file.read_text(encoding="utf-8").splitlines()
    )
    return {
        "category": category,
        "status": "scored",
        "seconds": round(time.time() - started, 1),
        "out_of_context": out_of_context,
        **first_json_line(score_copy),
    }


@app.function(image=lane_image, cpu=1, memory=2_048, timeout=600, volumes={REMOTE_RESULTS_ROOT: results_volume})
def aggregate_scores(run: str, tag: str) -> dict:
    """Leaderboard aggregate from the per-category score headers, via agentic-eval's scorer."""

    results_volume.reload()
    version = PROTOCOL.bfcl_version
    root = Path(REMOTE_RESULTS_ROOT) / run / tag
    paths = {
        category: root / "scores" / f"BFCL_v4_{category}_score.json"
        for category in version_categories(load_lanes(), version)
    }
    missing = [category for category, path in paths.items() if not path.exists()]
    metrics = {}
    for category, path in paths.items():
        if path.exists():
            header = first_json_line(path)
            metrics[f"bfcl_{version}.{category}"] = {
                "pass@1": {"accuracy": header["accuracy"], "num_entries": header["total_count"]}
            }
    scorer = load_scorer(version)
    if missing:
        # The overall score needs every category; report the buckets that are complete.
        buckets = {}
        bucket_functions = [
            scorer.calculate_non_live_single_turn_accuracy,
            scorer.calculate_live_single_turn_accuracy,
            scorer.calculate_multi_turn_accuracy,
        ]
        if version == "v4":
            bucket_functions.append(scorer.calculate_hallucination_measurement)
        for bucket in bucket_functions:
            try:
                buckets.update(bucket(metrics))
            except Exception:  # noqa: BLE001 - the scorer raises on a missing category
                continue
        return {"complete": False, "missing": missing, **buckets}
    aggregate = scorer.compute_score(metrics)[f"bfcl_{version}"]
    (root / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_volume.commit()
    return {"complete": True, **aggregate}


class VllmServer:
    """A local vLLM process whose log is teed to stdout and kept for startup checks."""

    def __init__(self, command: list[str]) -> None:
        self.lines: collections.deque[str] = collections.deque(maxlen=20_000)
        printable = [("<redacted>" if previous == "--api-key" else part) for previous, part in zip([""] + command, command)]
        print("+", " ".join(printable), flush=True)
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.reader = threading.Thread(target=self._tee, daemon=True)
        self.reader.start()

    def _tee(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.append(line)
            print(line, end="", flush=True)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{SERVING.port}{path}"

    def wait_ready(self) -> None:
        deadline = time.time() + SERVING.startup_timeout_s
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
        raise TimeoutError(f"vLLM was not healthy within {SERVING.startup_timeout_s}s")

    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()


def check_serving_window(lines: list[str], *, yarn: bool = True) -> None:
    """Confirm vLLM started with the YaRN window rather than silently capping it."""

    text = "".join(lines)
    if f"max_seq_len={PROTOCOL.max_model_len}" not in text:
        raise RuntimeError(f"vLLM did not report max_seq_len={PROTOCOL.max_model_len}; check --hf-overrides rope_scaling")
    if yarn and "yarn" not in text:
        raise RuntimeError("vLLM's startup log does not mention the yarn rope_scaling override")


def check_generation_defaults(lines: list[str]) -> dict:
    """Confirm vLLM applies the checkpoint's sampling defaults, including top_k=20."""

    for line in lines:
        if "default chat sampling params" not in line.lower():
            continue
        match = re.search(r"\{.*\}", line)
        defaults = ast.literal_eval(match.group(0)) if match else {}
        mismatched = {
            key: (defaults.get(key), value)
            for key, value in PROTOCOL.expected_generation_defaults.items()
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


def probe_endpoint(server: VllmServer, api_key: str, *, thinking: bool = True) -> dict:
    """The serving contract BFCL relies on: auth, overflow errors, split reasoning, native tool_calls."""

    status, _ = request_json(server.url("/v1/models"), None, None)
    if status != 401:
        raise RuntimeError(f"unauthenticated request returned {status}; the public tunnel must require the key")
    # NeMo-Skills sends max_completion_tokens; an overflow must be an error it
    # recognizes, or the task crashes its lane instead of soft-failing.
    status, error = request_json(
        server.url("/v1/chat/completions"),
        {
            "model": PROTOCOL.served_model_name,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": PROTOCOL.max_model_len,
        },
        api_key,
    )
    if status != 400 or not any(marker in json.dumps(error) for marker in CONTEXT_OVERFLOW_MARKERS):
        raise RuntimeError(f"context overflow returned HTTP {status} without a NeMo-Skills marker: {error}")
    body = {
        "model": PROTOCOL.served_model_name,
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
        status, response = request_json(server.url("/v1/chat/completions"), body, api_key)
        if status != 200:
            raise RuntimeError(f"tool-call probe returned HTTP {status}")
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        if "<think>" in content or "</think>" in content:
            raise RuntimeError("reasoning leaked into content; the qwen3 reasoning parser is not active")
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if thinking and not reasoning:
            raise RuntimeError("no reasoning field in the response; thinking is not being split")
        if not thinking and reasoning and reasoning.strip():
            raise RuntimeError("thinking is disabled but the response still reasons")
        calls = message.get("tool_calls") or []
        if calls:
            call = calls[0]["function"]
            json.loads(call["arguments"])
            if call["name"] != "get_weather":
                raise RuntimeError(f"probe called an unexpected tool: {call}")
            return {"attempts": attempt, "tool_call": call}
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


USAGE_PROXY_PORT = SERVING.port + 1
LANE_PREFIX = "/lane/"
# Hop-by-hop headers are the proxy's own; accept-encoding is dropped so upstream bodies stay parseable.
UNFORWARDED_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}


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

    Whether that reasoning then reaches the model depends on the chat template, which
    ``evaluation/bfcl_breakdown.py --template`` checks.
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


def start_usage_proxy(log_path: Path, tokenizer_path: str, spec: ModelSpec, stop: threading.Event) -> threading.Thread:
    """Forward lane requests to vLLM; log each completion's identity and usage.

    Lanes call ``/lane/<category>/v1/...`` so every record carries its category.
    """

    import asyncio

    import aiohttp
    from aiohttp import web
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    lock = threading.Lock()
    ready = threading.Event()

    def record(lane: str, identity: dict, response: dict) -> None:
        usage = response.get("usage") or {}
        for choice in response.get("choices") or []:
            message = choice.get("message") or {}
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            line = {
                "lane": lane,
                **identity,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": len(tokenizer.encode(reasoning, add_special_tokens=False)) if reasoning else 0,
                "finish_reason": choice.get("finish_reason"),
                "tool_calls": len(message.get("tool_calls") or []),
                "time": time.time(),
            }
            with lock:
                handle.write(json.dumps(line) + "\n")
                handle.flush()

    async def serve() -> None:
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_read=None), connector=aiohttp.TCPConnector(limit=0)
        )

        async def forward(request: web.Request) -> web.StreamResponse:
            lane, path = "", request.path
            if path.startswith(LANE_PREFIX):
                lane, _, rest = path[len(LANE_PREFIX) :].partition("/")
                path = "/" + rest
            body = await request.read()
            identity = None
            if path.endswith("/chat/completions") and request.method == "POST":
                parsed = json.loads(body)
                identity = {**request_identity(parsed), **history_reasoning(parsed)}
                if spec.max_completion_tokens is not None or spec.system_prompt is not None:
                    body = json.dumps(rewrite_request(parsed, spec)).encode("utf-8")
            headers = {key: value for key, value in request.headers.items() if key.lower() not in UNFORWARDED_HEADERS}
            url = f"http://127.0.0.1:{SERVING.port}{path}"
            if request.query_string:
                url += f"?{request.query_string}"
            async with session.request(request.method, url, data=body, headers=headers) as upstream:
                if "text/event-stream" in upstream.headers.get("Content-Type", ""):
                    raise web.HTTPNotImplemented(text="streaming responses are not logged; BFCL lanes do not stream")
                payload = await upstream.read()
                content_type = upstream.headers.get("Content-Type", "application/json")
                status = upstream.status
            if identity is not None and status == 200:
                try:
                    record(lane, identity, json.loads(payload))
                except Exception as error:  # noqa: BLE001 - logging must never fail a request
                    print(f"usage log failed: {error!r}", flush=True)
            return web.Response(body=payload, status=status, headers={"Content-Type": content_type})

        app_ = web.Application(client_max_size=1024**3)
        app_.router.add_route("*", "/{tail:.*}", forward)
        runner = web.AppRunner(app_, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", USAGE_PROXY_PORT).start()
        ready.set()
        while not stop.is_set():
            await asyncio.sleep(1)
        await runner.cleanup()
        await session.close()
        handle.close()

    thread = threading.Thread(target=lambda: asyncio.run(serve()), daemon=True)
    thread.start()
    if not ready.wait(60):
        raise RuntimeError("usage proxy did not start")
    return thread


def thinking_off_template(model_path: str) -> str:
    """The checkpoint's own chat template with enable_thinking forced to false."""

    template = json.loads((Path(model_path) / "tokenizer_config.json").read_text(encoding="utf-8"))["chat_template"]
    if "enable_thinking" not in template:
        raise RuntimeError(f"{model_path}'s chat template has no enable_thinking switch")
    path = Path("/tmp/chat_template_thinking_off.jinja")
    path.write_text("{%- set enable_thinking = false %}\n" + template, encoding="utf-8")
    return str(path)


def serving_record(spec: ModelSpec) -> dict:
    """Non-default serving overrides; empty for the original base/opd tags so their manifests still match."""

    defaults = ModelSpec(spec.path)
    return {
        name: getattr(spec, name)
        for name in ("thinking", "yarn", "reasoning_parser", "max_completion_tokens", "system_prompt")
        if getattr(spec, name) != getattr(defaults, name)
    }


AGENT_EFF_PROVENANCE = "agent_eff_provenance.json"


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


def write_or_check_manifest(path: Path, manifest: dict) -> None:
    """A results tree belongs to exactly one model and protocol; never mix them."""

    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != manifest:
            raise RuntimeError(f"{path} was written for a different model or protocol; use a new run id")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@app.function(
    image=serving_image,
    gpu=SERVING.gpu,
    cpu=8 * SERVING.gpus,
    memory=32_768 * SERVING.gpus,
    timeout=SERVING.server_timeout_s,
    volumes={
        REMOTE_MODEL_ROOT: model_volume.read_only(),
        REMOTE_CHECKPOINT_ROOT: checkpoint_volume.read_only(),
        "/root/.cache/vllm": vllm_cache_volume,
        REMOTE_RESULTS_ROOT: results_volume,
    },
)
def serve_and_evaluate(job: dict) -> dict:
    """Serve one model for exactly as long as its category lanes run."""

    tag, run, smoke_samples = job["tag"], job["run_id"], job["smoke_samples"]
    protocol = replace(PROTOCOL, seed=job["seed"])
    if run_id(smoke_samples, protocol) != run:
        raise RuntimeError(f"run id {run} does not match the protocol for seed {job['seed']}")
    spec = resolve_model(tag)
    model_path = spec.path
    checkpoint_volume.reload()  # a reused container would otherwise miss checkpoints exported since it started
    if not (Path(model_path) / "config.json").exists():
        raise FileNotFoundError(f"{tag} checkpoint not found at {model_path}")
    results_volume.reload()
    root = Path(REMOTE_RESULTS_ROOT) / run / tag
    manifest = {
        "tag": tag,
        "model_path": model_path,
        "model_identity": model_identity(model_path),
        "protocol": protocol.resolved(),
        "smoke_samples": smoke_samples,
    }
    if serving_record(spec):
        manifest["serving_overrides"] = serving_record(spec)
    provenance = Path(model_path) / AGENT_EFF_PROVENANCE
    if provenance.exists():
        manifest["training_provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
    write_or_check_manifest(root / "manifest.json", manifest)
    results_volume.commit()

    # token_urlsafe can start with '-', which vllm's argument parser reads as a flag.
    api_key = "sk-" + secrets.token_urlsafe(32)
    proxy = None
    template = None if spec.thinking else thinking_off_template(model_path)
    server = VllmServer(SERVING.vllm_command(model_path, api_key, protocol, spec, template))
    stop = threading.Event()
    try:
        server.wait_ready()
        check_serving_window(list(server.lines), yarn=spec.yarn)
        defaults = check_generation_defaults(list(server.lines))
        probe = probe_endpoint(server, api_key, thinking=spec.thinking)
        proxy = start_usage_proxy(root / "usage.jsonl", model_path, spec, stop)
        print(f"[{tag}] sampling defaults {defaults}; probe {probe}", flush=True)
        plan = plan_categories.remote(job["categories"], smoke_samples)
        with modal.forward(USAGE_PROXY_PORT) as tunnel:
            base_url = tunnel.url
            monitor = monitor_metrics(server, tag, root / "vllm_metrics.jsonl", stop)
            committer = commit_periodically(results_volume, stop, 600)
            jobs = [
                {
                    **entry,
                    "tag": tag,
                    "run_id": run,
                    "base_url": f"{base_url}{LANE_PREFIX}{entry['category']}/v1",
                    "api_key": api_key,
                    "smoke_samples": smoke_samples,
                    "seed": protocol.seed,
                }
                for entry in plan
            ]
            calls = [run_category.spawn(lane) for lane in jobs]
            outcomes = []
            for call in calls:
                try:
                    outcomes.append(call.get())
                except Exception as error:  # noqa: BLE001 - one failed lane must not hide the others
                    outcomes.append(error)
    finally:
        stop.set()
        server.stop()

    monitor.join()
    committer.join()
    if proxy is not None:
        proxy.join(timeout=30)
    lanes, failures = [], []
    for entry, outcome in zip(plan, outcomes):
        if isinstance(outcome, BaseException):
            failures.append({"category": entry["category"], "error": repr(outcome)})
        else:
            lanes.append(outcome)
    gpus = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout.split("\n")
    # A partial rerun (e.g. one category) must not drop the lanes an earlier launch finished.
    previous = root / "summary.json"
    merged_lanes = {lane["category"]: lane for lane in (json.loads(previous.read_text()).get("lanes", []) if previous.exists() else [])}
    merged_lanes.update({lane["category"]: lane for lane in lanes})
    summary = {
        "tag": tag,
        "run_id": run,
        # Throughput only; not part of the protocol, so a resume may land on either type.
        "gpus": [name for name in gpus if name],
        "lanes": sorted(merged_lanes.values(), key=lambda lane: lane["category"]),
        "failures": failures,
        "probe": probe,
    }
    if not smoke_samples and not failures:
        summary["aggregate"] = aggregate_scores.remote(run, tag)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_volume.commit()
    return summary


BUCKETS = {
    "v3": ("overall_accuracy", "overall_non_live", "overall_live", "overall_multi_turn"),
    "v4": (
        "overall_accuracy",
        "overall_non_live",
        "overall_live",
        "overall_multi_turn",
        "overall_agentic",
        "overall_hallucination",
    ),
}


def checkpoint_ready(tag: str) -> bool:
    """An agent-efficiency student is complete once export has written its provenance file (its last step)."""
    if not tag.startswith(AGENT_EFF_PREFIX):
        return True
    return (Path(resolve_model(tag).path) / AGENT_EFF_PROVENANCE).exists()


EVALUATION_CLAIM = "evaluation_claim.json"


@app.function(
    image=lane_image,
    cpu=1,
    memory=2_048,
    timeout=86_400,
    volumes={REMOTE_CHECKPOINT_ROOT: checkpoint_volume, REMOTE_RESULTS_ROOT: results_volume},
)
def evaluate_when_ready(jobs: list[dict], max_wait_hours: float = 12.0, poll_seconds: int = 60) -> dict:
    """Start each model's evaluation as soon as its checkpoint is complete, then wait for all of them.

    Runs server-side, so the local client may disconnect. Models whose checkpoints do not appear
    within ``max_wait_hours`` are reported as missing rather than failing the others. Before starting
    a model, a claim file is committed to its results directory; a model whose directory already holds
    a claim or a manifest is skipped (``claimed``), so a restarted input never starts a second
    evaluation of the same model. To rerun a model, remove its results directory first.
    """
    deadline = time.monotonic() + max_wait_hours * 3_600
    waiting, calls, claimed = list(jobs), {}, []
    while waiting and time.monotonic() < deadline:
        checkpoint_volume.reload()
        results_volume.reload()
        for job in [job for job in waiting if checkpoint_ready(job["tag"])]:
            waiting.remove(job)
            root = Path(REMOTE_RESULTS_ROOT) / job["run_id"] / job["tag"]
            if (root / EVALUATION_CLAIM).exists() or (root / "manifest.json").exists():
                print(f"[{job['tag']}] already claimed by an earlier evaluation; not starting another", flush=True)
                claimed.append(job["tag"])
                continue
            root.mkdir(parents=True, exist_ok=True)
            (root / EVALUATION_CLAIM).write_text(json.dumps({"time": time.time()}) + "\n", encoding="utf-8")
            results_volume.commit()
            print(f"[{job['tag']}] checkpoint complete; evaluating", flush=True)
            calls[job["tag"]] = serve_and_evaluate.spawn(job)
        if waiting:
            time.sleep(poll_seconds)
    summaries, errors = [], {}
    for tag, call in calls.items():
        try:
            summaries.append(call.get())
        except Exception as error:  # one failed model must not hide the others' results
            errors[tag] = f"{type(error).__name__}: {error}"
    print_comparison([summary for summary in summaries if summary.get("aggregate")])
    return {"summaries": summaries, "errors": errors, "claimed": claimed, "missing": [job["tag"] for job in waiting]}


def print_comparison(summaries: list[dict]) -> None:
    aggregates = {s["tag"]: s.get("aggregate") for s in summaries}
    if not all(a and a.get("complete") for a in aggregates.values()):
        return
    tags = list(aggregates)
    print("\n" + "bucket".ljust(24) + "".join(tag.rjust(10) for tag in tags))
    for bucket in BUCKETS[PROTOCOL.bfcl_version]:
        print(bucket.ljust(24) + "".join(f"{aggregates[tag][bucket]['accuracy']:10.4f}" for tag in tags))


@app.local_entrypoint()
def main(
    models: str = "base,opd",
    categories: str = "",
    smoke_samples: int = 0,
    seed: int = 0,
    wait_for_checkpoints: bool = False,
    max_wait_hours: float = 12.0,
) -> None:
    """Evaluate each model on BFCL, concurrently; ``seed`` selects the decoding-seed replicate.

    ``--wait-for-checkpoints`` hands the run to a server-side function that starts each model once its
    agent-efficiency checkpoint is exported, and returns immediately; models whose checkpoints do not appear
    within ``--max-wait-hours`` are reported as missing.
    """

    tags = [tag.strip() for tag in models.split(",") if tag.strip()]
    for tag in tags:
        resolve_model(tag)
    selected = [category.strip() for category in categories.split(",") if category.strip()]
    run = run_id(smoke_samples, replace(PROTOCOL, seed=seed))
    version = PROTOCOL.bfcl_version
    print(f"run {run}: models={tags} categories={selected or f'all {version}'} smoke_samples={smoke_samples}", flush=True)
    job = {"run_id": run, "categories": selected, "smoke_samples": smoke_samples, "seed": seed}
    jobs = [{"tag": tag, **job} for tag in tags]
    if wait_for_checkpoints:
        call = evaluate_when_ready.spawn(jobs, max_wait_hours)
        print(f"evaluations run server-side as each checkpoint completes: {call.object_id}", flush=True)
        return
    calls = [serve_and_evaluate.spawn(job) for job in jobs]
    summaries = [call.get() for call in calls]
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print_comparison(summaries)
