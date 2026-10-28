"""Local CPU benchmark for LFM3-1.7B. Numbers are from this host, not a sealed VM."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Mapping

PROMPT_GEN_RE = re.compile(
    r"Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s"
)


LLAMA_CPP_TAG = "b11938"
LLAMA_CPP_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_CPP_TAG}/llama-{LLAMA_CPP_TAG}-bin-ubuntu-x64.tar.gz"
)
GGUF_URL = (
    "https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-Q4_K_M.gguf"
)
DEFAULT_PROMPT = "Reply with the single word pong."
DEFAULT_MAX_TOKENS = 32


def default_cache_dir() -> Path:
    raw = os.environ.get("LFM3_CACHE_DIR")
    return Path(raw) if raw else Path("/tmp/lfm3-bench")


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def unpack_llama_cpp(archive: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / ".unpacked"
    if not marker.exists():
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest_dir)
        marker.write_text(LLAMA_CPP_TAG + "\n", encoding="utf-8")
    for name in ("llama-cli", "llama-server"):
        found = shutil.which(name, path=_bin_path(dest_dir))
        if found:
            return Path(found).parent
    raise FileNotFoundError(f"llama-cli not found under {dest_dir}")


def _bin_path(root: Path) -> str:
    parts = [str(root), str(root / "bin"), str(root / "build" / "bin")]
    extra = [str(path) for path in root.rglob("llama-cli")]
    return os.pathsep.join(parts + [str(Path(p).parent) for p in extra])


def find_binary(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_file() and os.access(direct, os.X_OK):
        return direct
    for path in root.rglob(name):
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(f"{name} not found under {root}")


def run_benchmark(
    *,
    cache_dir: Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, object]:
    cache = cache_dir or default_cache_dir()
    archive = download(
        LLAMA_CPP_URL, cache / f"llama-{LLAMA_CPP_TAG}-ubuntu-x64.tar.gz"
    )
    unpack_dir = cache / f"llama-{LLAMA_CPP_TAG}"
    unpack_llama_cpp(archive, unpack_dir)
    cli = find_binary(unpack_dir, "llama-cli")
    weights = download(GGUF_URL, cache / "LFM3-1.7B-Q4_K_M.gguf")

    command = [
        str(cli),
        "-m",
        str(weights),
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        "--temp",
        "0.1",
        "--top-k",
        "50",
        "--repeat-penalty",
        "1.1",
        "-st",
        "--no-display-prompt",
        "--reasoning",
        "off",
    ]
    started = time.monotonic()
    first_token_at: float | None = None
    chunks: list[str] = []
    env = os.environ.copy()
    libdir = str(cli.parent)
    env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None
    while True:
        line = proc.stdout.readline()
        if line == "" and proc.poll() is not None:
            break
        if not line:
            continue
        if first_token_at is None and line.strip():
            first_token_at = time.monotonic()
        chunks.append(line)
    stderr = proc.stderr.read()
    finished = time.monotonic()
    if proc.returncode != 0:
        raise RuntimeError(f"llama-cli failed ({proc.returncode}): {stderr[-2000:]}")
    combined = "".join(chunks) + "\n" + stderr
    timings = parse_llama_timings(combined)
    generated = _generated_text("".join(chunks), prompt)
    tokens = int(timings.get("eval_tokens") or 0)
    eval_ms = float(timings.get("eval_ms") or 0)
    tokens_per_second = timings.get("generation_tps")
    if tokens_per_second is None and tokens and eval_ms:
        tokens_per_second = tokens / (eval_ms / 1000.0)
    elif tokens_per_second is None and tokens and first_token_at is not None:
        tokens_per_second = tokens / max(finished - first_token_at, 1e-6)
    return {
        "host": "cloud_vm_not_sealed",
        "model": "LiquidAI/LFM3-1.7B-GGUF",
        "gguf": "LFM3-1.7B-Q4_K_M.gguf",
        "runtime": f"llama.cpp {LLAMA_CPP_TAG}",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "generated_text": generated,
        "load_seconds": timings.get("load_ms", 0) / 1000.0
        if timings.get("load_ms")
        else None,
        "time_to_first_token_seconds": (
            (float(timings["load_ms"]) + float(timings["prompt_eval_ms"])) / 1000.0
            if timings.get("load_ms") is not None
            and timings.get("prompt_eval_ms") is not None
            else ((first_token_at - started) if first_token_at else None)
        ),
        "boot_to_first_token_seconds": (first_token_at - started)
        if first_token_at
        else None,
        "tokens_per_second": tokens_per_second,
        "eval_tokens": tokens,
        "weight_bytes": weights.stat().st_size,
        "llama_stderr_timings": timings,
        "wall_seconds": finished - started,
    }


def parse_llama_timings(stderr: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    for line in stderr.splitlines():
        match = PROMPT_GEN_RE.search(line)
        if match:
            timings["prompt_tps"] = float(match.group(1))
            timings["generation_tps"] = float(match.group(2))
        if "load time" in line and "=" in line:
            timings["load_ms"] = _ms(line)
        elif "prompt eval time" in line and "=" in line:
            timings["prompt_eval_ms"] = _ms(line)
            timings["prompt_tokens"] = _tokens(line)
        elif "eval time" in line and "prompt eval" not in line and "=" in line:
            timings["eval_ms"] = _ms(line)
            timings["eval_tokens"] = _tokens(line)
    return timings


def _generated_text(stdout: str, prompt: str) -> str:
    marker = f"> {prompt}"
    if marker in stdout:
        after = stdout.split(marker, 1)[1]
        cleaned = []
        for line in after.splitlines():
            text = line.strip()
            if not text or text.startswith("[ Prompt:") or text.startswith("Exiting"):
                continue
            cleaned.append(text)
        return "\n".join(cleaned).strip()
    return stdout.strip()


def _ms(line: str) -> float:
    after = line.split("=", 1)[1]
    number = after.strip().split()[0].replace("ms", "")
    return float(number)


def _tokens(line: str) -> float:
    if "tokens" not in line:
        return 0.0
    # "eval time =    1234.56 ms /    32 tokens"
    left = line.split("/")[-1]
    return float(left.strip().split()[0])


def write_report(report: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2) + "\n", encoding="utf-8")
