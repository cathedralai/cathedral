"""Local CPU benchmark for LFM3 suite models.

Numbers are from this host, not a sealed VM. Label every report
``cloud_vm_not_sealed``.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
import tarfile
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Mapping

from .catalog import LFM3_1_7B_ID, SandboxModel, get_model

PROMPT_GEN_RE = re.compile(
    r"Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s"
)


LLAMA_CPP_TAG = "b11938"
DEFAULT_PROMPT = "Reply with the single word pong."
DEFAULT_MAX_TOKENS = 32
DEFAULT_IMAGE_PROMPT = "What is the dominant color in this image? Reply with one word."
DEFAULT_AUDIO_PROMPT = "Perform ASR in English."


def default_cache_dir() -> Path:
    raw = os.environ.get("LFM3_CACHE_DIR")
    return Path(raw) if raw else Path("/tmp/lfm3-bench")


def llama_cpp_archive_url(tag: str) -> str:
    return (
        "https://github.com/ggml-org/llama.cpp/releases/download/"
        f"{tag}/llama-{tag}-bin-ubuntu-x64.tar.gz"
    )


def hf_resolve_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def unpack_llama_cpp(archive: Path, dest_dir: Path, tag: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / ".unpacked"
    if not marker.exists() or marker.read_text(encoding="utf-8").strip() != tag:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest_dir)
        marker.write_text(tag + "\n", encoding="utf-8")
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


def host_info() -> dict[str, object]:
    cpu = "unknown"
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    mem_kib = 0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            mem_kib = int(line.split()[1])
            break
    return {
        "host": "cloud_vm_not_sealed",
        "host_cpu": cpu,
        "host_vcpu": os.cpu_count() or 0,
        "host_memory_gib": round(mem_kib / (1024 * 1024)),
        "note": "Measured on the cloud agent VM. Not a sealed AMD SEV-SNP host.",
    }


def write_red_png(path: Path, size: int = 64) -> Path:
    """Write a tiny solid red PNG without extra dependencies."""
    raw = b"".join(b"\x00" + (b"\xff\x00\x00" * size) for _ in range(size))
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return path


def write_sine_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> Path:
    """Write a short 440 Hz mono 16-bit PCM WAV."""
    n = int(seconds * rate)
    samples = bytearray()
    for i in range(n):
        value = int(0.3 * 32767 * math.sin(2 * math.pi * 440 * i / rate))
        samples += struct.pack("<h", value)
    data = bytes(samples)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        rate,
        rate * 2,
        2,
        16,
        b"data",
        len(data),
    )
    path.write_bytes(header + data)
    return path


def run_benchmark(
    *,
    cache_dir: Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model_id: str = LFM3_1_7B_ID,
    include_llama_bench: bool = True,
    include_modalities: bool = True,
) -> dict[str, object]:
    model = get_model(model_id)
    if (
        model.runtime != "llama.cpp"
        or not model.gguf_filename
        or not model.hf_gguf_repo
    ):
        raise ValueError(
            f"{model_id} is not a llama.cpp GGUF target; use the transformers path"
        )
    cache = cache_dir or default_cache_dir()
    tag = model.min_runtime_version
    archive = download(
        llama_cpp_archive_url(tag), cache / f"llama-{tag}-ubuntu-x64.tar.gz"
    )
    unpack_dir = cache / f"llama-{tag}"
    unpack_llama_cpp(archive, unpack_dir, tag)
    cli = find_binary(unpack_dir, "llama-cli")
    weights = download(
        hf_resolve_url(model.hf_gguf_repo, model.gguf_filename),
        cache / model.gguf_filename,
    )
    mmproj: Path | None = None
    if model.mmproj_filename:
        mmproj = download(
            hf_resolve_url(model.hf_gguf_repo, model.mmproj_filename),
            cache / model.mmproj_filename,
        )
    env = os.environ.copy()
    libdir = str(cli.parent)
    env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    text_report = _run_llama_cli(
        cli=cli,
        weights=weights,
        mmproj=mmproj,
        prompt=prompt,
        max_tokens=max_tokens,
        env=env,
        extra_args=("--reasoning", "off"),
    )
    report: dict[str, object] = {
        **host_info(),
        "model_id": model.id,
        "model": model.hf_gguf_repo,
        "gguf": model.gguf_filename,
        "mmproj": model.mmproj_filename or None,
        "runtime": f"llama.cpp {tag}",
        "prompt": prompt,
        "max_tokens": max_tokens,
        **text_report,
        "weight_bytes": weights.stat().st_size,
        "mmproj_bytes": mmproj.stat().st_size if mmproj else 0,
    }
    if include_llama_bench:
        try:
            bench = find_binary(unpack_dir, "llama-bench")
        except FileNotFoundError:
            report["llama_bench"] = None
        else:
            report["llama_bench"] = run_llama_bench(bench, weights, env)
    modalities: dict[str, object] = {"text": {"ok": True, "prompt": prompt}}
    if include_modalities and mmproj is not None:
        try:
            mtmd = find_binary(unpack_dir, "llama-mtmd-cli")
        except FileNotFoundError:
            mtmd = None
        if "image" in model.sandbox_input_modalities:
            modalities["image"] = _run_image_smoke(
                cli=mtmd or cli,
                weights=weights,
                mmproj=mmproj,
                cache=cache,
                env=env,
            )
        if "audio" in model.sandbox_input_modalities:
            modalities["audio"] = _run_audio_smoke(
                cli=mtmd or cli,
                weights=weights,
                mmproj=mmproj,
                cache=cache,
                env=env,
            )
    report["modalities"] = modalities
    return report


def _run_llama_cli(
    *,
    cli: Path,
    weights: Path,
    mmproj: Path | None,
    prompt: str,
    max_tokens: int,
    env: dict[str, str],
    extra_args: tuple[str, ...] = (),
    extra_media: tuple[str, ...] = (),
) -> dict[str, object]:
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
        *extra_args,
        *extra_media,
    ]
    if mmproj is not None:
        command.extend(["--mmproj", str(mmproj)])
    started = time.monotonic()
    first_token_at: float | None = None
    chunks: list[str] = []
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
        raise RuntimeError(f"{cli.name} failed ({proc.returncode}): {stderr[-2000:]}")
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
        "llama_stderr_timings": timings,
        "llama_cli_timings": {
            "prompt_tps": timings.get("prompt_tps"),
            "generation_tps": timings.get("generation_tps"),
        },
        "wall_seconds": finished - started,
    }


def run_llama_bench(
    bench: Path, weights: Path, env: dict[str, str]
) -> dict[str, object]:
    command = [
        str(bench),
        "-m",
        str(weights),
        "-p",
        "32",
        "-n",
        "32",
        "-t",
        str(os.cpu_count() or 4),
        "-o",
        "json",
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, env=env, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"llama-bench failed ({completed.returncode}): {completed.stderr[-2000:]}"
        )
    payload = json.loads(completed.stdout or "[]")
    if isinstance(payload, dict):
        rows = payload.get("results") or payload.get("data") or [payload]
    else:
        rows = payload
    pp32 = None
    tg32 = None
    model_size = None
    params = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        n_prompt = row.get("n_prompt") or row.get("pp")
        n_gen = row.get("n_gen") or row.get("tg")
        avg = row.get("avg_ts") or row.get("tokens_per_second")
        if n_prompt and not n_gen:
            pp32 = avg
        if n_gen and not n_prompt:
            tg32 = avg
        if n_prompt == 32 and (n_gen in (0, None)):
            pp32 = avg
        if n_gen == 32 and (n_prompt in (0, None)):
            tg32 = avg
        model_size = row.get("size_gib") or row.get("model_size") or model_size
        params = row.get("params") or params
    return {
        "model_size_gib": model_size,
        "params": params,
        "backend": "CPU",
        "threads": os.cpu_count() or 4,
        "pp32_tokens_per_second": pp32,
        "tg32_tokens_per_second": tg32,
        "raw_rows": rows,
    }


def _run_image_smoke(
    *,
    cli: Path,
    weights: Path,
    mmproj: Path,
    cache: Path,
    env: dict[str, str],
) -> dict[str, object]:
    png = write_red_png(cache / "smoke-red.png")
    try:
        result = _run_llama_cli(
            cli=cli,
            weights=weights,
            mmproj=mmproj,
            prompt=DEFAULT_IMAGE_PROMPT,
            max_tokens=16,
            env=env,
            extra_args=("--reasoning", "off"),
            extra_media=("--image", str(png)),
        )
        return {"ok": True, "prompt": DEFAULT_IMAGE_PROMPT, **result}
    except RuntimeError as exc:
        return {"ok": False, "prompt": DEFAULT_IMAGE_PROMPT, "error": str(exc)[-1500:]}


def _run_audio_smoke(
    *,
    cli: Path,
    weights: Path,
    mmproj: Path,
    cache: Path,
    env: dict[str, str],
) -> dict[str, object]:
    wav = write_sine_wav(cache / "smoke-tone.wav")
    try:
        result = _run_llama_cli(
            cli=cli,
            weights=weights,
            mmproj=mmproj,
            prompt="<__media__>",
            max_tokens=32,
            env=env,
            extra_args=("--reasoning", "off", "-sys", DEFAULT_AUDIO_PROMPT),
            extra_media=("--audio", str(wav)),
        )
        return {"ok": True, "prompt": DEFAULT_AUDIO_PROMPT, **result}
    except RuntimeError as exc:
        return {"ok": False, "prompt": DEFAULT_AUDIO_PROMPT, "error": str(exc)[-1500:]}


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
    left = line.split("/")[-1]
    return float(left.strip().split()[0])


def write_report(report: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2) + "\n", encoding="utf-8")


def evidence_path(model: SandboxModel) -> Path:
    return Path("docs/evidence") / f"{model.id}-cpu-benchmark.json"


def skipped_report(model: SandboxModel, reason: str) -> dict[str, object]:
    return {
        **host_info(),
        "model_id": model.id,
        "model": model.hf_repo or model.hf_gguf_repo,
        "runtime": model.runtime,
        "skipped": True,
        "skip_reason": reason,
        "weight_bytes": model.weight_bytes,
        "note": "Measured on the cloud agent VM. Not a sealed AMD SEV-SNP host.",
    }
