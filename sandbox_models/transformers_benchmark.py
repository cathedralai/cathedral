"""CPU smoke path for suite models that ship as Transformers checkpoints.

Numbers are from this host, not a sealed VM. Torch and transformers are
optional and are not package dependencies. Import this module only when
those packages are installed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .catalog import SandboxModel, get_model
from .cpu_benchmark import DEFAULT_PROMPT, host_info, write_red_png, write_sine_wav


def run_transformers_benchmark(
    *,
    model_id: str,
    cache_dir: Path,
    prompt: str = DEFAULT_PROMPT,
) -> dict[str, object]:
    """Load a published HF repo on CPU and run one prompt."""
    model = get_model(model_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if model.family == "d1":
        return _run_d1(model, cache_dir, started)
    return _run_causal(model, cache_dir, started, prompt)


def _run_causal(
    model: SandboxModel,
    cache_dir: Path,
    started: float,
    prompt: str,
) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(
        model.hf_repo, trust_remote_code=True, cache_dir=str(cache_dir)
    )
    loaded = AutoModelForCausalLM.from_pretrained(
        model.hf_repo,
        trust_remote_code=True,
        dtype=torch.float32,
        cache_dir=str(cache_dir),
    ).to("cpu")
    inputs = tokenizer(prompt, return_tensors="pt")
    first = time.monotonic()
    output = loaded.generate(inputs.input_ids, max_new_tokens=32, do_sample=False)
    finished = time.monotonic()
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    new_tokens = int(output.shape[-1] - inputs.input_ids.shape[-1])
    return {
        **host_info(),
        "model_id": model.id,
        "model": model.hf_repo,
        "runtime": "transformers",
        "prompt": prompt,
        "generated_text": text,
        "time_to_first_token_seconds": first - started,
        "boot_to_first_token_seconds": first - started,
        "tokens_per_second": new_tokens / max(finished - first, 1e-6),
        "eval_tokens": new_tokens,
        "weight_bytes": model.weight_bytes,
        "wall_seconds": finished - started,
        "modalities": {"text": {"ok": True, "prompt": prompt}},
    }


def _run_d1(model: SandboxModel, cache_dir: Path, started: float) -> dict[str, object]:
    loaded = AutoModel.from_pretrained(
        model.hf_repo,
        trust_remote_code=True,
        dtype=torch.float32,
        cache_dir=str(cache_dir),
    ).to("cpu")
    questions = {
        "refund": {
            "type": "noul",
            "instructions": "Is the customer asking for a refund?",
        }
    }
    state = "I was charged twice. Please refund one charge."
    out = loaded.system_one(state, questions)
    answers = out.get("answers") if isinstance(out, dict) else out
    first = time.monotonic()
    modalities: dict[str, object] = {"text": {"ok": True, "answers": answers}}
    if "image" in model.sandbox_input_modalities:
        modalities["image"] = _d1_image_smoke(loaded, cache_dir)
    if "audio" in model.sandbox_input_modalities:
        modalities["audio"] = _d1_audio_smoke(loaded, cache_dir)
    return {
        **host_info(),
        "model_id": model.id,
        "model": model.hf_repo,
        "runtime": f"transformers {model.min_runtime_version}+",
        "prompt": state,
        "generated_text": json.dumps(answers, default=str),
        "time_to_first_token_seconds": first - started,
        "boot_to_first_token_seconds": first - started,
        "tokens_per_second": None,
        "eval_tokens": 0,
        "weight_bytes": model.weight_bytes,
        "wall_seconds": time.monotonic() - started,
        "modalities": modalities,
    }


def _d1_image_smoke(loaded: Any, cache_dir: Path) -> dict[str, object]:
    png = write_red_png(cache_dir / "smoke-red.png")
    try:
        image = Image.open(png)
        out = loaded.system_one(
            None,
            {
                "red": {
                    "type": "noul",
                    "instructions": "Is the picture mostly red?",
                }
            },
            media=[image],
        )
        answers = out.get("answers") if isinstance(out, dict) else out
        return {"ok": True, "answers": answers}
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        return {"ok": False, "error": str(exc)[-1500:]}


def _d1_audio_smoke(loaded: Any, cache_dir: Path) -> dict[str, object]:
    wav = write_sine_wav(cache_dir / "smoke-tone.wav")
    try:
        out = loaded.system_one(
            None,
            {
                "speech": {
                    "type": "noul",
                    "instructions": "Does someone speak in the recording?",
                }
            },
            media=[{"type": "audio", "path": str(wav)}],
        )
        answers = out.get("answers") if isinstance(out, dict) else out
        return {"ok": True, "answers": answers}
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        return {"ok": False, "error": str(exc)[-1500:]}


def write_report(report: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2) + "\n", encoding="utf-8")
