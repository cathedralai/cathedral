"""Gated sealed-sandbox LFM3 probe.

This module never creates a paid sandbox unless every gate below is set:

* ``CATHEDRAL_SEALED_LFM3_TEST=1``
* ``CATHEDRAL_API_KEY`` (caller-supplied; this repo ships no key)
* ``CATHEDRAL_SSH_PUBLIC_KEY``

It does not run in CI. The live sealed run waits for Fred to approve spend.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Mapping
from urllib.parse import urljoin

from .catalog import LFM3_1_7B_ID, SandboxTier, resolve_create


ENABLE_ENV = "CATHEDRAL_SEALED_LFM3_TEST"
API_KEY_ENV = "CATHEDRAL_API_KEY"
API_URL_ENV = "CATHEDRAL_API_URL"
SSH_KEY_ENV = "CATHEDRAL_SSH_PUBLIC_KEY"
DEFAULT_API_URL = "https://cathedral.computer"
PROMPT = "Reply with the single word pong."


def sealed_test_enabled() -> bool:
    return os.environ.get(ENABLE_ENV) == "1" and bool(os.environ.get(API_KEY_ENV))


def skip_reason() -> str:
    return (
        f"set {ENABLE_ENV}=1 and {API_KEY_ENV} to run the paid sealed LFM3 "
        "probe; this test skips otherwise"
    )


def _api_url() -> str:
    return os.environ.get(API_URL_ENV, DEFAULT_API_URL).rstrip("/") + "/"


def _headers(*, idempotency_key: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {os.environ[API_KEY_ENV]}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _request(
    method: str,
    path: str,
    *,
    body: Mapping[str, object] | None = None,
    idempotency_key: str | None = None,
    timeout: float = 60,
) -> tuple[int, dict[str, object]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        urljoin(_api_url(), path.lstrip("/")),
        data=data,
        headers=_headers(idempotency_key=idempotency_key),
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            return resp.status, payload
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            payload = {"error": raw.decode("utf-8", errors="replace")}
        return exc.code, payload


def create_sealed_lfm3_body() -> dict[str, object]:
    ssh = os.environ.get(SSH_KEY_ENV)
    return resolve_create(
        LFM3_1_7B_ID,
        SandboxTier.SEALED,
        name="lfm3-1.7b-sealed-probe",
        lifetime_seconds=3600,
        max_spend_usd="1.00",
        ssh_public_key=ssh,
    )


def run_sealed_lfm3_probe(
    *,
    poll_seconds: float = 5,
    ready_timeout_seconds: float = 900,
    prompt: str = PROMPT,
) -> dict[str, object]:
    """Create a sealed sandbox, wait until running, ask one prompt, delete.

    Returns start time (create to first token) and tokens per second.
    """
    if not sealed_test_enabled():
        raise RuntimeError(skip_reason())

    created_at = time.monotonic()
    status, created = _request(
        "POST",
        "/v1/workers",
        body=create_sealed_lfm3_body(),
        idempotency_key=f"lfm3-sealed-{int(time.time())}",
        timeout=120,
    )
    if status >= 300:
        raise RuntimeError(f"sealed create failed HTTP {status}: {created}")
    worker_id = str(created.get("id") or created.get("worker_id") or "")
    if not worker_id:
        raise RuntimeError(f"sealed create missing id: {created}")

    try:
        deadline = time.monotonic() + ready_timeout_seconds
        worker: dict[str, object] = created
        while time.monotonic() < deadline:
            status, worker = _request("GET", f"/v1/workers/{worker_id}", timeout=60)
            if status >= 300:
                raise RuntimeError(f"sealed poll failed HTTP {status}: {worker}")
            state = str(worker.get("state") or "")
            if state == "running":
                break
            if state in {"failed", "deleted", "stopped"}:
                raise RuntimeError(f"sealed sandbox ended in {state}: {worker}")
            time.sleep(poll_seconds)
        else:
            raise TimeoutError("sealed sandbox did not reach running")

        command = {
            "command": (
                "curl -sS http://127.0.0.1:8080/v1/chat/completions "
                "-H 'Content-Type: application/json' "
                "-d '"
                + json.dumps(
                    {
                        "model": LFM3_1_7B_ID,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 32,
                        "temperature": 0.1,
                    }
                )
                + "'"
            ),
            "cwd": ".",
            "timeout_seconds": 90,
        }
        prompt_at = time.monotonic()
        status, result = _request(
            "POST",
            f"/v1/workers/{worker_id}/commands",
            body=command,
            idempotency_key=f"lfm3-cmd-{int(time.time())}",
            timeout=120,
        )
        if status >= 300:
            raise RuntimeError(f"sealed command failed HTTP {status}: {result}")
        first_token_at = time.monotonic()
        stdout = str(result.get("stdout") or "")
        completion = _parse_completion(stdout)
        usage = (
            completion.get("usage") if isinstance(completion.get("usage"), dict) else {}
        )
        completion_tokens = int(usage.get("completion_tokens") or 0)
        eval_seconds = max(first_token_at - prompt_at, 1e-6)
        return {
            "worker_id": worker_id,
            "state": worker.get("state"),
            "boot_to_first_token_seconds": first_token_at - created_at,
            "prompt_to_first_token_seconds": first_token_at - prompt_at,
            "tokens_per_second": completion_tokens / eval_seconds
            if completion_tokens
            else None,
            "completion_tokens": completion_tokens,
            "text": _completion_text(completion),
            "host": "sealed",
        }
    finally:
        _request("DELETE", f"/v1/workers/{worker_id}", timeout=60)


def _parse_completion(stdout: str) -> dict[str, object]:
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return {"raw": stdout}
    if isinstance(parsed, dict):
        return parsed
    return {"raw": stdout}


def _completion_text(completion: Mapping[str, object]) -> str:
    choices = completion.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text
    raw = completion.get("raw")
    return raw if isinstance(raw, str) else ""
