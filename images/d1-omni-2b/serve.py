#!/usr/bin/env python3
"""CPU decision server for d1-Omni-2B. Same contract as d1-2B."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoModel

MODEL_ID = os.environ.get("D1_HF_REPO", "/models")
HOST = os.environ.get("D1_HOST", "0.0.0.0")
PORT = int(os.environ.get("D1_PORT", "8080"))
ALIAS = os.environ.get("D1_ALIAS", "d1-omni-2b")

MODEL = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=torch.float32,
).to("cpu")
MODEL.eval()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/health", "/v1/models"}:
            self._json(200, {"status": "ok", "model": ALIAS})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return
        if self.path != "/v1/systemone":
            self._json(404, {"error": "not found"})
            return
        state = request.get("state")
        questions = request.get("questions") or {}
        if not isinstance(questions, dict):
            self._json(400, {"error": "questions must be an object"})
            return
        out: Any = MODEL.system_one(state, questions)
        if not isinstance(out, dict):
            self._json(500, {"error": "model returned a non-object"})
            return
        self._json(200, out)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
