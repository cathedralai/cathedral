#!/usr/bin/env python3
"""CPU completion server for LFM3-1.7B-Base."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("LFM3_HF_REPO", "/models")
HOST = os.environ.get("LFM3_HOST", "0.0.0.0")
PORT = int(os.environ.get("LFM3_PORT", "8080"))

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=torch.float32,
).to("cpu")
MODEL.eval()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/health", "/v1/models"}:
            self._json(200, {"status": "ok", "model": "lfm3-1.7b-base"})
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
        if self.path == "/v1/chat/completions":
            messages = request.get("messages") or []
            prompt = ""
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if isinstance(last, dict):
                    prompt = str(last.get("content") or "")
            max_tokens = int(request.get("max_tokens") or 32)
            text = _generate(prompt, max_tokens)
            self._json(
                200,
                {
                    "id": "lfm3-1.7b-base",
                    "object": "chat.completion",
                    "choices": [
                        {"message": {"role": "assistant", "content": text}}
                    ],
                    "usage": {"completion_tokens": 0},
                },
            )
            return
        if self.path == "/v1/completions":
            prompt = str(request.get("prompt") or "")
            max_tokens = int(request.get("max_tokens") or 32)
            text = _generate(prompt, max_tokens)
            self._json(
                200,
                {
                    "id": "lfm3-1.7b-base",
                    "object": "text_completion",
                    "choices": [{"text": text}],
                },
            )
            return
        self._json(404, {"error": "not found"})


def _generate(prompt: str, max_tokens: int) -> str:
    inputs = TOKENIZER(prompt, return_tensors="pt")
    output = MODEL.generate(
        inputs.input_ids, max_new_tokens=max_tokens, do_sample=False
    )
    return TOKENIZER.decode(output[0], skip_special_tokens=True)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
