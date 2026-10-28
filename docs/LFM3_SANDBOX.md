# LFM3 suite sandbox model options

This is not live. The catalog and images live in this repository. The hotel
broker in cathedral-sandbox and the private service layer must consume the
same template names before a customer can pick them.

Subnet copy in this document is SN94.

## Suite

Confirmed on Hugging Face on this cloud VM. Repo ids were not invented.
Every published checkpoint uses LFM Open License v1.0. The VL LICENSE file
differs by a trailing space on the title line only. The terms match,
including the $10,000,000 annual revenue cap.

`LiquidAI/LFM3-1.7B-Base-GGUF` is not published (Hugging Face 401). That
format is listed as not available and is skipped.

| Template | HF (BF16) | GGUF | File used | Bytes | Runtime | Sandbox in | Sandbox out | GPU | Memory | Image (not published) |
|---|---|---|---|---|---|---|---|---|---|---|
| `lfm3-1.7b` | [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) | [LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF) | `LFM3-1.7B-Q4_K_M.gguf` | 1,099,627,808 | llama.cpp `b11938` (also vLLM and Transformers on BF16/GPU) | text | text | No | 4 GiB | `cathedral/lfm3-1.7b:q4_k_m` |
| `lfm3-1.7b-base` | [LiquidAI/LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) | not published | `model.safetensors` | 3,457,635,448 | Transformers (vLLM can serve BF16 on GPU) | text | text | No | 8 GiB (BF16 needs extra RAM) | `cathedral/lfm3-1.7b-base:bf16` |
| `lfm3-vl-2b` | [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) | [LFM3-VL-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF) | `LFM3-VL-2B-Q4_K_M.gguf` plus `mmproj-LFM3-VL-2B-Q8_0.gguf` | 1,099,627,808 + 203,031,872 | llama.cpp `b11938` (Transformers; vLLM on GPU is in progress) | text, image | text | No | 6 GiB | `cathedral/lfm3-vl-2b:q4_k_m` |
| `lfm3-audio-2b` | [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) | [LFM3-Audio-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF) | `LFM3-Audio-2B-Q4_K_M.gguf` plus `mmproj-LFM3-Audio-2B-Q8_0.gguf` | 1,099,626,016 + 337,862,080 | llama.cpp `b11941` (also Transformers) | text, audio | text, speech | No | 6 GiB | `cathedral/lfm3-audio-2b:q4_k_m` |
| `lfm3-omni-2b` | [LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) | [LFM3-Omni-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) | `LFM3-Omni-2B-Q4_K_M.gguf` plus `mmproj-LFM3-Omni-2B-Q8_0.gguf` | 1,099,630,080 + 540,892,768 | llama.cpp `b11941` (also Transformers) | text, image, audio | text, speech | No | 8 GiB | `cathedral/lfm3-omni-2b:q4_k_m` |
| `d1-2b` | [LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B) | [d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF) | image uses BF16 `model.safetensors` (GGUF `d1-2B-Q4_K_M.gguf` is 1,099,629,888 bytes) | 3,836,417,920 | Transformers `>=5.19`. llama.cpp pending [PR 31275](https://github.com/ggml-org/llama.cpp/pull/31275) | text, image | decision | No | 12 GiB | `cathedral/d1-2b:bf16` |
| `d1-omni-2b` | [LiquidAI/d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B) | [d1-Omni-2B-GGUF](https://huggingface.co/LiquidAI/d1-Omni-2B-GGUF) | image uses BF16 `model.safetensors`. GGUF has Q8_0 only (1,846,241,184 bytes), no Q4_K_M | 4,333,941,640 | Transformers `>=5.19`. llama.cpp pending [PR 31275](https://github.com/ggml-org/llama.cpp/pull/31275) | text, image, audio | decision | No | 14 GiB | `cathedral/d1-omni-2b:bf16` |

Model cards also offer video input on VL, Omni, and d1. llama.cpp does not
accept LFM3 video yet. The sandbox recipes do not wire video.

d1 models return calibrated answers in one forward pass. They do not generate
tokens. The sandbox speaks `/v1/systemone`, not chat completions.

## How a caller picks one

The existing create contracts stay in place. There is no new request field.

Standard (`POST /v1/sandboxes`):

```json
{"template": "lfm3-vl-2b", "network": "deny_all", "lifetime_seconds": 3600, "max_spend_usd": "1.00"}
```

Production currently refuses new Standard creates with `503 sealed_only`. These
options are wired for that contract. They do not reopen Standard sales.

Sealed (`POST /v1/workers`), AMD SEV-SNP confidential VMs, live sealed tier:

```json
{
  "profile": "custom.v1",
  "name": "lfm3-vl-2b-sealed",
  "lifetime": {"mode": "bounded_service", "reuse": "allowed", "max_runtime_minutes": 60},
  "workload": {"image": "cathedral/lfm3-vl-2b:q4_k_m"},
  "resources": {"hardware_class": "tdx_cpu", "cpu": 4, "memory_gib": 16},
  "budget": {"max_spend_usd": 1, "auto_stop": true}
}
```

`sandbox_models.resolve_create("<template>", SandboxTier.STANDARD|SEALED)`
builds those bodies. Sealed memory is at least 16 GiB.

## License

The open license allows commercial use only while the Legal Entity's annual
revenue is under $10,000,000. At or above that threshold, commercial use is
not licensed. Contact sales@liquid.ai for a commercial license. Hosting the
weights for paying customers inside a compute product is Commercial Use.

Quoted terms are in the pull request description. No suite model uses a
different license.

## CPU benchmark on this cloud VM

Numbers from `python scripts/lfm3_cpu_benchmark.py` and
`python scripts/lfm3_suite_cpu_benchmark.py` on the cloud agent VM (4 vCPU
Intel Xeon, 15 GiB RAM). They are not sealed-host numbers. The live sealed
run waits for spend approval.

### LFM3-1.7B (already measured)

| Measurement | Value |
|---|---|
| GGUF on disk | 1,099,627,808 bytes (1.02 GiB) |
| One prompt ("Reply with the single word pong.") | answer `PONG` |
| Process start to first token | 4.94 s (load plus prefill) |
| Generation | 21.5 tokens/s |
| Prompt processing (same run) | 105.1 tokens/s |
| llama-bench pp32 | 279.18 tokens/s |
| llama-bench tg32 | 21.45 tokens/s |
| Suggested sandbox memory | 4 GiB |
| Estimated image size | about 1.2 GiB (debian-slim plus llama.cpp plus GGUF). Image is not built or published. |

### Other published models

Fill from `docs/evidence/<template>-cpu-benchmark.json` after the cloud VM
suite run. Until that file exists for a model, treat CPU numbers as not yet
measured on this VM.

```bash
python scripts/lfm3_cpu_benchmark.py --model-id lfm3-1.7b --report docs/evidence/lfm3-1.7b-cpu-benchmark.json
python scripts/lfm3_suite_cpu_benchmark.py
```

Raw JSON: `docs/evidence/lfm3-1.7b-cpu-benchmark.json` and
`docs/evidence/lfm3-suite-cpu-benchmark.json`.

## Sealed test (paid, opt-in)

The test boots a sealed sandbox for `lfm3-1.7b` only, loads the model, answers
one prompt, and reports boot-to-first-token and tokens per second. It skips
unless every gate is set. It does not run in CI.

```bash
export CATHEDRAL_SEALED_LFM3_TEST=1
export CATHEDRAL_API_KEY=...
export CATHEDRAL_SSH_PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)"
pytest -q tests/thin/test_sandbox_models.py -k live_sealed
```

Do not point a funded key at production until Fred approves the spend.

## Follow-up outside this repository

- cathedral-sandbox `broker/compat.go`: add each template in the suite table
  to `compatTemplates` and `compatCatalog` with the matching image.
- Private service layer (`GET /v1/sandboxes/catalog` and console create):
  add the same templates. This repository cannot see that code.
- Publish each image to a public registry before a sealed `workload.image`
  create can start.
