# LFM3-1.7B sandbox model option

This is not live. The catalog and image live in this repository. The hotel
broker in cathedral-sandbox and the private service layer must consume the
same template name before a customer can pick it.

Subnet copy in this document is SN94.

## Model

| Field | Value |
|---|---|
| Family | LFM3 |
| Size | LFM3-1.7B (1.73B parameters) |
| Hugging Face (BF16 / Transformers) | [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) |
| Hugging Face (GGUF) | [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF) |
| File used | `LFM3-1.7B-Q4_K_M.gguf` (about 1.10 GiB) |
| Runtime | llama.cpp `b11938` or newer (LFM3 architecture landed in [PR 31252](https://github.com/ggml-org/llama.cpp/pull/31252)) |
| GPU | Not required |
| Suggested memory | 4 GiB (Standard `room-1x4` shape). Sealed smallest SKU is 4 vCPU / 16 GiB. |
| Suggested disk | 10 GiB |
| Image (not published) | `cathedral/lfm3-1.7b:q4_k_m`, built from `images/lfm3-1.7b` |
| License | [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE) |

## How a caller picks it

The existing create contracts stay in place. There is no new request field.

Standard (`POST /v1/sandboxes`):

```json
{"template": "lfm3-1.7b", "network": "deny_all", "lifetime_seconds": 3600, "max_spend_usd": "1.00"}
```

Production currently refuses new Standard creates with `503 sealed_only`. This
option is wired for that contract. It does not reopen Standard sales.

Sealed (`POST /v1/workers`), AMD SEV-SNP confidential VMs, live sealed tier:

```json
{
  "profile": "custom.v1",
  "name": "lfm3-1.7b-sealed",
  "lifetime": {"mode": "bounded_service", "reuse": "allowed", "max_runtime_minutes": 60},
  "workload": {"image": "cathedral/lfm3-1.7b:q4_k_m"},
  "resources": {"hardware_class": "tdx_cpu", "cpu": 4, "memory_gib": 16},
  "budget": {"max_spend_usd": 1, "auto_stop": true}
}
```

`sandbox_models.resolve_create("lfm3-1.7b", SandboxTier.STANDARD|SEALED)` builds
those bodies.

## License

The open license allows commercial use only while the Legal Entity's annual
revenue is under $10,000,000. At or above that threshold, commercial use is
not licensed. Contact sales@liquid.ai for a commercial license. Hosting the
weights for paying customers inside a compute product is Commercial Use.

Quoted terms are in the pull request description.

## CPU benchmark on this cloud VM

Numbers from `python scripts/lfm3_cpu_benchmark.py` and `llama-bench` on the
cloud agent VM (4 vCPU Intel Xeon, 15 GiB RAM). They are not sealed-host
numbers. The live sealed run waits for spend approval.

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
| Estimated image size | about 1.2 GiB (debian-slim + llama.cpp + GGUF). Image is not built or published. |

```bash
python scripts/lfm3_cpu_benchmark.py --report docs/evidence/lfm3-1.7b-cpu-benchmark.json
```

Raw JSON: `docs/evidence/lfm3-1.7b-cpu-benchmark.json`.

## Sealed test (paid, opt-in)

The test boots a sealed sandbox, loads the model, answers one prompt, and
reports boot-to-first-token and tokens per second. It skips unless every
gate is set. It does not run in CI.

```bash
export CATHEDRAL_SEALED_LFM3_TEST=1
export CATHEDRAL_API_KEY=...
export CATHEDRAL_SSH_PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)"
pytest -q tests/thin/test_sandbox_models.py -k live_sealed
```

Do not point a funded key at production until Fred approves the spend.

## Follow-up outside this repository

- cathedral-sandbox `broker/compat.go`: add `lfm3-1.7b` to `compatTemplates`
  and return it from `compatCatalog` with image `cathedral/lfm3-1.7b:q4_k_m`.
- Private service layer (`GET /v1/sandboxes/catalog` and console create):
  add the same template. This repository cannot see that code.
- Publish the image to a public registry before a sealed `workload.image`
  create can start.
