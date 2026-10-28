# CPU-only LFM3-1.7B-Base for Cathedral standard and sealed sandboxes.
# No public GGUF. Image uses the published BF16 safetensors via Transformers.
# GPU is not required. Extra memory versus the Q4 chat image.

FROM python:3.12-slim-bookworm

ARG HF_REPO=LiquidAI/LFM3-1.7B-Base

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lfm3

RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch \
    && pip install --no-cache-dir "transformers>=4.51" huggingface_hub

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${HF_REPO}', local_dir='/models')"

COPY serve.py /opt/lfm3/serve.py
COPY LICENSE /opt/lfm3/LICENSE
COPY LICENSE /models/LICENSE

EXPOSE 8080
ENV LFM3_HF_REPO=/models \
    LFM3_HOST=0.0.0.0 \
    LFM3_PORT=8080 \
    HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/opt/lfm3/serve.py"]
