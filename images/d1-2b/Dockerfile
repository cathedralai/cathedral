# CPU-only d1-2B for Cathedral standard and sealed sandboxes.
# GGUF is published but llama.cpp support is pending PR 31275.
# Image uses the published BF16 safetensors via Transformers.
# GPU is not required. BF16 on CPU needs extra memory.

FROM python:3.12-slim-bookworm

ARG HF_REPO=LiquidAI/d1-2B

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/d1

RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch \
    && pip install --no-cache-dir "transformers>=5.19" huggingface_hub pillow

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${HF_REPO}', local_dir='/models')"

COPY serve.py /opt/d1/serve.py
COPY LICENSE /opt/d1/LICENSE

EXPOSE 8080
ENV D1_HF_REPO=/models \
    D1_HOST=0.0.0.0 \
    D1_PORT=8080 \
    HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/opt/d1/serve.py"]
