# CPU-only LFM3-VL-2B for Cathedral standard and sealed sandboxes.
# llama.cpp b11938 is the first release with LFM3-VL image support.
# Video input is not supported in llama.cpp yet. GPU is not required.

FROM debian:bookworm-slim

ARG LLAMA_CPP_TAG=b11938
ARG GGUF_REPO=LiquidAI/LFM3-VL-2B-GGUF
ARG GGUF_FILE=LFM3-VL-2B-Q4_K_M.gguf
ARG MMPROJ_FILE=mmproj-LFM3-VL-2B-Q8_0.gguf

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lfm3

RUN curl -fsSL \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_TAG}/llama-${LLAMA_CPP_TAG}-bin-ubuntu-x64.tar.gz" \
      -o /tmp/llama.tgz \
    && mkdir -p /opt/llama.cpp \
    && tar -xzf /tmp/llama.tgz -C /opt/llama.cpp \
    && rm /tmp/llama.tgz \
    && for bin in llama-server llama-cli llama-mtmd-cli; do \
         found="$(find /opt/llama.cpp -type f -name "$bin" -perm -111 | head -n 1)"; \
         if [ -n "$found" ]; then ln -s "$found" "/usr/local/bin/$bin"; fi; \
       done \
    && test -x /usr/local/bin/llama-server

RUN mkdir -p /models \
    && curl -fsSL \
      "https://huggingface.co/${GGUF_REPO}/resolve/main/${GGUF_FILE}" \
      -o "/models/${GGUF_FILE}" \
    && curl -fsSL \
      "https://huggingface.co/${GGUF_REPO}/resolve/main/${MMPROJ_FILE}" \
      -o "/models/${MMPROJ_FILE}" \
    && curl -fsSL \
      "https://huggingface.co/${GGUF_REPO}/resolve/main/LICENSE" \
      -o /models/LICENSE

COPY serve.sh /opt/lfm3/serve.sh
RUN chmod 0755 /opt/lfm3/serve.sh

EXPOSE 8080
ENV LFM3_MODEL=/models/LFM3-VL-2B-Q4_K_M.gguf \
    LFM3_MMPROJ=/models/mmproj-LFM3-VL-2B-Q8_0.gguf \
    LFM3_HOST=0.0.0.0 \
    LFM3_PORT=8080 \
    LFM3_THREADS=0

ENTRYPOINT ["/opt/lfm3/serve.sh"]
