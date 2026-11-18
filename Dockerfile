
# === Ubuntu 24.04.1 & Python 3.10.12 ===
FROM ubuntu:noble-20240904.1@sha256:b359f1067efa76f37863778f7b6d0e8d911e3ee8efa807ad01fbf5dc1ef9006b AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libffi-dev liblzma-dev libncurses-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN curl -fsSLO https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz \
    && tar -xzf Python-3.10.12.tgz \
    && cd Python-3.10.12 \
    && ./configure --prefix=/opt/python --with-ensurepip=install \
    && make -j2 \
    && make install


# === 建立 Otter vLLM 執行環境 ===
FROM ubuntu:noble-20240904.1@sha256:b359f1067efa76f37863778f7b6d0e8d911e3ee8efa807ad01fbf5dc1ef9006b

# 安裝 Python、Phison NPU、OpenCV 所需函式庫
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3t64 zlib1g libbz2-1.0 \
    libreadline8t64 libsqlite3-0 libffi8 liblzma5 \
    libncursesw6 libuuid1 liburing2 libgomp1 libstdc++6 \
    libxcb1 libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

# 複製 Python 3.10.12
COPY --from=builder /opt/python /opt/python

ENV PATH="/opt/python/bin:${PATH}" \
    PYTHONUNBUFFERED=1

WORKDIR /phison_npu_vllm

COPY requirements.txt otter_torch-2.9.9-cp310-cp310-linux_x86_64.whl otter_vllm-2.9.9-py3-none-any.whl ./

# 安裝 Python 套件與 Otter
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-deps --force-reinstall otter_torch-2.9.9-cp310-cp310-linux_x86_64.whl \
    && pip3 install --no-deps --force-reinstall otter_vllm-2.9.9-py3-none-any.whl

# entrypoint.sh 由使用者在執行時掛載，不包進映像。

LABEL org.opencontainers.image.title="Phison NPU vLLM" \
    org.opencontainers.image.version="3.4.0" \
    com.phison.otter.version="2.9.9"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD /opt/python/bin/python3.10 -c 'import os, urllib.request; urllib.request.urlopen("http://127.0.0.1:" + os.environ.get("PORT", "8000") + "/health", timeout=4)' || exit 1

# === 啟動 Otter vLLM ===
ENTRYPOINT ["/bin/bash", "/phison_npu_vllm/entrypoint.sh"]
CMD []
