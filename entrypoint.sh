#!/usr/bin/env bash
set -euo pipefail


exec /opt/python/bin/python3.10 -m vllm.entrypoints.openai.api_server \
  --model /Llama-3.1-8B-Instruct-W8A8 \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --max-model-len 8096 \
  --max-num-seqs 1 \
  --no-enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --generation-config vllm \
  --enable-force-include-usage \
  "$@"
