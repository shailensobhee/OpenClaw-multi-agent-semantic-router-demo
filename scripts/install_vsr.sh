#!/bin/bash
# Install vLLM Semantic Router tier in front of the 3 Qwen backends.
# Decision model = ModernBERT category classifier (CPU, zero GPU VRAM).
# Envoy OpenAI entrypoint :8899 -> router -> Qwen 0.5B/7B/32B on host ports 8001/8002/8003.
set -uo pipefail
LOG=/var/log/booth-vllm-sr.log
exec > >(tee -a "$LOG") 2>&1
echo "========== VLLM-SR SETUP START $(date) =========="

python3 -m venv /opt/vsr_venv
. /opt/vsr_venv/bin/activate
pip install -q --upgrade pip
pip install -q vllm-sr && echo "vllm-sr installed: $(vllm-sr --version 2>/dev/null || echo '?')"

mkdir -p /opt/vsr_run && cd /opt/vsr_run
vllm-sr config init --output config.yaml --force
echo "--- generated template head ---"; head -30 config.yaml
echo "========== VLLM-SR INSTALLED $(date) =========="
touch /root/_vsr_installed.done
