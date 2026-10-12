#!/bin/bash
# Launch the Qwen fleet on the SHARED vLLM-SR DOCKER BRIDGE (not host networking).
#
# WHY A BRIDGE, NOT --network host: the semantic router and Envoy run on Docker
# bridge networks. If the backends run with `--network host`, every router->backend
# call is a bridge->HOST connection (via host.docker.internal) that traverses the
# host INPUT firewall chain. On a ufw host (INPUT policy DROP) that traffic is
# dropped and the router returns 503 "router down". Putting the backends on the
# SAME bridge as the router makes router->backend pure container-to-container
# traffic on the Docker FORWARD chain (ACCEPT, Docker-managed), which NEVER touches
# the host INPUT firewall. There is then nothing for the firewall to interfere with.
# VERIFIED 2026-10-12: with ALL docker-bridge host-firewall ACCEPT rules removed
# AND conntrack flushed, all three tiers still routed end-to-end (30/30 requests,
# 0 failures), and survived a router/envoy restart. The router reaches backends by
# CONTAINER NAME (qwen-05b:8001, qwen-7b:8002, qwen-32b:8003).
#
# Ports are still published to host loopback (127.0.0.1:800x) ONLY so the
# dashboard's Prometheus /metrics scraper keeps working; the routing path does not
# use them.
#
#   fast/trivial   : Qwen3.5-2B   :8001 (Feb 2026, Mamba-hybrid)
#   standard       : Qwen3.5-9B   :8002 (Feb 2026, Mamba-hybrid)
#   hard/reasoning : Qwen3.8-27B  :8003 (Aug 2026, SOTA, in 3-month window)
#
# Image/model traps worked around (both cost real time once):
#  * rocm/vllm:latest's bare `serve` is a Click group that REJECTS a positional
#    model; must use `--entrypoint vllm <img> serve <model> ...`.
#  * Qwen3.5 is Mamba-hybrid: default max_num_seqs=1024 aborts CUDA-graph capture
#    at a low gpu fraction. Cap --max-num-seqs (128 is plenty for a booth).
set -uo pipefail
NET=vllm-sr-network
IMG=rocm/vllm:latest
HF_TOKEN="${HF_TOKEN:-}"
LOG=/var/log/booth-qwen-fleet.log
exec > >(tee -a "$LOG") 2>&1
echo "========== QWEN3 FLEET (BRIDGE) START $(date) =========="
echo "hf_token_present=$([ -n "$HF_TOKEN" ] && echo yes || echo no)"

# Stop any previous fleet (host-net or bridge) to free the GPU.
docker rm -f qwen-05b qwen-7b qwen-32b vllm 2>/dev/null || true
sleep 6
rocm-smi --showmeminfo vram 2>/dev/null | grep -i "used" | head -1

run_one() { # name model port gpu_frac max_num_seqs
  local name=$1 model=$2 port=$3 frac=$4 seqs=$5
  echo "--- launch $name ($model) on $NET :$port frac=$frac maxseqs=$seqs ---"
  docker rm -f "$name" 2>/dev/null || true
  docker run -d --name "$name" --network "$NET" \
    -p 127.0.0.1:${port}:${port} \
    --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16g \
    --security-opt seccomp=unconfined -e HF_TOKEN="$HF_TOKEN" \
    --entrypoint vllm "$IMG" serve "$model" \
    --served-model-name "$(basename "$model")" --port "$port" \
    --max-model-len 32768 --gpu-memory-utilization "$frac" --max-num-seqs "$seqs" \
    --enable-auto-tool-choice --tool-call-parser hermes
}

# Heaviest first so it grabs its block, then the smaller ones.
run_one qwen-32b Qwen/Qwen3.8-27B 8003 0.45 128
sleep 4
run_one qwen-7b  Qwen/Qwen3.5-9B  8002 0.26 128
sleep 4
run_one qwen-05b Qwen/Qwen3.5-2B  8001 0.10 128

echo "--- launched on bridge; router endpoints: qwen-05b:8001 qwen-7b:8002 qwen-32b:8003 ---"
echo "--- torch.compile warmup ~90s per model ---"
docker ps --format '{{.Names}} {{.Status}}' | grep -i qwen
echo "========== QWEN3 FLEET LAUNCH ISSUED $(date) =========="
touch /root/_qwen_fleet_launched.done
