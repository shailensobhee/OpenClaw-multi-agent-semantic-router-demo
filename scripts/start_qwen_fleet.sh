#!/bin/bash
# Newest-per-tier Qwen fleet on ONE MI300X (co-resident), verified 2026-10-12.
#   fast/trivial   : Qwen3.5-2B   :8001 (Feb 2026, Mamba-hybrid)
#   standard       : Qwen3.5-9B   :8002 (Feb 2026, Mamba-hybrid)
#   hard/reasoning : Qwen3.8-27B  :8003 (Aug 2026, SOTA, in 3-month window)
# Co-resident via explicit --gpu-memory-utilization fractions (~130GB/192GB used).
#
# Two image/model traps this script works around (both cost real time once):
#  * rocm/vllm:latest's bare `serve` is a Click group that REJECTS a positional
#    model; you must use `--entrypoint vllm <img> serve <model> ...`.
#  * Qwen3.5 is a Mamba-hybrid arch: at a low gpu fraction the default
#    max_num_seqs=1024 aborts CUDA-graph capture ("exceeds available Mamba cache
#    blocks"). Cap --max-num-seqs (128 is plenty for a booth).
#
# NETWORK PREREQ: the vllm-sr router/envoy run on docker bridges and reach these
# host-network backends via host.docker.internal. On a ufw host (INPUT policy
# DROP) that bridge->host traffic is dropped and the router returns 503; allow the
# docker bridge subnets into the host INPUT chain and persist it (see README).
set -uo pipefail
LOG=/var/log/booth-qwen-fleet.log
exec > >(tee -a "$LOG") 2>&1
echo "========== QWEN3 FLEET START $(date) =========="

IMG=rocm/vllm:latest
# Models are public Apache-2.0; HF token is belt-and-suspenders for rate limits.
HF_TOKEN="${HF_TOKEN:-}"
echo "hf_token_present=$([ -n "$HF_TOKEN" ] && echo yes || echo no)"

# --- stop any previous fleet to free the GPU ---
echo "--- stopping old fleet ---"
docker rm -f qwen-05b qwen-7b qwen-32b vllm 2>/dev/null || true
sleep 6
rocm-smi --showmeminfo vram 2>/dev/null | grep -i "used" | head -1

run_one() { # name model port gpu_frac max_num_seqs
  local name=$1 model=$2 port=$3 frac=$4 seqs=$5
  echo "--- launch $name ($model) :$port frac=$frac maxseqs=$seqs ---"
  docker rm -f "$name" 2>/dev/null || true
  docker run -d --name "$name" --network host --device=/dev/kfd --device=/dev/dri \
    --ipc=host --shm-size 16g --security-opt seccomp=unconfined -e HF_TOKEN="$HF_TOKEN" \
    --entrypoint vllm "$IMG" serve "$model" \
    --served-model-name "$(basename "$model")" --port "$port" \
    --max-model-len 32768 --gpu-memory-utilization "$frac" --max-num-seqs "$seqs" \
    --enable-auto-tool-choice --tool-call-parser hermes
}

# Heaviest first so it grabs its block, then the smaller ones.
run_one qwen-32b Qwen/Qwen3.8-27B 8003 0.42 128
sleep 4
run_one qwen-7b  Qwen/Qwen3.5-9B  8002 0.26 128
sleep 4
run_one qwen-05b Qwen/Qwen3.5-2B  8001 0.10 128

echo "--- launch issued; torch.compile warmup ~90s per model ---"
docker ps --format '{{.Names}} {{.Status}}' | grep -i qwen
echo "========== QWEN3 FLEET LAUNCH ISSUED $(date) =========="
touch /root/_qwen_fleet_launched.done
