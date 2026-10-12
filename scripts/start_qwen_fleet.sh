#!/bin/bash
# Stage A: replace the single 72B with a 3-model Qwen fleet on ONE MI300X.
# 0.5B :8001 (fast) | 7B :8002 (standard) | 32B :8003 (hard/reason).
# Co-resident on one GPU via explicit --gpu-memory-utilization fractions.
set -uo pipefail
LOG=/var/log/booth-qwen-fleet.log
exec > >(tee -a "$LOG") 2>&1
echo "========== QWEN FLEET START $(date) =========="

IMG=rocm/vllm:latest
# Reuse HF token from the existing 72B container env (never printed).
HF_TOKEN=$(docker inspect vllm --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | sed -n 's/^HF_TOKEN=//p' | head -1)
echo "hf_token_present=$([ -n "$HF_TOKEN" ] && echo yes || echo no)"

# --- ROLLBACK command saved for safety ---
cat > /root/_vllm_72b_rollback.sh <<'RB'
#!/bin/bash
docker rm -f qwen-05b qwen-7b qwen-32b 2>/dev/null
docker run -d --rm --name vllm --network host \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size 16g --security-opt seccomp=unconfined \
  -e HF_TOKEN="$HF_TOKEN_PLACEHOLDER" \
  rocm/vllm:latest serve Qwen/Qwen2.5-72B-Instruct \
  --served-model-name Qwen2.5-72B-Instruct --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hermes
RB
sed -i "s#\$HF_TOKEN_PLACEHOLDER#${HF_TOKEN}#" /root/_vllm_72b_rollback.sh
chmod +x /root/_vllm_72b_rollback.sh
echo "rollback saved: /root/_vllm_72b_rollback.sh"

# --- stop the 72B to free the GPU ---
echo "--- stopping 72B ---"
docker rm -f vllm 2>/dev/null || true
sleep 8
rocm-smi --showmeminfo vram 2>/dev/null | grep -i "used" | head -1

# NOTE: no --group-add; container runs as root and the GPU devices are
# root:render mode 660, so root already has rw (matches how the 72B ran).
common=(--rm -d --network host --device=/dev/kfd --device=/dev/dri
  --ipc=host --shm-size 16g
  --security-opt seccomp=unconfined -e HF_TOKEN="$HF_TOKEN" "$IMG")

launch() { # name model port gpu_frac maxlen
  local name=$1 model=$2 port=$3 frac=$4 maxlen=$5
  echo "--- launch $name ($model) :$port frac=$frac ---"
  docker rm -f "$name" 2>/dev/null || true
  docker run --name "$name" "${common[@]}" serve "$model" \
    --served-model-name "$(basename "$model")" --port "$port" \
    --max-model-len "$maxlen" --gpu-memory-utilization "$frac" \
    --enable-auto-tool-choice --tool-call-parser hermes
}

# Launch heaviest first so it grabs its block, then smaller ones.
launch qwen-32b Qwen/Qwen2.5-32B-Instruct 8003 0.45 32768
sleep 5
launch qwen-7b  Qwen/Qwen2.5-7B-Instruct  8002 0.14 32768
sleep 5
launch qwen-05b Qwen/Qwen2.5-0.5B-Instruct 8001 0.05 32768

echo "--- launched; health comes from the watcher ---"
docker ps --format '{{.Names}} {{.Status}}' | grep -i qwen
echo "========== QWEN FLEET LAUNCH ISSUED $(date) =========="
touch /root/_qwen_fleet_launched.done
