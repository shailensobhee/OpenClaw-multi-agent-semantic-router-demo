#!/bin/bash
# Serve the vLLM Semantic Router stack (Envoy :8899 + router + dashboard).
# First run pulls the router image + the Vela 2.0 0.3B decision model (multi-GB, multi-minute).
set -uo pipefail
LOG=/var/log/booth-vsr-serve.log
exec > >(tee -a "$LOG") 2>&1
echo "========== VSR SERVE START $(date) =========="
. /opt/vsr_venv/bin/activate
cd /opt/vsr_run
# serve runs a compose stack in the background itself; run detached + log.
setsid nohup vllm-sr serve --config config.yaml > /var/log/booth-vsr-serve-inner.log 2>&1 < /dev/null &
echo "serve launched pid=$!"
# wait for Envoy OpenAI entrypoint :8899 to answer
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8899/v1/models 2>/dev/null)
  echo "[$i] envoy :8899 /v1/models -> $code"
  [ "$code" = "200" ] && { echo "ENVOY_READY"; break; }
  sleep 15
done
ufw allow 8899/tcp 2>/dev/null || true
echo "--- containers ---"; docker ps --format '{{.Names}} {{.Status}}' | grep -iE 'envoy|router|semantic|vsr|dashboard' | head
echo "========== VSR SERVE DONE $(date) =========="
touch /root/_vsr_serve.done
