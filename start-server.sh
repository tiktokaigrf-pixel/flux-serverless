#!/bin/bash
set -e

echo "=== Flux Serverless Worker Starting ==="

# Log-Verzeichnis erstellen
mkdir -p /var/log/portal

# HuggingFace Cache-Verzeichnis (persistent Volume falls vorhanden)
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$HF_HOME"

# Optional: HuggingFace Token für gated Models
# Wird über vast.ai Account-Level Environment Variables gesetzt
if [ -n "$HF_TOKEN" ]; then
    echo "HuggingFace token detected"
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
fi

# FastAPI Server starten, Logs umleiten
echo "Starting FastAPI server on port 18000..."
python /app/server.py > /var/log/portal/flux.log 2>&1 &

echo "Server PID: $!"
echo "Waiting for model to load (this may take a few minutes on first run)..."
