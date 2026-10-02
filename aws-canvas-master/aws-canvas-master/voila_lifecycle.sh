#!/bin/bash
# Voila Lifecycle Configuration for SageMaker Studio JupyterLab
set -e

NOTEBOOK_S3="s3://canvas-datasets-vorujzada/notebooks/Run_Full_Training_Pipeline.ipynb"
NOTEBOOK_LOCAL="/home/sagemaker-user/Run_Full_Training_Pipeline.ipynb"
VOILA_PORT=8866
LOG_FILE="/home/sagemaker-user/.voila.log"

echo "=== Voila Lifecycle Configuration starting ==="

# Download the latest notebook from S3
echo "Downloading notebook from S3..."
aws s3 cp "$NOTEBOOK_S3" "$NOTEBOOK_LOCAL" --region us-east-1 2>/dev/null || \
    echo "Warning: Could not download from S3, using local copy if exists"

# Install Voila if not present
pip install voila jupyter-server-proxy --quiet 2>&1 | tail -3

# Kill any existing Voila on this port
pkill -f "voila.*$VOILA_PORT" 2>/dev/null || true
sleep 2

# Start Voila on localhost only (required in Studio network isolation)
nohup voila \
    --no-browser \
    --port=$VOILA_PORT \
    --Voila.ip=127.0.0.1 \
    --notebook_path="$NOTEBOOK_LOCAL" \
    > "$LOG_FILE" 2>&1 &

VOILA_PID=$!
echo "Voila started (PID: $VOILA_PID) on port $VOILA_PORT"

# Wait for Voila to be ready
for i in $(seq 1 30); do
    if curl -s "http://127.0.0.1:$VOILA_PORT" > /dev/null 2>&1; then
        echo "Voila is ready!"
        break
    fi
    sleep 2
done

echo "Access at: /jupyterlab/default/proxy/$VOILA_PORT/"
echo "=== Lifecycle Configuration done ==="
