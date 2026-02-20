#!/bin/bash
set -e

MODE=${MODE:-train}

if [ "$MODE" = "serve" ]; then
  echo "Starting inference API (uvicorn)..."
  uvicorn inference_api.app:app --host 0.0.0.0 --port 8000
elif [ "$MODE" = "serve-prod" ]; then
  echo "Starting inference API (gunicorn) production mode..."
  exec gunicorn -k uvicorn.workers.UvicornWorker -w 4 inference_api.app:app --bind 0.0.0.0:8000
else
  echo "Starting training pipeline..."
  python training_pipeline/train.py training_pipeline/train_config.json
fi
