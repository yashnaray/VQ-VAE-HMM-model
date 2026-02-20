Deployment Guide for VQ-VAE-HMM Portfolio System

Overview
- This repository supports containerized deployment using Docker and optional Kubernetes manifests for scalable serving.
- We provide a training pipeline, a FastAPI inference API, and a production-ready serving option via Gunicorn.

Prerequisites
- Docker and (optionally) Docker Compose installed locally or in CI.
- Optional: access to a Kubernetes cluster (minikube, kind, GKE, EKS, etc.).
- Access to source of model weights and config used for serving.

Deployment steps (local Docker Compose)
1) Build and run the services:
   docker compose up --build
   This starts: trainer (MODE=train) and server (MODE=serve-prod) for production API.
2) Verify health: GET http://localhost:8000/health
3) Inference: POST to /infer with a JSON body containing an "x" matrix.

Deployment steps (Kubernetes)
   docker build -t REPO/vae-hmm:latest -f Dockerfile .
   docker push REPO/vae-hmm:latest
2) Apply manifests:
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
3) Expose via an external load balancer or Ingress as per your cloud provider.

Observability
- Logs: stdout of containers.
- Metrics: integrate with Prometheus/Grafana if desired.

Next steps
- Add a small smoke-test script to validate the inference API in CI.
- Parameterize inference via config map/secret for model path and config.
