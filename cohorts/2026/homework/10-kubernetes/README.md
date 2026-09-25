# Homework 10: Kubernetes and Model Serving

## Solution Script
`homework_10_solution.py` - Reproduces all answers for Homework 10.

## Requirements
```bash
pip install pyyaml
```

## Prerequisites
1. Build the Docker image from Homework 5:
```bash
cd cohorts/2026/homework/05-deployment
docker build -t zoomcamp-model:2026-hw10 .
```

## Running the Solution
```bash
cd cohorts/2026/homework/10-kubernetes
python homework_10_solution.py
```

## What the Script Does
1. Reads the expected probability from Homework 5 (Q1)
2. Provides setup commands for kind/kubectl (Q2)
3. Answers Kubernetes knowledge questions (Q3-Q4)
4. Reads `deployment.yaml` for container port (Q6)
5. Reads `service.yaml` for selector (Q7)
6. Reads `hpa.yaml` for maxReplicas (Q8)

## Expected Output
```
Q1. Conversion probability: 0.770
Q2. Environment: Setup check
Q3. Smallest deployable unit: Pod
Q4. Default service type: ClusterIP
Q5. Load image command: kind load docker-image
Q6. Container port: 9696
Q7. Service selector: {'app': 'subscription'}
Q8. HPA maxReplicas: 3
```

## Full Kubernetes Deployment (Optional)
If you want to deploy to a local kind cluster:

```bash
# Install kind and kubectl first
# Then:

# Create cluster
kind create cluster --name mlzoomcamp-2026

# Load image into kind
kind load docker-image zoomcamp-model:2026-hw10 --name mlzoomcamp-2026

# Deploy
kubectl apply -f deployment.yaml --context kind-mlzoomcamp-2026
kubectl rollout status deployment/subscription --context kind-mlzoomcamp-2026

# Expose
kubectl apply -f service.yaml --context kind-mlzoomcamp-2026

# Port forward and test
kubectl port-forward service/subscription 9696:80 --context kind-mlzoomcamp-2026
python ../05-deployment/q6_test.py

# Autoscaling
kubectl apply -f hpa.yaml --context kind-mlzoomcamp-2026
kubectl get hpa subscription-hpa --context kind-mlzoomcamp-2026

# Cleanup
kind delete cluster --name mlzoomcamp-2026
```

## Manifest Files
- `deployment.yaml` - Deployment with image `zoomcamp-model:2026-hw10`, port 9696, readiness probe on `/health`
- `service.yaml` - ClusterIP service selecting `app: subscription`, port 80 → targetPort 9696
- `hpa.yaml` - HorizontalPodAutoscaler with minReplicas=1, maxReplicas=3, CPU target 20%