#!/usr/bin/env python3
"""
Homework 10: Kubernetes and Model Serving - Solution
DataTalks.Club Machine Learning Zoomcamp 2026

This script reproduces all answers for Homework 10.
"""

import yaml
import json
import subprocess
import sys
from pathlib import Path


def question_1_local_container():
    """Q1: Local container schema - conversion probability."""
    print("\n" + "=" * 60)
    print("QUESTION 1: Local Container Schema")
    print("=" * 60)
    print("  Run the container from 05-deployment:")
    print("  cd ../05-deployment")
    print("  docker build -t zoomcamp-model:2026-hw10 .")
    print("  docker run --rm -p 9696:9696 zoomcamp-model:2026-hw10")
    print("  Then run: python q6_test.py")
    print("  Expected probability: 0.770 (from Homework 5, Q4)")
    return 0.770


def question_2_env_check():
    """Q2: Environment check - kind and kubectl versions."""
    print("\n" + "=" * 60)
    print("QUESTION 2: Environment Check")
    print("=" * 60)
    print("  Run: kind --version")
    print("  Run: kubectl version --client")
    print("  This is a setup diagnostic, not a graded answer.")
    return "Setup check"


def question_3_k8s_primitives():
    """Q3: Smallest deployable computing unit in Kubernetes."""
    print("\n" + "=" * 60)
    print("QUESTION 3: Kubernetes Primitives")
    print("=" * 60)
    print("  The smallest deployable unit is a Pod.")
    print("  Options: Node, Pod, Deployment, Service")
    print("  Answer: Pod")
    return "Pod"


def question_4_default_service():
    """Q4: Default service type in kind cluster."""
    print("\n" + "=" * 60)
    print("QUESTION 4: Default Service Type")
    print("=" * 60)
    print("  In a kind cluster, the default 'kubernetes' service is ClusterIP.")
    print("  Options: NodePort, ClusterIP, ExternalName, LoadBalancer")
    print("  Answer: ClusterIP")
    return "ClusterIP"


def question_5_load_image():
    """Q5: Command to load Docker image into kind."""
    print("\n" + "=" * 60)
    print("QUESTION 5: Load Local Image")
    print("=" * 60)
    print("  Command: kind load docker-image zoomcamp-model:2026-hw10 --name mlzoomcamp-2026")
    print("  Options: kind create cluster, kind build node-image, kind load docker-image, kubectl apply")
    print("  Answer: kind load docker-image")
    return "kind load docker-image"


def question_6_container_port():
    """Q6: Container port from deployment.yaml."""
    print("\n" + "=" * 60)
    print("QUESTION 6: Container Port")
    print("=" * 60)
    
    deploy_path = Path(__file__).parent / "deployment.yaml"
    with open(deploy_path, "r") as f:
        deploy = yaml.safe_load(f)
    
    container_port = deploy['spec']['template']['spec']['containers'][0]['ports'][0]['containerPort']
    print(f"  Container port from deployment.yaml: {container_port}")
    print("  Options: 80, 8080, 9000, 9696")
    return container_port


def question_7_service_selector():
    """Q7: Service selector from service.yaml."""
    print("\n" + "=" * 60)
    print("QUESTION 7: Service Selector")
    print("=" * 60)
    
    svc_path = Path(__file__).parent / "service.yaml"
    with open(svc_path, "r") as f:
        svc = yaml.safe_load(f)
    
    selector = svc['spec']['selector']
    print(f"  Selector from service.yaml: {selector}")
    print("  Options: app: api, app: subscription, app: lead-scoring, app: zoomcamp-model")
    return selector


def question_8_hpa_max_replicas():
    """Q8: HPA maxReplicas from hpa.yaml."""
    print("\n" + "=" * 60)
    print("QUESTION 8: HPA Configuration")
    print("=" * 60)
    
    hpa_path = Path(__file__).parent / "hpa.yaml"
    with open(hpa_path, "r") as f:
        hpa = yaml.safe_load(f)
    
    max_replicas = hpa['spec']['maxReplicas']
    print(f"  maxReplicas from hpa.yaml: {max_replicas}")
    print("  Options: 1, 2, 3, 4")
    return max_replicas


def main():
    print("=" * 60)
    print("Homework 10: Kubernetes and Model Serving - Solution")
    print("DataTalks.Club ML Zoomcamp 2026")
    print("=" * 60)
    
    q1 = question_1_local_container()
    q2 = question_2_env_check()
    q3 = question_3_k8s_primitives()
    q4 = question_4_default_service()
    q5 = question_5_load_image()
    q6 = question_6_container_port()
    q7 = question_7_service_selector()
    q8 = question_8_hpa_max_replicas()
    
    print("\n" + "=" * 60)
    print("SUMMARY OF ANSWERS")
    print("=" * 60)
    print(f"Q1. Conversion probability: {q1:.3f}")
    print(f"Q2. Environment: {q2}")
    print(f"Q3. Smallest deployable unit: {q3}")
    print(f"Q4. Default service type: {q4}")
    print(f"Q5. Load image command: {q5}")
    print(f"Q6. Container port: {q6}")
    print(f"Q7. Service selector: {q7}")
    print(f"Q8. HPA maxReplicas: {q8}")


if __name__ == "__main__":
    main()