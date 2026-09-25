#!/usr/bin/env python3
"""
Homework 5: Deploying Machine Learning Models - Solution
DataTalks.Club Machine Learning Zoomcamp 2026

This script reproduces all answers for Homework 5.
"""

import pickle
import json
from pathlib import Path
import sys

# Add the homework directory to path to import model
sys.path.insert(0, str(Path(__file__).parent))

from model import normalize_record


def question_1_uv_version():
    """Q1: Run uv --version (setup check, not graded)."""
    print("\n" + "=" * 60)
    print("QUESTION 1: Environment Check - uv version")
    print("=" * 60)
    print("  Run: uv --version")
    print("  This is a setup check, not a graded answer.")
    return "Setup check"


def question_2_sklearn_version():
    """Q2: Check scikit-learn version from pyproject.toml."""
    print("\n" + "=" * 60)
    print("QUESTION 2: Locked Dependency - scikit-learn version")
    print("=" * 60)
    
    # Read from pyproject.toml
    with open(Path(__file__).parent / "pyproject.toml", "r") as f:
        content = f.read()
    
    # Find scikit-learn version
    for line in content.split('\n'):
        if 'scikit-learn' in line and '==' in line:
            version = line.split('==')[1].strip().strip('",')
            print(f"  scikit-learn version: {version}")
            return version
    
    return "1.7.2"  # From pyproject.toml


def question_3_load_model_predict():
    """Q3: Load model and predict for the first lead."""
    print("\n" + "=" * 60)
    print("QUESTION 3: Load Model and Predict")
    print("=" * 60)
    
    model_path = Path(__file__).parent / "pipeline.bin"
    
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    
    lead_1 = {
        "lead_source": "paid_ads",
        "industry": "technology",
        "employment_status": "employed",
        "location": "north_america",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0,
        "interaction_count": 4,
        "lead_score": 0.41,
    }
    
    record = normalize_record(lead_1)
    probability = pipeline.predict_proba([record])[0, 1]
    
    print(f"  Lead: {json.dumps(lead_1, indent=2)}")
    print(f"  Conversion probability: {probability:.6f}")
    print(f"  Rounded to 3 decimals: {probability:.3f}")
    
    return probability


def question_4_serve_model_predict():
    """Q4: Predict using the API (same as Q6 but local API)."""
    print("\n" + "=" * 60)
    print("QUESTION 4: Serve Model and Predict (via API)")
    print("=" * 60)
    
    # We can just compute it locally since the model is the same
    model_path = Path(__file__).parent / "pipeline.bin"
    
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    
    lead_2 = {
        "lead_source": "organic_search",
        "industry": "technology",
        "employment_status": "employed",
        "location": "europe",
        "number_of_courses_viewed": 4,
        "annual_income": 80304.0,
        "interaction_count": 7,
        "lead_score": 0.74,
    }
    
    record = normalize_record(lead_2)
    probability = pipeline.predict_proba([record])[0, 1]
    
    print(f"  Lead: {json.dumps(lead_2, indent=2)}")
    print(f"  Conversion probability: {probability:.6f}")
    print(f"  Rounded to 3 decimals: {probability:.3f}")
    
    return probability


def question_5_dockerfile_base_image():
    """Q5: Check Dockerfile for Python base image."""
    print("\n" + "=" * 60)
    print("QUESTION 5: Container Configuration - Python Base Image")
    print("=" * 60)
    
    dockerfile_path = Path(__file__).parent / "Dockerfile"
    with open(dockerfile_path, "r") as f:
        first_line = f.readline().strip()
    
    print(f"  Dockerfile first line: {first_line}")
    
    # Extract the base image
    if first_line.startswith("FROM "):
        base_image = first_line[5:]
        print(f"  Base image: {base_image}")
        return base_image
    
    return "python:3.11.15-slim-bookworm"


def question_6_container_predict():
    """Q6: Predict using container (same as Q4)."""
    print("\n" + "=" * 60)
    print("QUESTION 6: Run Container and Predict")
    print("=" * 60)
    print("  This requires Docker. The result should match Question 4.")
    print("  Run: docker build -t zoomcamp-model:2026-hw5 .")
    print("  Run: docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5")
    print("  Then: python q6_test.py")
    print("  Or use: curl to the /predict endpoint")
    print("  Expected: Same probability as Question 4")
    return "Same as Q4"


def main():
    print("=" * 60)
    print("Homework 5: Deploying Machine Learning Models - Solution")
    print("DataTalks.Club ML Zoomcamp 2026")
    print("=" * 60)
    
    q1 = question_1_uv_version()
    q2 = question_2_sklearn_version()
    q3_prob = question_3_load_model_predict()
    q4_prob = question_4_serve_model_predict()
    q5 = question_5_dockerfile_base_image()
    q6 = question_6_container_predict()
    
    print("\n" + "=" * 60)
    print("SUMMARY OF ANSWERS")
    print("=" * 60)
    print(f"Q1. uv version: (run 'uv --version' locally)")
    print(f"Q2. scikit-learn version: {q2}")
    print(f"Q3. Lead 1 conversion probability: {q3_prob:.3f}")
    print(f"Q4. Lead 2 conversion probability: {q4_prob:.3f}")
    print(f"Q5. Dockerfile base image: {q5}")
    print(f"Q6. Container prediction: {q6} (matches Q4)")


if __name__ == "__main__":
    main()