# Homework 5: Deploying Machine Learning Models

## Solution Script
`homework_05_solution.py` - Reproduces all answers for Homework 5.

## Requirements
```bash
cd cohorts/2026/homework/05-deployment
uv sync --locked
```

## Running the Solution
```bash
cd cohorts/2026/homework/05-deployment
python homework_05_solution.py
```

## What the Script Does
1. Reads the locked scikit-learn version from `pyproject.toml`
2. Loads the frozen model artifact (`pipeline.bin`)
3. Computes predictions for the two test leads locally
4. Reads the Dockerfile to identify the base image
5. Provides instructions for container-based inference

## Expected Output
```
Q1. uv version: (run 'uv --version' locally)
Q2. scikit-learn version: 1.7.2
Q3. Lead 1 conversion probability: 0.533
Q4. Lead 2 conversion probability: 0.770
Q5. Dockerfile base image: python:3.11.15-slim-bookworm
Q6. Container prediction: Same as Q4 (matches Q4)
```

## Container Testing (Optional)
```bash
# Build and run the container
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5

# In another terminal, test the API
python q6_test.py
# Or use curl
curl -s http://localhost:9696/predict -H 'Content-Type: application/json' -d '{"lead_source": "organic_search", "industry": "technology", "employment_status": "employed", "location": "europe", "number_of_courses_viewed": 4, "annual_income": 80304.0, "interaction_count": 7, "lead_score": 0.74}'
```