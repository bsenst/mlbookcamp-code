# Homework 9: Serverless Deep Learning

## Solution Script
`homework_09_solution.py` - Reproduces all answers for Homework 9.

## Requirements
```bash
pip install onnxruntime numpy pillow
```

## Prerequisites
Download the reference assets:
```bash
cd cohorts/2026/homework/09-serverless
PREFIX="https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
curl -fL -o hair_classifier_v1.onnx "${PREFIX}/hair_classifier_v1.onnx"
curl -fL -o hair_classifier_v1.onnx.data "${PREFIX}/hair_classifier_v1.onnx.data"
curl -fL -o sample.jpeg "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

# Verify checksums
sha256sum hair_classifier_v1.onnx hair_classifier_v1.onnx.data sample.jpeg
```

## Running the Solution
```bash
cd cohorts/2026/homework/09-serverless
python homework_09_solution.py
```

## What the Script Does
1. Loads the ONNX model and inspects input/output nodes (Q1)
2. Reads the preprocessing code to determine target size (Q2)
3. Downloads sample image, runs preprocessing, extracts first R channel value (Q3)
4. Runs ONNX inference with ONNX Runtime CPU provider (Q4)
5. Reads Dockerfile for base image (Q5)
6. Notes that container invocation matches local inference (Q6)

## Expected Output
```
Q1. Output node name: output
Q2. Target size: 200x200
Q3. First R channel value: -1.056
Q4. Straight probability: 0.728
Q5. Dockerfile base image: public.ecr.aws/lambda/python:3.13
Q6. Container prediction: Same as Q4
```

## Container Testing (Optional)
```bash
# Build and run Lambda-compatible container
docker build -t mlzoomcamp-2026-serverless .
docker run --rm -p 9000:8080 mlzoomcamp-2026-serverless

# In another terminal, invoke
curl -s -XPOST 'http://localhost:9000/2015-03-31/functions/function/invocations' \
  -H 'Content-Type: application/json' \
  -d '{"image_url":"https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}'
```

## Smoke Test
```bash
python smoke_test.py
```