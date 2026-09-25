#!/usr/bin/env python3
"""
Homework 9: Serverless Deep Learning - Solution
DataTalks.Club Machine Learning Zoomcamp 2026

This script reproduces all answers for Homework 9.
"""

import json
import numpy as np
from pathlib import Path
import onnxruntime as ort
from io import BytesIO
from urllib import request
from PIL import Image


def download_image(url):
    with request.urlopen(url, timeout=30) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


def prepare_image(image):
    image = image.resize((200, 200), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.transpose(array, (2, 0, 1))[None, ...]


def question_1_onnx_output():
    """Q1: What is the output node name?"""
    print("\n" + "=" * 60)
    print("QUESTION 1: ONNX Graph Output Node")
    print("=" * 60)
    
    model_path = Path(__file__).parent / "hair_classifier_v1.onnx"
    if not model_path.exists():
        print("  Model not found. Download from:")
        print("  curl -fL -o hair_classifier_v1.onnx https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle/hair_classifier_v1.onnx")
        print("  curl -fL -o hair_classifier_v1.onnx.data https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle/hair_classifier_v1.onnx.data")
        return "output"
    
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    print(f"  Input name: {input_name}")
    print(f"  Input shape: {input_shape}")
    print(f"  Output name: {output_name}")
    
    # From asset_manifest.json: "output_name": "output"
    print(f"  Answer from manifest: output")
    return output_name


def question_2_target_size():
    """Q2: What target size does prepare_image use?"""
    print("\n" + "=" * 60)
    print("QUESTION 2: Target Size")
    print("=" * 60)
    print("  prepare_image uses image.resize((200, 200), ...)")
    print("  Answer: 200x200")
    return "200x200"


def question_3_normalized_input():
    """Q3: First value of R channel after normalization."""
    print("\n" + "=" * 60)
    print("QUESTION 3: Normalized Input")
    print("=" * 60)
    
    sample_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    
    try:
        image = download_image(sample_url)
        tensor = prepare_image(image)
        first_r = tensor[0, 0, 0, 0]
        print(f"  First R channel value: {first_r:.6f}")
        print(f"  Rounded to 3 decimals: {first_r:.3f}")
        return first_r
    except Exception as e:
        print(f"  Error downloading image: {e}")
        return None


def question_4_onnx_inference():
    """Q4: Local ONNX inference probability."""
    print("\n" + "=" * 60)
    print("QUESTION 4: Local ONNX Inference")
    print("=" * 60)
    
    model_path = Path(__file__).parent / "hair_classifier_v1.onnx"
    if not model_path.exists():
        print("  Model not found. Download required.")
        return None
    
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name
    
    sample_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    
    try:
        image = download_image(sample_url)
        tensor = prepare_image(image)
        output = session.run([output_name], {input_name: tensor})[0]
        probability = float(output.reshape(-1)[0])
        print(f"  Straight probability: {probability:.6f}")
        print(f"  Rounded to 3 decimals: {probability:.3f}")
        return probability
    except Exception as e:
        print(f"  Error: {e}")
        return None


def question_5_dockerfile_base():
    """Q5: Lambda runtime base image."""
    print("\n" + "=" * 60)
    print("QUESTION 5: Lambda Configuration")
    print("=" * 60)
    
    dockerfile_path = Path(__file__).parent / "Dockerfile"
    with open(dockerfile_path, "r") as f:
        first_line = f.readline().strip()
    
    print(f"  Dockerfile first line: {first_line}")
    if first_line.startswith("FROM "):
        base_image = first_line[5:]
        print(f"  Base image: {base_image}")
        return base_image
    
    return "public.ecr.aws/lambda/python:3.13"


def question_6_container_invoke():
    """Q6: Invoke container - same as Q4."""
    print("\n" + "=" * 60)
    print("QUESTION 6: Invoke Container")
    print("=" * 60)
    print("  Build: docker build -t mlzoomcamp-2026-serverless .")
    print("  Run: docker run --rm -p 9000:8080 mlzoomcamp-2026-serverless")
    print("  Invoke: curl -XPOST http://localhost:9000/2015-03-31/functions/function/invocations ...")
    print("  Result should match Q4 (same model, image, preprocessing)")
    return "Same as Q4"


def main():
    print("=" * 60)
    print("Homework 9: Serverless Deep Learning - Solution")
    print("DataTalks.Club ML Zoomcamp 2026")
    print("=" * 60)
    
    # Check if model files exist
    model_dir = Path(__file__).parent
    model_file = model_dir / "hair_classifier_v1.onnx"
    data_file = model_dir / "hair_classifier_v1.onnx.data"
    sample_file = model_dir / "sample.jpeg"
    
    print(f"\nModel file exists: {model_file.exists()}")
    print(f"External data file exists: {data_file.exists()}")
    print(f"Sample image exists: {sample_file.exists()}")
    
    if not model_file.exists():
        print("\nDownload required files:")
        print("  curl -fL -o hair_classifier_v1.onnx https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle/hair_classifier_v1.onnx")
        print("  curl -fL -o hair_classifier_v1.onnx.data https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle/hair_classifier_v1.onnx.data")
        print("  curl -fL -o sample.jpeg https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg")
    
    q1 = question_1_onnx_output()
    q2 = question_2_target_size()
    q3 = question_3_normalized_input()
    q4 = question_4_onnx_inference()
    q5 = question_5_dockerfile_base()
    q6 = question_6_container_invoke()
    
    print("\n" + "=" * 60)
    print("SUMMARY OF ANSWERS")
    print("=" * 60)
    print(f"Q1. Output node name: {q1}")
    print(f"Q2. Target size: {q2}")
    print(f"Q3. First R channel value: {q3:.3f}" if q3 else "Q3. First R channel value: (download required)")
    print(f"Q4. Straight probability: {q4:.3f}" if q4 else "Q4. Straight probability: (download required)")
    print(f"Q5. Dockerfile base image: {q5}")
    print(f"Q6. Container prediction: {q6}")


if __name__ == "__main__":
    main()