#!/usr/bin/env python3
"""
Homework 8: Neural Networks and Deep Learning - Solution
DataTalks.Club Machine Learning Zoomcamp 2026

This script references the reference training to produce answers.
The reference_train.py script should be run to generate the history files.
"""

import json
import statistics
import numpy as np
from pathlib import Path


def question_1_loss():
    """Q1: Which loss matches the model?"""
    print("\n" + "=" * 60)
    print("QUESTION 1: Loss and Output Contract")
    print("=" * 60)
    print("  The model emits one logit and uses nn.BCEWithLogitsLoss()")
    print("  Answer: nn.BCEWithLogitsLoss()")
    return "nn.BCEWithLogitsLoss()"


def question_2_parameters():
    """Q2: Total number of trainable parameters."""
    print("\n" + "=" * 60)
    print("QUESTION 2: Parameter Count")
    print("=" * 60)
    # From the reference model architecture:
    # Conv2d(3, 32, kernel_size=3, stride=1, padding=0) -> 32 * (3*3*3 + 1) = 32 * 28 = 896
    # Linear(32 * 99 * 99, 64) -> 32 * 99 * 99 * 64 + 64 = 20,070,464 + 64 = 20,070,528
    # Linear(64, 1) -> 64 * 1 + 1 = 65
    # Total = 896 + 20,070,528 + 65 = 20,071,489
    
    # Let's calculate exactly:
    conv_params = 32 * (3 * 3 * 3 + 1)  # 896
    hidden_params = 32 * 99 * 99 * 64 + 64  # 20,070,528
    output_params = 64 * 1 + 1  # 65
    total = conv_params + hidden_params + output_params
    
    print(f"  Conv2d params: {conv_params}")
    print(f"  Hidden Linear params: {hidden_params}")
    print(f"  Output Linear params: {output_params}")
    print(f"  Total: {total}")
    print(f"  Closest option: {total}")
    
    return total


def question_3_4_baseline(history_baseline_path):
    """Q3 & Q4: Baseline training metrics."""
    print("\n" + "=" * 60)
    print("QUESTION 3 & 4: Baseline Training Metrics")
    print("=" * 60)
    
    if not Path(history_baseline_path).exists():
        print(f"  File not found: {history_baseline_path}")
        print("  Run reference_train.py first to generate history files")
        return None, None
    
    with open(history_baseline_path) as f:
        history = json.load(f)
    
    train_acc = history['train_accuracy']
    train_loss = history['train_loss']
    
    median_acc = statistics.median(train_acc)
    std_loss = np.std(train_loss, ddof=0)
    
    print(f"  Train accuracy: {train_acc}")
    print(f"  Median train accuracy: {median_acc:.4f} -> rounded: {median_acc:.2f}")
    print(f"  Train loss: {train_loss}")
    print(f"  Population std of train loss: {std_loss:.6f} -> rounded: {std_loss:.3f}")
    
    return median_acc, std_loss


def question_5_6_augmented(history_augmented_path):
    """Q5 & Q6: Augmented training metrics."""
    print("\n" + "=" * 60)
    print("QUESTION 5 & 6: Augmented Training Metrics")
    print("=" * 60)
    
    if not Path(history_augmented_path).exists():
        print(f"  File not found: {history_augmented_path}")
        print("  Run reference_train.py first to generate history files")
        return None, None
    
    with open(history_augmented_path) as f:
        history = json.load(f)
    
    eval_loss = history['evaluation_loss']
    eval_acc = history['evaluation_accuracy']
    
    mean_eval_loss = statistics.mean(eval_loss)
    last_five_acc = statistics.mean(eval_acc[-5:])
    
    print(f"  Evaluation loss: {eval_loss}")
    print(f"  Mean evaluation loss: {mean_eval_loss:.6f} -> rounded: {mean_eval_loss:.3f}")
    print(f"  Evaluation accuracy: {eval_acc}")
    print(f"  Last 5 evaluation accuracies: {eval_acc[-5:]}")
    print(f"  Mean of last 5: {last_five_acc:.4f} -> rounded: {last_five_acc:.2f}")
    
    return mean_eval_loss, last_five_acc


def main():
    print("=" * 60)
    print("Homework 8: Neural Networks and Deep Learning - Solution")
    print("DataTalks.Club ML Zoomcamp 2026")
    print("=" * 60)
    
    print("\nNOTE: Run reference_train.py first to generate history files:")
    print("  python reference_train.py --data-dir data --output-dir runs/reference")
    
    base_dir = Path(__file__).parent
    history_baseline = base_dir / "runs" / "reference" / "history_baseline.json"
    history_augmented = base_dir / "runs" / "reference" / "history_augmented.json"
    
    q1 = question_1_loss()
    q2 = question_2_parameters()
    q3, q4 = question_3_4_baseline(history_baseline)
    q5, q6 = question_5_6_augmented(history_augmented)
    
    print("\n" + "=" * 60)
    print("SUMMARY OF ANSWERS")
    print("=" * 60)
    print(f"Q1. Loss function: {q1}")
    print(f"Q2. Total parameters: {q2} -> closest option: 20073473")
    print(f"Q3. Median baseline train accuracy: {q3:.2f}" if q3 else "Q3. Median baseline train accuracy: (run reference_train.py)")
    print(f"Q4. Std baseline train loss: {q4:.3f}" if q4 else "Q4. Std baseline train loss: (run reference_train.py)")
    print(f"Q5. Mean augmented eval loss: {q5:.3f}" if q5 else "Q5. Mean augmented eval loss: (run reference_train.py)")
    print(f"Q6. Mean last 5 augmented eval acc: {q6:.2f}" if q6 else "Q6. Mean last 5 augmented eval acc: (run reference_train.py)")


if __name__ == "__main__":
    main()