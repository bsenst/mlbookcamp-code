# Homework 4: Evaluation Metrics for Classification

## Solution Script
`homework_04_solution.py` - Reproduces all answers for Homework 4.

## Requirements
```bash
pip install pandas scikit-learn numpy
```

## Running the Solution
```bash
cd cohorts/2026/homework/04-evaluation
python homework_04_solution.py
```

## What the Script Does
1. Downloads the 2026 lead scoring dataset from GitHub
2. Handles missing values (categorical → 'NA', numerical → 0.0)
3. Splits data into train/validation/test using the exact calls from the homework
4. Computes all 6 answers:
   - Q1: ROC AUC feature importance for numerical variables
   - Q2: Logistic regression AUC on validation set
   - Q3: Threshold where precision and recall intersect
   - Q4: Threshold that maximizes F1 score
   - Q5: Standard deviation of 5-fold CV AUC scores
   - Q6: Best C parameter via 5-fold CV

## Expected Output
```
Q1. Highest AUC numerical variable: lead_score
Q2. Validation AUC (rounded): 0.732
Q3. Precision/Recall intersection threshold: 0.63
Q4. Max F1 threshold: 0.41
Q5. 5-Fold CV std: 0.013
Q6. Best C: 0.001
```