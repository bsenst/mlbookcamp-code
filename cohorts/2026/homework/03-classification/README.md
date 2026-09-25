# Homework 3: Machine Learning for Classification - Solution

This folder contains the solution script for **DataTalks.Club Machine Learning Zoomcamp 2026 - Homework 3**.

## Files

- `homework_03_solution.py` - Reproducible Python script that computes all homework answers

## Requirements

```bash
pip install scikit-learn pandas numpy
```

## How to Run

```bash
cd /workspaces/mlbookcamp-code/cohorts/2026/homework/03-classification
python homework_03_solution.py
```

## Expected Output

The script will print all 6 answers mapped to the multiple-choice options:

```
1. Mode for industry: retail
2. Biggest correlation: annual_income and interaction_count
3. Biggest MI: lead_source
4. Accuracy: 0.85
5. Feature selection: industry
6. Parameter tuning: 0.001
```

## Notes

- The script downloads the dataset directly from the official GitHub URL
- Convergence warnings for small C values (Q6) are expected and don't affect the accuracy scores
- Uses fixed random seed (42) for reproducibility across scikit-learn versions