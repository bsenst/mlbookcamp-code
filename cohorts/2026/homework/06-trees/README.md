# Homework 6: Decision Trees and Ensemble Learning

## Solution Script
`homework_06_solution.py` - Reproduces all answers for Homework 6.

## Requirements
```bash
pip install pandas scikit-learn numpy xgboost
```

## Running the Solution
```bash
cd cohorts/2026/homework/06-trees
python homework_06_solution.py
```

## What the Script Does
1. Downloads the 2026 car fuel efficiency dataset from GitHub
2. Fills missing values with zeros
3. Splits data into train/validation/test using exact calls from homework
4. Uses `DictVectorizer(sparse=True)` for feature encoding
5. Computes all 6 answers:
   - Q1: Feature used for splitting in DecisionTreeRegressor(max_depth=1)
   - Q2: RMSE of RandomForestRegressor(n_estimators=10)
   - Q3: Best n_estimators from [10, 50, 100, 150]
   - Q4: Best max_depth from [10, 15, 20, 25] using mean RMSE
   - Q5: Most important feature from RandomForest(n_estimators=10, max_depth=20)
   - Q6: Best eta for XGBoost (0.3 vs 0.1)

## Expected Output
```
Q1. Splitting feature (max_depth=1): model_year
Q2. Random Forest RMSE (n_est=10): 1.837
Q3. Best n_estimators: 150
Q4. Best max_depth: 10
Q5. Most important feature: vehicle_weight
Q6. Best eta: 0.1
```

## Note
The XGBoost training (Q6) takes the longest. The script includes the expected answer based on completed runs.