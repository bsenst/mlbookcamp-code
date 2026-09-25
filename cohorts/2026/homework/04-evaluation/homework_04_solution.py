#!/usr/bin/env python3
"""
Homework 4: Evaluation Metrics for Classification - Solution
DataTalks.Club Machine Learning Zoomcamp 2026

This script reproduces all answers for Homework 4.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score


def load_data():
    """Load the 2026 lead scoring dataset."""
    url = 'https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv'
    data = pd.read_csv(url)
    return data


def prepare_data(data):
    """Prepare data with missing value handling and train/val/test split."""
    # Check for missing values
    print(f"Missing values:\n{data.isnull().sum()}")
    
    # Separate features and target
    target = 'converted'
    y = data[target].values
    X = data.drop(target, axis=1)
    
    # Identify categorical and numerical columns
    cat = ['lead_source', 'industry', 'employment_status', 'location']
    numeric = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']
    
    # Fill missing values
    X[cat] = X[cat].fillna('NA')
    X[numeric] = X[numeric].fillna(0.0)
    
    # Split data
    df_full_train, df_test = train_test_split(X, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    
    y_full_train = y[df_full_train.index]
    y_train = y[df_train.index]
    y_val = y[df_val.index]
    y_test = y[df_test.index]
    
    return df_train, df_val, df_test, y_train, y_val, y_test, y_full_train, cat, numeric


def question_1_roc_auc_feature_importance(df_train, y_train, numeric):
    """Q1: ROC AUC feature importance for numerical variables."""
    print("\n" + "=" * 60)
    print("QUESTION 1: ROC AUC Feature Importance")
    print("=" * 60)
    
    aucs = {}
    for var in numeric:
        scores = df_train[var].values
        auc = roc_auc_score(y_train, scores)
        if auc < 0.5:
            auc = roc_auc_score(y_train, -scores)
        aucs[var] = auc
        print(f"  {var}: AUC = {auc:.6f}")
    
    best_var = max(aucs, key=aucs.get)
    print(f"\n  Highest AUC: {best_var} ({aucs[best_var]:.6f})")
    return best_var


def question_2_train_model(df_train, df_val, y_train, y_val, cat, numeric):
    """Q2: Train logistic regression and compute AUC on validation."""
    print("\n" + "=" * 60)
    print("QUESTION 2: Training the Model")
    print("=" * 60)
    
    dv = DictVectorizer(sparse=False)
    
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    
    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    
    print(f"  Validation AUC: {auc:.6f}")
    print(f"  Rounded to 3 digits: {auc:.3f}")
    return auc


def question_3_precision_recall(df_val, y_val, model, dv):
    """Q3: Find threshold where precision and recall intersect."""
    print("\n" + "=" * 60)
    print("QUESTION 3: Precision and Recall Intersection")
    print("=" * 60)
    
    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    y_scores = model.predict_proba(X_val)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_val, y_scores)
    
    # Find intersection (excluding where both are 0)
    best_idx = None
    min_diff = float('inf')
    best_threshold = None
    
    for i, (p, r) in enumerate(zip(precision, recall)):
        if p == 0 and r == 0:
            continue
        diff = abs(p - r)
        if diff < min_diff:
            min_diff = diff
            best_idx = i
            # threshold corresponds to precision[i], recall[i]
            # but thresholds array is one element shorter
            if i < len(thresholds):
                best_threshold = thresholds[i]
            else:
                best_threshold = 1.0
    
    print(f"  Intersection threshold: {best_threshold:.4f}")
    print(f"  Precision at threshold: {precision[best_idx]:.4f}")
    print(f"  Recall at threshold: {recall[best_idx]:.4f}")
    print(f"  Absolute difference: {min_diff:.4f}")
    
    # Check the options: 0.43, 0.63, 0.73, 0.83
    options = [0.43, 0.63, 0.73, 0.83]
    closest = min(options, key=lambda x: abs(x - best_threshold))
    print(f"  Closest option: {closest}")
    
    return closest


def question_4_f1_score(df_val, y_val, model, dv):
    """Q4: Find threshold that maximizes F1 score."""
    print("\n" + "=" * 60)
    print("QUESTION 4: F1 Score Maximization")
    print("=" * 60)
    
    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    y_scores = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_f1 = 0
    best_threshold = 0
    
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    print(f"  Best threshold: {best_threshold:.2f}")
    print(f"  Best F1: {best_f1:.6f}")
    
    options = [0.21, 0.41, 0.61, 0.81]
    closest = min(options, key=lambda x: abs(x - best_threshold))
    print(f"  Closest option: {closest}")
    
    return closest


def question_5_cross_validation(df_full_train, y_full_train, cat, numeric):
    """Q5: 5-Fold CV standard deviation."""
    print("\n" + "=" * 60)
    print("QUESTION 5: 5-Fold Cross-Validation")
    print("=" * 60)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df_full_train)):
        df_train_fold = df_full_train.iloc[train_idx]
        df_val_fold = df_full_train.iloc[val_idx]
        y_train_fold = y_full_train[train_idx]
        y_val_fold = y_full_train[val_idx]
        
        dv = DictVectorizer(sparse=False)
        train_dict = df_train_fold.to_dict(orient='records')
        X_train = dv.fit_transform(train_dict)
        
        val_dict = df_val_fold.to_dict(orient='records')
        X_val = dv.transform(val_dict)
        
        model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
        model.fit(X_train, y_train_fold)
        
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred)
        scores.append(auc)
        print(f"  Fold {fold + 1}: AUC = {auc:.6f}")
    
    scores = np.array(scores)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"  Mean AUC: {mean_score:.6f}")
    print(f"  Std AUC: {std_score:.6f}")
    print(f"  Std rounded to 3 digits: {std_score:.3f}")
    
    options = [0.001, 0.007, 0.013, 0.060]
    closest = min(options, key=lambda x: abs(x - std_score))
    print(f"  Closest option: {closest}")
    
    return closest


def question_6_hyperparameter_tuning(df_full_train, y_full_train, cat, numeric):
    """Q6: Hyperparameter tuning with 5-Fold CV."""
    print("\n" + "=" * 60)
    print("QUESTION 6: Hyperparameter Tuning")
    print("=" * 60)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    C_values = [0.000001, 0.001, 1]
    
    results = []
    for C in C_values:
        scores = []
        for train_idx, val_idx in kfold.split(df_full_train):
            df_train_fold = df_full_train.iloc[train_idx]
            df_val_fold = df_full_train.iloc[val_idx]
            y_train_fold = y_full_train[train_idx]
            y_val_fold = y_full_train[val_idx]
            
            dv = DictVectorizer(sparse=False)
            train_dict = df_train_fold.to_dict(orient='records')
            X_train = dv.fit_transform(train_dict)
            
            val_dict = df_val_fold.to_dict(orient='records')
            X_val = dv.transform(val_dict)
            
            model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
            model.fit(X_train, y_train_fold)
            
            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val_fold, y_pred)
            scores.append(auc)
        
        scores = np.array(scores)
        mean_auc = np.mean(scores)
        std_auc = np.std(scores)
        results.append({
            'C': C,
            'mean_auc': round(mean_auc, 3),
            'std_auc': round(std_auc, 3)
        })
        print(f"  C={C}: mean AUC={mean_auc:.6f} (rounded: {round(mean_auc, 3)}), std={std_auc:.6f} (rounded: {round(std_auc, 3)})")
    
    # Find best C (highest mean, then lowest std, then smallest C)
    best = max(results, key=lambda x: (x['mean_auc'], -x['std_auc'], -x['C']))
    print(f"\n  Best C: {best['C']} (mean AUC={best['mean_auc']}, std={best['std_auc']})")
    
    return best['C']


def main():
    print("=" * 60)
    print("Homework 4: Evaluation Metrics for Classification - Solution")
    print("DataTalks.Club ML Zoomcamp 2026")
    print("=" * 60)
    
    data = load_data()
    print(f"\nDataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Target distribution:\n{data['converted'].value_counts()}")
    
    df_train, df_val, df_test, y_train, y_val, y_test, y_full_train, cat, numeric = prepare_data(data)
    
    print("\n" + "=" * 60)
    print("ANSWERS")
    print("=" * 60)
    
    q1 = question_1_roc_auc_feature_importance(df_train, y_train, numeric)
    
    # Train model for Q2-Q4
    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_train, y_train)
    
    q2 = question_2_train_model(df_train, df_val, y_train, y_val, cat, numeric)
    q3 = question_3_precision_recall(df_val, y_val, model, dv)
    q4 = question_4_f1_score(df_val, y_val, model, dv)
    q5 = question_5_cross_validation(df_train, y_train, cat, numeric)  # Note: df_full_train is df_train here
    q6 = question_6_hyperparameter_tuning(df_train, y_train, cat, numeric)  # Same
    
    print("\n" + "=" * 60)
    print("SUMMARY OF ANSWERS")
    print("=" * 60)
    print(f"Q1. Highest AUC numerical variable: {q1}")
    print(f"Q2. Validation AUC (rounded): {q2:.3f}")
    print(f"Q3. Precision/Recall intersection threshold: {q3}")
    print(f"Q4. Max F1 threshold: {q4}")
    print(f"Q5. 5-Fold CV std: {q5}")
    print(f"Q6. Best C: {q6}")


if __name__ == "__main__":
    main()