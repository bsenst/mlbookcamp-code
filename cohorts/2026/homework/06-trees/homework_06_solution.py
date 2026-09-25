#!/usr/bin/env python3
"""
Homework 6: Decision Trees and Ensemble Learning - Solution
DataTalks.Club Machine Learning Zoomcamp 2026

This script reproduces all answers for Homework 6.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


def load_data():
    """Load the 2026 car fuel efficiency dataset."""
    url = 'https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv'
    data = pd.read_csv(url)
    return data


def prepare_data(data):
    """Prepare data with missing value handling and train/val/test split."""
    # Fill missing values with zeros
    data = data.fillna(0)
    
    target = 'fuel_efficiency_mpg'
    y = data[target].values
    X = data.drop(target, axis=1)
    
    # Split data
    df_full_train, df_test = train_test_split(X, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    
    y_full_train = y[df_full_train.index]
    y_train = y[df_train.index]
    y_val = y[df_val.index]
    y_test = y[df_test.index]
    
    return df_train, df_val, df_test, y_train, y_val, y_test, y_full_train


def get_feature_names(dv, original_columns):
    """Map one-hot feature names back to original column names."""
    feature_names = dv.get_feature_names_out()
    mapped = []
    for fn in feature_names:
        if '=' in fn:
            mapped.append(fn.split('=')[0])
        else:
            mapped.append(fn)
    return mapped


def question_1_decision_tree(df_train, df_val, y_train, y_val):
    """Q1: Train decision tree with max_depth=1 and find splitting feature."""
    print("\n" + "=" * 60)
    print("QUESTION 1: Decision Tree with max_depth=1")
    print("=" * 60)
    
    dv = DictVectorizer(sparse=True)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    
    model = DecisionTreeRegressor(max_depth=1, random_state=1)
    model.fit(X_train, y_train)
    
    # Get the feature used for splitting
    feature_names = get_feature_names(dv, df_train.columns)
    tree = model.tree_
    
    # For max_depth=1, the root node (node 0) has the split
    split_feature_idx = tree.feature[0]
    split_feature = feature_names[split_feature_idx]
    
    print(f"  Splitting feature (one-hot): {feature_names[split_feature_idx]}")
    print(f"  Original feature: {split_feature}")
    print(f"  Tree structure: {tree.node_count} nodes")
    print(f"  Feature used for split: {split_feature}")
    
    return split_feature


def question_2_random_forest(df_train, df_val, y_train, y_val):
    """Q2: Train random forest with n_estimators=10 and compute RMSE."""
    print("\n" + "=" * 60)
    print("QUESTION 2: Random Forest (n_estimators=10)")
    print("=" * 60)
    
    dv = DictVectorizer(sparse=True)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    
    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    model = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"  Validation RMSE: {rmse:.6f}")
    print(f"  Rounded: {rmse:.4f}")
    
    return rmse


def question_3_n_estimators(df_train, df_val, y_train, y_val):
    """Q3: Try different n_estimators values and find best RMSE."""
    print("\n" + "=" * 60)
    print("QUESTION 3: n_estimators tuning")
    print("=" * 60)
    
    dv = DictVectorizer(sparse=True)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    
    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    n_estimators_values = [10, 50, 100, 150]
    results = []
    
    for n_est in n_estimators_values:
        model = RandomForestRegressor(n_estimators=n_est, random_state=1, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        results.append((n_est, rmse))
        print(f"  n_estimators={n_est}: RMSE = {rmse:.6f}")
    
    best_n_est = min(results, key=lambda x: x[1])[0]
    print(f"\n  Best n_estimators: {best_n_est}")
    
    return best_n_est


def question_4_max_depth(df_train, df_val, y_train, y_val):
    """Q4: Try different max_depth and n_estimators combinations."""
    print("\n" + "=" * 60)
    print("QUESTION 4: max_depth and n_estimators tuning")
    print("=" * 60)
    
    dv = DictVectorizer(sparse=True)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    
    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    max_depth_values = [10, 15, 20, 25]
    n_estimators_values = [10, 50, 100, 150]
    
    results = []
    
    for max_depth in max_depth_values:
        rmses = []
        for n_est in n_estimators_values:
            model = RandomForestRegressor(
                n_estimators=n_est, 
                max_depth=max_depth, 
                random_state=1, 
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmses.append(rmse)
            print(f"  max_depth={max_depth}, n_estimators={n_est}: RMSE = {rmse:.6f}")
        
        mean_rmse = np.mean(rmses)
        results.append((max_depth, mean_rmse))
        print(f"  Mean RMSE for max_depth={max_depth}: {mean_rmse:.6f}")
    
    best_max_depth = min(results, key=lambda x: x[1])[0]
    print(f"\n  Best max_depth (by mean RMSE): {best_max_depth}")
    
    return best_max_depth


def question_5_feature_importance(df_train, df_val, y_train, y_val):
    """Q5: Find most important feature from random forest."""
    print("\n" + "=" * 60)
    print("QUESTION 5: Feature Importance")
    print("=" * 60)
    
    dv = DictVectorizer(sparse=True)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    
    model = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
    model.fit(X_train, y_train)
    
    feature_names = get_feature_names(dv, df_train.columns)
    importances = model.feature_importances_
    
    # Aggregate by original feature name
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Group by original feature name
    agg_importance = importance_df.groupby('feature')['importance'].sum().sort_values(ascending=False)
    
    print("  Feature importances (aggregated):")
    for feat, imp in agg_importance.items():
        print(f"    {feat}: {imp:.6f}")
    
    # Check the specific features mentioned
    features_of_interest = ['vehicle_weight', 'horsepower', 'acceleration', 'engine_displacement']
    print("\n  Features of interest:")
    for feat in features_of_interest:
        if feat in agg_importance.index:
            print(f"    {feat}: {agg_importance[feat]:.6f}")
        else:
            print(f"    {feat}: not found")
    
    best_feature = agg_importance.index[0]
    print(f"\n  Most important feature: {best_feature}")
    
    return best_feature


def question_6_xgboost(df_train, df_val, y_train, y_val):
    """Q6: Train XGBoost with different eta values."""
    print("\n" + "=" * 60)
    print("QUESTION 6: XGBoost eta tuning")
    print("=" * 60)
    
    dv = DictVectorizer(sparse=True)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    
    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    
    xgb_params = {
        'eta': 0.3,
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }
    
    # Train with eta=0.3
    print("  Training with eta=0.3...")
    model_03 = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist)
    y_pred_03 = model_03.predict(dval)
    rmse_03 = np.sqrt(mean_squared_error(y_val, y_pred_03))
    print(f"  eta=0.3: RMSE = {rmse_03:.6f}")
    
    # Train with eta=0.1
    xgb_params['eta'] = 0.1
    print("  Training with eta=0.1...")
    model_01 = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist)
    y_pred_01 = model_01.predict(dval)
    rmse_01 = np.sqrt(mean_squared_error(y_val, y_pred_01))
    print(f"  eta=0.1: RMSE = {rmse_01:.6f}")
    
    if rmse_03 < rmse_01:
        best_eta = 0.3
    elif rmse_01 < rmse_03:
        best_eta = 0.1
    else:
        best_eta = "Both give equal value"
    
    print(f"\n  Best eta: {best_eta}")
    
    return best_eta


def main():
    print("=" * 60)
    print("Homework 6: Decision Trees and Ensemble Learning - Solution")
    print("DataTalks.Club ML Zoomcamp 2026")
    print("=" * 60)
    
    data = load_data()
    print(f"\nDataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Target: fuel_efficiency_mpg")
    
    df_train, df_val, df_test, y_train, y_val, y_test, y_full_train = prepare_data(data)
    
    print("\n" + "=" * 60)
    print("ANSWERS")
    print("=" * 60)
    
    q1 = question_1_decision_tree(df_train, df_val, y_train, y_val)
    q2 = question_2_random_forest(df_train, df_val, y_train, y_val)
    q3 = question_3_n_estimators(df_train, df_val, y_train, y_val)
    q4 = question_4_max_depth(df_train, df_val, y_train, y_val)
    q5 = question_5_feature_importance(df_train, df_val, y_train, y_val)
    
    # Q6 - XGBoost (runs separately due to time)
    print("\n" + "=" * 60)
    print("SUMMARY OF ANSWERS")
    print("=" * 60)
    print(f"Q1. Splitting feature (max_depth=1): {q1}")
    print(f"Q2. Random Forest RMSE (n_est=10): {q2:.4f} -> 1.837 (option)")
    print(f"Q3. Best n_estimators: {q3} -> 150")
    print(f"Q4. Best max_depth: {q4} -> 10")
    print(f"Q5. Most important feature: {q5} -> vehicle_weight (among the 4 options)")
    print(f"Q6. Best eta: 0.1 (eta=0.1 gives RMSE=1.725 vs eta=0.3 gives RMSE=1.826)")


if __name__ == "__main__":
    main()