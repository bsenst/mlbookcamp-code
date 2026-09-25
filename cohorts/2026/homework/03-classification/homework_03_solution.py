#!/usr/bin/env python3
"""
Homework 3: Machine Learning for Classification - Solution
DataTalks.Club Machine Learning Zoomcamp 2026

This script reproduces all answers for Homework 3.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score


def load_data():
    """Load the course lead scoring dataset."""
    url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv'
    data = pd.read_csv(url)
    return data


def question_1_mode_industry(data):
    """Q1: What is the most frequent observation (mode) for the column `industry`?"""
    mode = data['industry'].mode()[0]
    counts = data['industry'].value_counts()
    print(f"Q1: Mode for industry = {mode} (count: {counts[mode]})")
    print(f"   All counts:\n{counts}")
    return mode


def question_2_biggest_correlation(data):
    """Q2: What are the two features that have the biggest correlation?"""
    data_numeric = data.drop(
        ['lead_source', 'industry', 'employment_status', 'location', 'converted'],
        axis=1
    )
    corr_matrix = data_numeric.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    max_corr_pair = upper.unstack().sort_values(ascending=False).index[0]
    max_corr_value = upper.unstack().sort_values(ascending=False).iloc[0]
    print(f"Q2: Biggest correlation = {max_corr_pair} ({max_corr_value:.6f})")
    return max_corr_pair


def question_3_biggest_mi(data):
    """Q3: Which categorical variable has the biggest mutual information score with target?"""
    SEED = 42
    df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=SEED)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=SEED)
    
    df_train = df_train.reset_index(drop=True)
    y_train = df_train.converted.values
    
    cat = ['lead_source', 'industry', 'employment_status', 'location']
    df_mi = df_train[cat].fillna('NA').apply(lambda s: mutual_info_score(s, y_train))
    df_mi = df_mi.sort_values(ascending=False)
    
    best_feature = df_mi.index[0]
    best_score = df_mi.iloc[0]
    print(f"Q3: Biggest MI = {best_feature} ({best_score:.6f})")
    print(f"   All MI scores:\n{df_mi}")
    return best_feature


def question_4_accuracy(data):
    """Q4: Train logistic regression and calculate accuracy on validation set."""
    SEED = 42
    df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=SEED)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=SEED)
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    y_train = df_train.converted.values
    y_val = df_val.converted.values
    
    df_train = df_train.drop('converted', axis=1)
    df_val = df_val.drop('converted', axis=1)
    
    cat = ['lead_source', 'industry', 'employment_status', 'location']
    numeric = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']
    
    df_train[cat] = df_train[cat].fillna('NA')
    df_train[numeric] = df_train[numeric].fillna(0)
    df_val[cat] = df_val[cat].fillna('NA')
    df_val[numeric] = df_val[numeric].fillna(0)
    
    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    y_pred = model.predict(X_val)
    accuracy = np.round(accuracy_score(y_val, y_pred), 2)
    
    print(f"Q4: Accuracy on validation set = {accuracy}")
    return accuracy


def question_5_feature_selection(data):
    """Q5: Which feature has the smallest difference when eliminated?"""
    SEED = 42
    df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=SEED)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=SEED)
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    
    y_train = df_train.converted.values
    y_val = df_val.converted.values
    
    df_train = df_train.drop('converted', axis=1)
    df_val = df_val.drop('converted', axis=1)
    
    cat = ['lead_source', 'industry', 'employment_status', 'location']
    numeric = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']
    
    df_train[cat] = df_train[cat].fillna('NA')
    df_train[numeric] = df_train[numeric].fillna(0)
    df_val[cat] = df_val[cat].fillna('NA')
    df_val[numeric] = df_val[numeric].fillna(0)
    
    # Baseline model with all features
    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)
    y_pred = model.predict(X_val)
    original_score = accuracy_score(y_val, y_pred)
    
    # Feature elimination (2026 questionnaire options)
    features = df_train.columns.to_list()
    eliminate = ['lead_source', 'number_of_courses_viewed', 'interaction_count']
    
    results = []
    for feature in eliminate:
        subset = features.copy()
        subset.remove(feature)
        
        dv = DictVectorizer(sparse=False)
        train_dict = df_train[subset].to_dict(orient='records')
        X_train = dv.fit_transform(train_dict)
        
        model = LogisticRegression(C=1, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        val_dict = df_val[subset].to_dict(orient='records')
        X_val = dv.transform(val_dict)
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        
        diff = original_score - score
        results.append({'feature': feature, 'accuracy': score, 'difference': diff})
        print(f"   Without {feature}: accuracy={score:.6f}, diff={diff:.6f}")
    
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['difference'].idxmin(), 'feature']
    print(f"Q5: Feature with smallest difference = {best}")
    return best


def question_6_parameter_tuning(data):
    """Q6: Which C value from the given options leads to best accuracy?"""
    SEED = 42
    df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=SEED)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=SEED)
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    
    y_train = df_train.converted.values
    y_val = df_val.converted.values
    
    df_train = df_train.drop('converted', axis=1)
    df_val = df_val.drop('converted', axis=1)
    
    cat = ['lead_source', 'industry', 'employment_status', 'location']
    numeric = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']
    
    df_train[cat] = df_train[cat].fillna('NA')
    df_train[numeric] = df_train[numeric].fillna(0)
    df_val[cat] = df_val[cat].fillna('NA')
    df_val[numeric] = df_val[numeric].fillna(0)
    
    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    # Test the exact C values from the homework form
    C_values = [0.000001, 0.00001, 0.0001, 0.001]
    best_C = None
    best_accuracy = 0
    
    for C in C_values:
        model = LogisticRegression(max_iter=1000, C=C, random_state=SEED, solver='lbfgs')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"   C={C}: accuracy={accuracy:.6f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = C
    
    print(f"Q6: Best C = {best_C} (accuracy={best_accuracy:.6f})")
    return best_C


def main():
    print("=" * 60)
    print("Homework 3: Machine Learning for Classification - Solution")
    print("DataTalks.Club ML Zoomcamp 2026")
    print("=" * 60)
    
    data = load_data()
    print(f"\nDataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Target distribution:\n{data['converted'].value_counts()}")
    
    print("\n" + "=" * 60)
    print("ANSWERS")
    print("=" * 60)
    
    q1 = question_1_mode_industry(data)
    q2 = question_2_biggest_correlation(data)
    q3 = question_3_biggest_mi(data)
    q4 = question_4_accuracy(data)
    q5 = question_5_feature_selection(data)
    q6 = question_6_parameter_tuning(data)
    
    print("\n" + "=" * 60)
    print("SUMMARY OF ANSWERS")
    print("=" * 60)
    print(f"1. Mode for industry: {q1}")
    print(f"2. Biggest correlation: {q2}")
    print(f"3. Biggest MI: {q3}")
    print(f"4. Accuracy: {q4}")
    print(f"5. Feature selection (smallest diff): {q5}")
    print(f"6. Parameter tuning (best C): {q6}")
    
    print("\n" + "=" * 60)
    print("MULTIPLE CHOICE MAPPING (2026 questionnaire)")
    print("=" * 60)
    print("1. Mode for industry: retail")
    print("2. Biggest correlation: annual_income and interaction_count")
    print("3. Biggest MI: lead_source")
    print("4. Accuracy: 0.85")
    print("5. Feature selection: lead_source")
    print("6. Parameter tuning: 0.001")


if __name__ == "__main__":
    main()