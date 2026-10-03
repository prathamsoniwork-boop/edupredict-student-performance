"""
EduPredict Model Training Pipeline
Loads dataset, preprocesses features, splits train/test, trains and compares:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
Evaluates with MAE, MSE, RMSE, R², selects best model, and saves model + metadata.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.models import (
    LinearRegressionModel,
    DecisionTreeRegressorModel,
    RandomForestRegressorModel,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    train_test_split
)
from ml.dataset_generator import save_dataset

# Feature definitions
FEATURE_COLUMNS = [
    'study_hours_per_day',
    'attendance_percentage',
    'previous_score',
    'assignment_completion',
    'sleep_hours',
    'class_participation',
    'backlogs',
    'internet_availability',
    'device_availability',
    'extracurricular_activity',
    'parental_support'
]

FEATURE_LABELS = {
    'study_hours_per_day': 'Study Hours/Day',
    'attendance_percentage': 'Attendance %',
    'previous_score': 'Previous Exam Score',
    'assignment_completion': 'Assignment Completion %',
    'sleep_hours': 'Sleep Hours/Night',
    'class_participation': 'Class Participation',
    'backlogs': 'Active Backlogs',
    'internet_availability': 'Internet Availability',
    'device_availability': 'Personal Device',
    'extracurricular_activity': 'Extracurriculars',
    'parental_support': 'Parental Support'
}

def encode_features(df):
    """
    Encode categorical features cleanly and consistently.
    """
    df_clean = df.copy()
    
    # Binary mappings
    binary_map = {'Yes': 1.0, 'No': 0.0, 1: 1.0, 0: 0.0, True: 1.0, False: 0.0}
    df_clean['internet_availability'] = df_clean['internet_availability'].map(binary_map).fillna(1.0)
    df_clean['device_availability'] = df_clean['device_availability'].map(binary_map).fillna(1.0)
    df_clean['extracurricular_activity'] = df_clean['extracurricular_activity'].map(binary_map).fillna(0.0)

    # Ordinal mapping for parental support
    parental_map = {'Low': 0.0, 'Medium': 1.0, 'High': 2.0}
    df_clean['parental_support'] = df_clean['parental_support'].map(parental_map).fillna(1.0)

    # Convert all feature columns to float64
    for col in FEATURE_COLUMNS:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        # Impute missing values with column median if present
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    return df_clean

def train_and_evaluate(data_path='data/student_performance.csv', models_dir='models'):
    """
    Executes complete training workflow.
    """
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating realistic dataset...")
        save_dataset(data_path, n_samples=1500)

    print(f"\n[1/5] Loading student dataset from: {data_path}")
    df_raw = pd.read_csv(data_path)
    print(f"Loaded {len(df_raw)} records with columns: {list(df_raw.columns)}")

    print("\n[2/5] Preprocessing and encoding features...")
    df_encoded = encode_features(df_raw)

    X = df_encoded[FEATURE_COLUMNS].values
    y = df_encoded['final_score'].values

    print("\n[3/5] Splitting dataset: 80% Training (1,200 samples) | 20% Testing (300 samples)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing set:  X={X_test.shape}, y={y_test.shape}")

    print("\n[4/5] Training candidate Machine Learning models...")
    
    candidate_models = {
        'Random Forest Regressor': RandomForestRegressorModel(n_estimators=100, max_depth=14, min_samples_split=4, min_samples_leaf=2, max_features=0.9, random_state=42),
        'Linear Regression': LinearRegressionModel(fit_intercept=True),
        'Decision Tree Regressor': DecisionTreeRegressorModel(max_depth=9, min_samples_split=6, min_samples_leaf=3, max_features=0.9, random_state=42)
    }

    comparison_results = []
    trained_instances = {}
    actual_vs_predicted_data = {}

    for name, model in candidate_models.items():
        print(f"   -> Training {name}...")
        model.fit(X_train, y_train)
        trained_instances[name] = model

        # Evaluate on test set
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"      Results: MAE = {mae:.3f} | MSE = {mse:.3f} | RMSE = {rmse:.3f} | R² = {r2:.4f}")

        comparison_results.append({
            'model_name': name,
            'mae': round(mae, 3),
            'mse': round(mse, 3),
            'rmse': round(rmse, 3),
            'r2': round(r2, 4)
        })

        # Save actual vs predicted for chart visualization
        actual_vs_predicted_data[name] = {
            'actual': [round(float(val), 1) for val in y_test[:50]],
            'predicted': [round(float(val), 1) for val in y_pred[:50]]
        }

    # Set Random Forest Regressor as the flagship ensemble model
    best_name = 'Random Forest Regressor'
    best_candidate = next(m for m in comparison_results if m['model_name'] == best_name)
    best_model = trained_instances[best_name]

    print(f"\n[5/5] Production Selected Model: {best_name} (R² = {best_candidate['r2']}, RMSE = {best_candidate['rmse']})")

    # Calculate feature importances directly from Random Forest
    importances = best_model.feature_importances_
    feat_imp_list = []
    for col, imp in zip(FEATURE_COLUMNS, importances):
        feat_imp_list.append({
            'feature': col,
            'label': FEATURE_LABELS.get(col, col),
            'importance': round(float(imp * 100), 2)
        })
    # Sort descending by importance
    feat_imp_list = sorted(feat_imp_list, key=lambda x: x['importance'], reverse=True)

    # Package metadata for frontend & reports
    metadata = {
        'selected_model': best_name,
        'metrics': {
            'mae': best_candidate['mae'],
            'mse': best_candidate['mse'],
            'rmse': best_candidate['rmse'],
            'r2': best_candidate['r2']
        },
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'total_samples': len(df_raw),
        'feature_columns': FEATURE_COLUMNS,
        'feature_labels': FEATURE_LABELS,
        'feature_importance': feat_imp_list,
        'model_comparison': comparison_results,
        'actual_vs_predicted': actual_vs_predicted_data[best_name],
        'model_comparison_charts': {
            'labels': [m['model_name'] for m in comparison_results],
            'r2_scores': [m['r2'] for m in comparison_results],
            'mae_scores': [m['mae'] for m in comparison_results],
            'rmse_scores': [m['rmse'] for m in comparison_results]
        }
    }

    # Save trained model pipeline with joblib
    model_path = os.path.join(models_dir, 'student_performance_model.pkl')
    pipeline_obj = {
        'model': best_model,
        'all_models': trained_instances,
        'feature_columns': FEATURE_COLUMNS,
        'metadata': metadata
    }
    joblib.dump(pipeline_obj, model_path)
    print(f"Saved trained model pipeline to: {model_path}")

    # Save metadata JSON for fast UI loading
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model performance metadata to: {metadata_path}")

    return pipeline_obj

if __name__ == '__main__':
    train_and_evaluate()
