"""
EduPredict - Student Performance Prediction System
Flask Web Application & REST API Server
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

import database
from ml.predict import predict_student
from ml.train_model import FEATURE_COLUMNS, FEATURE_LABELS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.secret_key = 'edupredict-ai-secret-key-2026'
handler = app

# Initialize database on startup
database.init_db()

def get_metadata():
    meta_path = os.path.join(BASE_DIR, 'models', 'model_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}

# -------------------------------------------------------------
# PAGE ROUTES
# -------------------------------------------------------------

@app.route('/')
def index():
    """Landing / Dashboard page with overview KPIs and quick actions."""
    metadata = get_metadata()
    kpis = database.get_dashboard_kpis()
    recent_predictions = database.get_all_predictions(limit=5)
    
    # Dataset statistics
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'student_performance.csv')
    dataset_summary = {'total_records': 1500, 'mean_score': 63.3}
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            dataset_summary['total_records'] = len(df)
            dataset_summary['mean_score'] = round(df['final_score'].mean(), 1)
        except Exception:
            pass

    return render_template(
        'index.html',
        metadata=metadata,
        kpis=kpis,
        recent_predictions=recent_predictions,
        dataset_summary=dataset_summary
    )

@app.route('/predict', methods=['GET'])
def predict_page():
    """Prediction page with interactive input form and demo presets."""
    return render_template('predict.html', feature_labels=FEATURE_LABELS)

@app.route('/model-performance')
def model_performance():
    """Model performance and evaluation analytics page."""
    metadata = get_metadata()
    return render_template('performance.html', metadata=metadata)

@app.route('/insights')
def student_insights():
    """Student dataset analytics, distribution charts, and risk factors."""
    metadata = get_metadata()
    return render_template('insights.html', metadata=metadata)

@app.route('/about')
def about():
    """About project page detailing problem, solution, ML pipeline, and tech stack."""
    return render_template('about.html')

@app.route('/history')
def history_page():
    """Full prediction history log with search, filters, and deletion controls."""
    predictions = database.get_all_predictions(limit=200)
    return render_template('history.html', predictions=predictions)

# -------------------------------------------------------------
# REST API ENDPOINTS
# -------------------------------------------------------------

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """
    POST /predict
    Accepts JSON or form data, predicts score, computes recommendations, and persists record.
    """
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    student_name = data.get('student_name', 'Student').strip() or 'Anonymous Student'
    
    # Run prediction through ML pipeline
    result = predict_student(data)
    
    if not result.get('success'):
        return jsonify(result), 400

    # Persist in SQLite
    new_id = database.save_prediction(student_name, result['inputs'], result)
    result['id'] = new_id
    result['student_name'] = student_name

    return jsonify(result)

@app.route('/api/model-performance', methods=['GET'])
def api_model_performance():
    """GET /api/model-performance - returns model metrics, feature importances, and chart data."""
    metadata = get_metadata()
    return jsonify(metadata)

@app.route('/api/demo-data', methods=['GET'])
def api_demo_data():
    """
    GET /api/demo-data
    Returns multiple realistic student profiles for instant form auto-fill.
    """
    presets = {
        'recommended': {
            'student_name': 'Aarav Patel',
            'study_hours_per_day': 4.5,
            'attendance_percentage': 88.0,
            'previous_score': 78.0,
            'assignment_completion': 92.0,
            'sleep_hours': 7.0,
            'class_participation': 8,
            'backlogs': 0,
            'internet_availability': 'Yes',
            'device_availability': 'Yes',
            'extracurricular_activity': 'Yes',
            'parental_support': 'High'
        },
        'high_achiever': {
            'student_name': 'Meera Sen',
            'study_hours_per_day': 6.5,
            'attendance_percentage': 96.0,
            'previous_score': 89.0,
            'assignment_completion': 98.0,
            'sleep_hours': 7.5,
            'class_participation': 9,
            'backlogs': 0,
            'internet_availability': 'Yes',
            'device_availability': 'Yes',
            'extracurricular_activity': 'Yes',
            'parental_support': 'High'
        },
        'average_student': {
            'student_name': 'Karan Gupta',
            'study_hours_per_day': 3.2,
            'attendance_percentage': 76.0,
            'previous_score': 62.0,
            'assignment_completion': 70.0,
            'sleep_hours': 6.5,
            'class_participation': 5,
            'backlogs': 1,
            'internet_availability': 'Yes',
            'device_availability': 'Yes',
            'extracurricular_activity': 'No',
            'parental_support': 'Medium'
        },
        'at_risk_student': {
            'student_name': 'Vikram Rao',
            'study_hours_per_day': 1.5,
            'attendance_percentage': 52.0,
            'previous_score': 42.0,
            'assignment_completion': 45.0,
            'sleep_hours': 5.0,
            'class_participation': 2,
            'backlogs': 3,
            'internet_availability': 'No',
            'device_availability': 'No',
            'extracurricular_activity': 'No',
            'parental_support': 'Low'
        }
    }
    return jsonify(presets)

@app.route('/api/insights', methods=['GET'])
def api_insights():
    """
    GET /api/insights
    Aggregates dataset distributions (attendance, study hours, categories, correlations).
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'student_performance.csv')
    if not os.path.exists(csv_path):
        return jsonify({'error': 'Dataset not found'}), 404

    df = pd.read_csv(csv_path)

    # Score category distribution
    def get_cat(s):
        if s >= 80: return 'Excellent'
        elif s >= 65: return 'Good'
        elif s >= 50: return 'Average'
        else: return 'At Risk'

    categories = df['final_score'].apply(get_cat).value_counts().to_dict()

    # Attendance distribution (10% bins)
    att_bins = [40, 50, 60, 70, 80, 90, 100]
    att_labels = ['40-49%', '50-59%', '60-69%', '70-79%', '80-89%', '90-100%']
    att_dist = pd.cut(df['attendance_percentage'], bins=att_bins, labels=att_labels).value_counts(sort=False).to_dict()

    # Study hours distribution (2-hour bins)
    study_bins = [0, 2, 4, 6, 8, 11]
    study_labels = ['0-2 hrs', '2-4 hrs', '4-6 hrs', '6-8 hrs', '8+ hrs']
    study_dist = pd.cut(df['study_hours_per_day'], bins=study_bins, labels=study_labels).value_counts(sort=False).to_dict()

    # Average score by backlogs
    backlog_impact = df.groupby('backlogs')['final_score'].mean().round(1).to_dict()

    # Key contributing factors / correlations with final score
    numeric_cols = [
        'study_hours_per_day', 'attendance_percentage', 'previous_score',
        'assignment_completion', 'sleep_hours', 'class_participation', 'backlogs'
    ]
    correlations = df[numeric_cols].apply(lambda col: round(float(col.corr(df['final_score'])), 3)).to_dict()

    return jsonify({
        'total_records': len(df),
        'average_score': round(float(df['final_score'].mean()), 1),
        'min_score': round(float(df['final_score'].min()), 1),
        'max_score': round(float(df['final_score'].max()), 1),
        'category_distribution': categories,
        'attendance_distribution': att_dist,
        'study_hours_distribution': study_dist,
        'backlog_impact': backlog_impact,
        'correlations': correlations
    })

@app.route('/prediction-history', methods=['GET'])
def get_prediction_history():
    """GET /prediction-history - returns JSON of stored predictions."""
    predictions = database.get_all_predictions(limit=100)
    return jsonify(predictions)

@app.route('/prediction-history/<int:pred_id>', methods=['DELETE'])
def delete_prediction(pred_id):
    """DELETE /prediction-history/<id> - delete a single prediction."""
    success = database.delete_prediction_by_id(pred_id)
    if success:
        return jsonify({'success': True, 'message': f'Prediction #{pred_id} deleted.'})
    return jsonify({'success': False, 'message': 'Record not found'}), 404

@app.route('/prediction-history', methods=['DELETE'])
def clear_prediction_history():
    """DELETE /prediction-history - delete all prediction history."""
    database.clear_all_predictions()
    return jsonify({'success': True, 'message': 'All prediction history cleared.'})

if __name__ == '__main__':
    print("EduPredict server starting on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
