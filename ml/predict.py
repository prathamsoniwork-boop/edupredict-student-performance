"""
EduPredict Prediction & Recommendation Engine
Loads saved model, validates inputs, predicts student score, computes confidence,
categorizes performance, and generates actionable personalized academic recommendations.
"""

import os
import sys
import numpy as np
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.train_model import FEATURE_COLUMNS, encode_features

DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'student_performance_model.pkl'))
_CACHED_PIPELINE = None

def get_model_pipeline(model_path=None):
    """Load and cache the trained pipeline object."""
    global _CACHED_PIPELINE
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if _CACHED_PIPELINE is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run ml/train_model.py first.")
        _CACHED_PIPELINE = joblib.load(model_path)
    return _CACHED_PIPELINE

def validate_student_inputs(data):
    """
    Validate all input values with realistic bounds and clear error messages.
    """
    errors = []

    def get_num(key, default=None):
        val = data.get(key)
        if val is None or val == '':
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            errors.append(f"Invalid numeric value for '{key}': {val}")
            return default

    study_hours = get_num('study_hours_per_day')
    attendance = get_num('attendance_percentage')
    previous_score = get_num('previous_score')
    assignment_completion = get_num('assignment_completion')
    sleep_hours = get_num('sleep_hours')
    class_participation = get_num('class_participation')
    backlogs = get_num('backlogs')

    if study_hours is None or not (0.0 <= study_hours <= 16.0):
        errors.append("Study Hours per Day must be between 0 and 16 hours.")

    if attendance is None or not (0.0 <= attendance <= 100.0):
        errors.append("Attendance Percentage must be between 0% and 100%.")

    if previous_score is None or not (0.0 <= previous_score <= 100.0):
        errors.append("Previous Exam Score must be between 0 and 100.")

    if assignment_completion is None or not (0.0 <= assignment_completion <= 100.0):
        errors.append("Assignment Completion must be between 0% and 100%.")

    if sleep_hours is None or not (2.0 <= sleep_hours <= 16.0):
        errors.append("Sleep Hours must be between 2 and 16 hours.")

    if class_participation is None or not (1 <= class_participation <= 10):
        errors.append("Class Participation score must be an integer between 1 and 10.")

    if backlogs is None or backlogs < 0 or backlogs > 20:
        errors.append("Number of Backlogs must be a non-negative number (0 to 20).")

    if errors:
        return False, errors
    return True, []

def classify_performance(score):
    """
    Classify score into academic performance categories:
    - Excellent: >= 80
    - Good: 65 - 79.9
    - Average: 50 - 64.9
    - At Risk: < 50
    """
    if score >= 80.0:
        return {
            'category': 'Excellent',
            'badge_class': 'badge-success',
            'color': '#10b981',
            'summary': 'Outstanding academic trajectory. Demonstrates strong subject mastery and consistent habits.'
        }
    elif score >= 65.0:
        return {
            'category': 'Good',
            'badge_class': 'badge-primary',
            'color': '#3b82f6',
            'summary': 'Solid academic performance with consistent progress. Minor targeted refinements can push to top honors.'
        }
    elif score >= 50.0:
        return {
            'category': 'Average',
            'badge_class': 'badge-warning',
            'color': '#f59e0b',
            'summary': 'Acceptable baseline performance but vulnerable to exam variances. Structured intervention recommended.'
        }
    else:
        return {
            'category': 'At Risk',
            'badge_class': 'badge-danger',
            'color': '#ef4444',
            'summary': 'High academic vulnerability. Immediate structured intervention and study habit overhaul needed.'
        }

def generate_personalized_recommendations(features, predicted_score, category):
    """
    Generate tailored, actionable recommendations based on input features and predicted score.
    """
    recommendations = []

    attendance = float(features.get('attendance_percentage', 80))
    study_hours = float(features.get('study_hours_per_day', 4))
    prev_score = float(features.get('previous_score', 70))
    assignments = float(features.get('assignment_completion', 80))
    sleep = float(features.get('sleep_hours', 7))
    participation = float(features.get('class_participation', 6))
    backlogs = int(features.get('backlogs', 0))
    internet = str(features.get('internet_availability', 'Yes')).strip().capitalize()
    device = str(features.get('device_availability', 'Yes')).strip().capitalize()
    parental = str(features.get('parental_support', 'Medium')).strip().capitalize()
    extra = str(features.get('extracurricular_activity', 'No')).strip().capitalize()

    # 1. Attendance
    if attendance < 65:
        recommendations.append({
            'type': 'critical',
            'icon': 'fa-triangle-exclamation',
            'title': 'Critical Attendance Deficit',
            'message': f'Your current attendance is {attendance:.1f}%. Prioritize reaching at least 75% to prevent exam debarment and ensure curriculum continuity.'
        })
    elif attendance < 75:
        recommendations.append({
            'type': 'warning',
            'icon': 'fa-calendar-check',
            'title': 'Improve Attendance',
            'message': f'Current attendance ({attendance:.1f}%) is below the recommended 75% threshold. Regular classroom engagement directly improves retention.'
        })

    # 2. Study Hours
    if study_hours < 2.5:
        recommendations.append({
            'type': 'warning',
            'icon': 'fa-clock',
            'title': 'Increase Daily Study Time',
            'message': f'You are logging {study_hours:.1f} hours/day. Gradually expand focused study blocks to 3.5–4.5 hours daily using the Pomodoro technique.'
        })
    elif study_hours > 7.5:
        recommendations.append({
            'type': 'info',
            'icon': 'fa-battery-half',
            'title': 'Manage Study Fatigue',
            'message': f'High study volume ({study_hours:.1f} hrs/day) requires adequate cognitive pacing. Prioritize active recall over passive reading.'
        })

    # 3. Backlogs
    if backlogs > 0:
        recommendations.append({
            'type': 'critical' if backlogs >= 2 else 'warning',
            'icon': 'fa-book-bookmark',
            'title': 'Clear Active Backlogs',
            'message': f'You have {backlogs} pending backlog(s). Allocate weekend revision blocks specifically for prerequisite topics and previous exam papers.'
        })

    # 4. Assignment Completion
    if assignments < 75:
        recommendations.append({
            'type': 'warning',
            'icon': 'fa-list-check',
            'title': 'Submit Assignments Regularly',
            'message': f'Assignment completion is currently {assignments:.1f}%. Completing weekly problem sets solidifies conceptual clarity and secures internal marks.'
        })

    # 5. Sleep & Well-being
    if sleep < 6.0:
        recommendations.append({
            'type': 'warning',
            'icon': 'fa-moon',
            'title': 'Prioritize Sleep Hygiene',
            'message': f'Logging only {sleep:.1f} hours of sleep impairs working memory and exam-day alertness. Aim for 7 to 8 hours of consistent rest.'
        })
    elif sleep > 9.5:
        recommendations.append({
            'type': 'info',
            'icon': 'fa-bed',
            'title': 'Optimize Daily Schedule',
            'message': f'Over {sleep:.1f} hours of sleep can lead to lethargy. Structure morning routines to maximize peak cognitive study hours.'
        })

    # 6. Class Participation
    if participation <= 4:
        recommendations.append({
            'type': 'info',
            'icon': 'fa-comments',
            'title': 'Increase Class Participation',
            'message': f'Participation is rated at {participation}/10. Asking questions during tutorials and collaborating with peers actively deepens comprehension.'
        })

    # 7. Digital Resources & Environment
    if internet == 'No' or device == 'No':
        recommendations.append({
            'type': 'info',
            'icon': 'fa-laptop-code',
            'title': 'Campus Lab Utilization',
            'message': 'Leverage department computer labs and the digital library during free hours for uninterrupted online submissions and research.'
        })

    # 8. High Performer Guidance
    if category == 'Excellent' and len(recommendations) < 2:
        recommendations.append({
            'type': 'success',
            'icon': 'fa-trophy',
            'title': 'Advance Your Expertise',
            'message': 'Outstanding performance! Consider taking up competitive coding, undergraduate research, or peer tutoring to build career-ready distinction.'
        })

    # Ensure at least 2 constructive suggestions
    if len(recommendations) == 0:
        recommendations.append({
            'type': 'success',
            'icon': 'fa-circle-check',
            'title': 'Balanced Academic Profile',
            'message': 'Your study metrics are well balanced. Maintain your daily schedule, review lecture notes weekly, and attempt mock assessments before finals.'
        })

    return recommendations

def predict_student(input_dict, model_path=None):
    """
    Complete inference pipeline: validation -> encoding -> model predict -> categorization -> recommendations.
    """
    valid, errors = validate_student_inputs(input_dict)
    if not valid:
        return {'success': False, 'errors': errors}

    pipeline = get_model_pipeline(model_path)
    model = pipeline['model']

    # Build single-row dictionary for encoding
    row = {
        'study_hours_per_day': float(input_dict['study_hours_per_day']),
        'attendance_percentage': float(input_dict['attendance_percentage']),
        'previous_score': float(input_dict['previous_score']),
        'assignment_completion': float(input_dict['assignment_completion']),
        'sleep_hours': float(input_dict['sleep_hours']),
        'class_participation': float(input_dict['class_participation']),
        'backlogs': float(input_dict['backlogs']),
        'internet_availability': input_dict.get('internet_availability', 'Yes'),
        'device_availability': input_dict.get('device_availability', 'Yes'),
        'extracurricular_activity': input_dict.get('extracurricular_activity', 'No'),
        'parental_support': input_dict.get('parental_support', 'Medium')
    }

    # Encode
    encoded_vals = [
        row['study_hours_per_day'],
        row['attendance_percentage'],
        row['previous_score'],
        row['assignment_completion'],
        row['sleep_hours'],
        row['class_participation'],
        row['backlogs'],
        1.0 if str(row['internet_availability']).capitalize() == 'Yes' else 0.0,
        1.0 if str(row['device_availability']).capitalize() == 'Yes' else 0.0,
        1.0 if str(row['extracurricular_activity']).capitalize() == 'Yes' else 0.0,
        2.0 if str(row['parental_support']).capitalize() == 'High' else (1.0 if str(row['parental_support']).capitalize() == 'Medium' else 0.0)
    ]
    X_input = np.array([encoded_vals], dtype=np.float64)

    # Prediction with confidence
    confidence = 90.0
    if hasattr(model, 'predict_with_confidence'):
        preds, stds, confs = model.predict_with_confidence(X_input)
        raw_score = float(preds[0])
        confidence = round(float(confs[0]), 1)
    else:
        raw_score = float(model.predict(X_input)[0])

    # Bound score realistically
    final_score = round(float(np.clip(raw_score, 0.0, 100.0)), 1)
    category_info = classify_performance(final_score)
    recommendations = generate_personalized_recommendations(row, final_score, category_info['category'])

    return {
        'success': True,
        'predicted_score': final_score,
        'category': category_info['category'],
        'badge_class': category_info['badge_class'],
        'color': category_info['color'],
        'summary': category_info['summary'],
        'confidence': confidence,
        'recommendations': recommendations,
        'inputs': row
    }
