"""
EduPredict Database Management Module
Uses SQLite to store, retrieve, manage, and delete prediction records.
"""

import sqlite3
import os
import json
from datetime import datetime

import shutil

def get_db_path():
    """
    Returns appropriate database path.
    On Vercel serverless functions, the root filesystem is read-only,
    so we use /tmp/edupredict.db.
    """
    if os.environ.get('VERCEL'):
        tmp_db = '/tmp/edupredict.db'
        source_db = os.path.join(os.path.dirname(__file__), 'data', 'edupredict.db')
        if not os.path.exists(tmp_db) and os.path.exists(source_db):
            try:
                shutil.copy2(source_db, tmp_db)
            except Exception:
                pass
        return tmp_db
    return os.path.join(os.path.dirname(__file__), 'data', 'edupredict.db')

def get_db_connection():
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            study_hours REAL NOT NULL,
            attendance REAL NOT NULL,
            previous_score REAL NOT NULL,
            assignment_completion REAL NOT NULL,
            sleep_hours REAL NOT NULL,
            class_participation REAL NOT NULL,
            backlogs INTEGER NOT NULL,
            internet_availability TEXT NOT NULL,
            device_availability TEXT NOT NULL,
            extracurricular_activity TEXT NOT NULL,
            parental_support TEXT NOT NULL,
            predicted_score REAL NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            recommendations_json TEXT NOT NULL
        )
    ''')
    conn.commit()

    # Check if empty; if so, populate with 5 initial demo records
    cursor.execute('SELECT COUNT(*) FROM predictions')
    count = cursor.fetchone()[0]
    if count == 0:
        seed_initial_history(conn)
    conn.close()

def seed_initial_history(conn):
    cursor = conn.cursor()
    samples = [
        {
            'name': 'Aarav Sharma',
            'created_at': '2026-10-02 14:32',
            'study_hours': 5.5,
            'attendance': 92.0,
            'previous_score': 84.0,
            'assignment_completion': 95.0,
            'sleep_hours': 7.5,
            'class_participation': 9,
            'backlogs': 0,
            'internet': 'Yes',
            'device': 'Yes',
            'extra': 'Yes',
            'parental': 'High',
            'score': 83.4,
            'category': 'Excellent',
            'confidence': 91.2,
            'recs': [
                {'type': 'success', 'title': 'Advance Your Expertise', 'message': 'Outstanding performance! Consider taking up competitive coding or undergraduate research.'}
            ]
        },
        {
            'name': 'Priya Patel',
            'created_at': '2026-10-02 16:15',
            'study_hours': 4.2,
            'attendance': 85.0,
            'previous_score': 74.0,
            'assignment_completion': 88.0,
            'sleep_hours': 7.0,
            'class_participation': 7,
            'backlogs': 0,
            'internet': 'Yes',
            'device': 'Yes',
            'extra': 'No',
            'parental': 'Medium',
            'score': 72.8,
            'category': 'Good',
            'confidence': 88.5,
            'recs': [
                {'type': 'success', 'title': 'Balanced Academic Profile', 'message': 'Solid baseline metrics. Keep reviewing weekly lecture notes before finals.'}
            ]
        },
        {
            'name': 'Rahul Verma',
            'created_at': '2026-10-03 09:45',
            'study_hours': 2.8,
            'attendance': 68.0,
            'previous_score': 58.0,
            'assignment_completion': 65.0,
            'sleep_hours': 6.0,
            'class_participation': 5,
            'backlogs': 1,
            'internet': 'Yes',
            'device': 'Yes',
            'extra': 'No',
            'parental': 'Medium',
            'score': 56.4,
            'category': 'Average',
            'confidence': 84.0,
            'recs': [
                {'type': 'warning', 'title': 'Improve Attendance', 'message': 'Current attendance is below the recommended 75% threshold.'},
                {'type': 'warning', 'title': 'Clear Active Backlogs', 'message': 'Allocate targeted revision blocks for your pending backlog.'}
            ]
        },
        {
            'name': 'Rohan Nair',
            'created_at': '2026-10-03 11:20',
            'study_hours': 1.5,
            'attendance': 54.0,
            'previous_score': 42.0,
            'assignment_completion': 48.0,
            'sleep_hours': 5.0,
            'class_participation': 3,
            'backlogs': 3,
            'internet': 'No',
            'device': 'No',
            'extra': 'No',
            'parental': 'Low',
            'score': 39.8,
            'category': 'At Risk',
            'confidence': 82.5,
            'recs': [
                {'type': 'critical', 'title': 'Critical Attendance Deficit', 'message': 'Prioritize reaching at least 75% to prevent exam debarment.'},
                {'type': 'critical', 'title': 'Clear Active Backlogs', 'message': 'You have 3 pending backlogs. Immediate remedial coaching recommended.'}
            ]
        },
        {
            'name': 'Ananya Das',
            'created_at': '2026-10-03 13:10',
            'study_hours': 4.8,
            'attendance': 89.0,
            'previous_score': 79.0,
            'assignment_completion': 91.0,
            'sleep_hours': 7.2,
            'class_participation': 8,
            'backlogs': 0,
            'internet': 'Yes',
            'device': 'Yes',
            'extra': 'Yes',
            'parental': 'High',
            'score': 76.5,
            'category': 'Good',
            'confidence': 89.4,
            'recs': [
                {'type': 'success', 'title': 'Balanced Academic Profile', 'message': 'Strong foundational habits observed. Maintain your study rhythm.'}
            ]
        }
    ]

    for s in samples:
        cursor.execute('''
            INSERT INTO predictions (
                student_name, created_at, study_hours, attendance, previous_score,
                assignment_completion, sleep_hours, class_participation, backlogs,
                internet_availability, device_availability, extracurricular_activity,
                parental_support, predicted_score, category, confidence, recommendations_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            s['name'], s['created_at'], s['study_hours'], s['attendance'], s['previous_score'],
            s['assignment_completion'], s['sleep_hours'], s['class_participation'], s['backlogs'],
            s['internet'], s['device'], s['extra'], s['parental'],
            s['score'], s['category'], s['confidence'], json.dumps(s['recs'])
        ))
    conn.commit()

def save_prediction(student_name, features, prediction_result):
    """Save a new prediction record to SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    name = (student_name or 'Anonymous Student').strip()

    cursor.execute('''
        INSERT INTO predictions (
            student_name, created_at, study_hours, attendance, previous_score,
            assignment_completion, sleep_hours, class_participation, backlogs,
            internet_availability, device_availability, extracurricular_activity,
            parental_support, predicted_score, category, confidence, recommendations_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        name,
        now_str,
        float(features.get('study_hours_per_day', 0)),
        float(features.get('attendance_percentage', 0)),
        float(features.get('previous_score', 0)),
        float(features.get('assignment_completion', 0)),
        float(features.get('sleep_hours', 0)),
        float(features.get('class_participation', 0)),
        int(features.get('backlogs', 0)),
        str(features.get('internet_availability', 'Yes')),
        str(features.get('device_availability', 'Yes')),
        str(features.get('extracurricular_activity', 'No')),
        str(features.get('parental_support', 'Medium')),
        float(prediction_result.get('predicted_score', 0)),
        str(prediction_result.get('category', 'Average')),
        float(prediction_result.get('confidence', 85.0)),
        json.dumps(prediction_result.get('recommendations', []))
    ))
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id

def get_all_predictions(limit=100):
    """Retrieve all predictions sorted by created_at desc."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    results = []
    for r in rows:
        item = dict(r)
        try:
            item['recommendations'] = json.loads(item['recommendations_json'])
        except Exception:
            item['recommendations'] = []
        results.append(item)
    conn.close()
    return results

def delete_prediction_by_id(pred_id):
    """Delete a specific prediction by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions WHERE id = ?', (pred_id,))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0

def clear_all_predictions():
    """Clear all records from prediction history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()
    return True

def get_dashboard_kpis():
    """Compute live dashboard KPIs from prediction history and dataset."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*), AVG(predicted_score) FROM predictions')
    row = cursor.fetchone()
    total_preds = row[0] or 0
    avg_score = round(row[1], 1) if row[1] else 0.0

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE category = 'At Risk'")
    at_risk_count = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE category = 'Excellent'")
    excellent_count = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE category = 'Good'")
    good_count = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE category = 'Average'")
    average_count = cursor.fetchone()[0] or 0

    conn.close()
    return {
        'total_predictions': total_preds,
        'average_score': avg_score,
        'at_risk_count': at_risk_count,
        'excellent_count': excellent_count,
        'good_count': good_count,
        'average_count': average_count
    }
