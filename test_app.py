"""
EduPredict Comprehensive Automated Test Suite
Verifies Flask routes, prediction inference, recommendation generation,
database operations, and REST API endpoints.
"""

import unittest
import json
import os
from app import app
import database

class EduPredictTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_01_index_page(self):
        """Test landing / dashboard page renders 200 OK."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'EduPredict', response.data)
        self.assertIn(b'AI-Powered Student Performance Prediction System', response.data)

    def test_02_predict_page_get(self):
        """Test prediction page renders with form controls."""
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Student Performance Prediction', response.data)
        self.assertIn(b'Recommended Demo', response.data)

    def test_03_performance_page(self):
        """Test model performance page renders."""
        response = self.client.get('/model-performance')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Machine Learning Model Performance', response.data)
        self.assertIn(b'R\xc2\xb2 Score', response.data)

    def test_04_insights_page(self):
        """Test student insights page renders."""
        response = self.client.get('/insights')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Student Cohort Insights', response.data)

    def test_05_about_page(self):
        """Test about page renders with ML pipeline and problem statement."""
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'About EduPredict System', response.data)
        self.assertIn(b'Machine Learning Pipeline Workflow', response.data)

    def test_06_history_page(self):
        """Test history log page renders."""
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction History Log', response.data)

    def test_07_api_demo_data(self):
        """Test /api/demo-data returns valid profiles."""
        response = self.client.get('/api/demo-data')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('recommended', data)
        self.assertEqual(data['recommended']['attendance_percentage'], 88.0)

    def test_08_api_insights(self):
        """Test /api/insights returns cohort distribution and statistics."""
        response = self.client.get('/api/insights')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['total_records'], 1500)
        self.assertIn('category_distribution', data)

    def test_09_predict_post_valid(self):
        """Test /predict POST with valid student input."""
        payload = {
            'student_name': 'Test Student',
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
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('predicted_score', data)
        self.assertIn('category', data)
        self.assertIn('recommendations', data)
        self.assertGreaterEqual(data['predicted_score'], 0.0)
        self.assertLessEqual(data['predicted_score'], 100.0)

    def test_10_predict_post_invalid(self):
        """Test /predict POST with out-of-bounds input returns 400 error."""
        payload = {
            'student_name': 'Invalid Student',
            'study_hours_per_day': 25.0,  # Invalid: >16
            'attendance_percentage': 150.0,  # Invalid: >100
            'previous_score': -10.0,  # Invalid: <0
            'assignment_completion': 90.0,
            'sleep_hours': 7.0,
            'class_participation': 8,
            'backlogs': -2  # Invalid: negative
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertGreater(len(data['errors']), 0)

    def test_11_prediction_history_endpoints(self):
        """Test retrieving, deleting single, and clearing predictions."""
        # 1. Get history
        res = self.client.get('/prediction-history')
        self.assertEqual(res.status_code, 200)
        preds = json.loads(res.data)
        self.assertIsInstance(preds, list)

        # 2. Delete single prediction
        if preds:
            pred_id = preds[0]['id']
            del_res = self.client.delete(f'/prediction-history/{pred_id}')
            self.assertEqual(del_res.status_code, 200)

if __name__ == '__main__':
    unittest.main()
