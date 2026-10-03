"""
Vercel Serverless Function Entry Point for EduPredict
Exposes Flask WSGI app instance for Vercel's Python runtime.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Vercel entry handler
app.debug = False
