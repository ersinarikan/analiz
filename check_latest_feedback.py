#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from app import create_app
from app.models.feedback import Feedback

def check_latest_feedback():
    app = create_app()
    with app.app_context():
        # En son content feedback'i al
        latest = Feedback.query.filter_by(
            feedback_source='MANUAL_USER_CONTENT_CORRECTION'
        ).order_by(Feedback.created_at.desc()).first()
        
        if latest:
            print(f"En son feedback frame_path: {latest.frame_path}")
            print(f"Feedback ID: {latest.id}")
            print(f"Analysis ID: {latest.analysis_id}")
            print(f"Created at: {latest.created_at}")
        else:
            print("Hiç content feedback bulunamadı!")

if __name__ == "__main__":
    check_latest_feedback() 