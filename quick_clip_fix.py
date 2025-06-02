#!/usr/bin/env python3
"""
CLIP category_feedback field kontrol scripti
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models.feedback import Feedback
import json

def main():
    app = create_app()
    
    with app.app_context():
        print("🔍 category_feedback Field Kontrol")
        print("=" * 40)
        
        content_feedbacks = Feedback.query.filter(
            Feedback.feedback_source == 'MANUAL_USER_CONTENT_CORRECTION'
        ).all()
        
        print(f"Toplam MANUAL_USER_CONTENT_CORRECTION: {len(content_feedbacks)}")
        
        for i, feedback in enumerate(content_feedbacks[:5]):
            print(f"\n--- Feedback {i+1} (ID: {feedback.id}) ---")
            print(f"category_feedback type: {type(feedback.category_feedback)}")
            print(f"category_feedback value: {feedback.category_feedback}")
            
            # Type kontrolü
            if isinstance(feedback.category_feedback, str):
                print("⚠️ STRING BULUNDU! JSON parse denenecek...")
                try:
                    parsed = json.loads(feedback.category_feedback)
                    print(f"✅ JSON parse başarılı: {parsed}")
                except Exception as e:
                    print(f"❌ JSON parse hatası: {e}")
            elif isinstance(feedback.category_feedback, dict):
                print("✅ DICT tipinde - OK")
                for key, value in feedback.category_feedback.items():
                    print(f"  {key}: {value} (type: {type(value)})")
            else:
                print(f"❓ Bilinmeyen tip: {type(feedback.category_feedback)}")

if __name__ == "__main__":
    main() 