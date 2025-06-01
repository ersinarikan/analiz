#!/usr/bin/env python3
"""
Reset Feedback Training Status
Reset feedback training status for testing incremental learning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models.feedback import Feedback

def reset_feedback_status():
    """Reset training status of feedback data"""
    print("🔄 RESETTING FEEDBACK TRAINING STATUS")
    print("=" * 40)
    
    app = create_app()
    with app.app_context():
        # Find all used feedback data
        used_feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | 
            (Feedback.feedback_type == 'age_pseudo')
        ).filter(
            Feedback.training_status == 'used_in_training'
        ).all()
        
        print(f"📊 Found {len(used_feedbacks)} used feedback records")
        
        if len(used_feedbacks) == 0:
            print("ℹ️  No used feedback records found")
            
            # Check all feedback records
            all_feedbacks = Feedback.query.filter(
                (Feedback.feedback_type == 'age') | 
                (Feedback.feedback_type == 'age_pseudo')
            ).all()
            
            print(f"📋 Total feedback records: {len(all_feedbacks)}")
            
            for feedback in all_feedbacks[:5]:  # Show first 5
                print(f"   ID: {feedback.id}, Type: {feedback.feedback_type}, Status: {feedback.training_status}")
            
            return
        
        # Reset status
        reset_count = 0
        for feedback in used_feedbacks:
            feedback.training_status = None
            reset_count += 1
        
        # Commit changes
        db.session.commit()
        
        print(f"✅ Reset {reset_count} feedback records")
        print("📊 Training status set to None (available for training)")
        
        # Verify
        available_feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | 
            (Feedback.feedback_type == 'age_pseudo')
        ).filter(
            db.or_(
                Feedback.training_status.is_(None),
                Feedback.training_status != 'used_in_training'
            )
        ).all()
        
        print(f"✅ Available feedback records after reset: {len(available_feedbacks)}")
        
        # Show details
        manual_count = sum(1 for f in available_feedbacks if f.feedback_source == 'MANUAL_USER')
        pseudo_count = sum(1 for f in available_feedbacks if f.feedback_source != 'MANUAL_USER')
        
        print(f"   - Manual feedback: {manual_count}")
        print(f"   - Pseudo feedback: {pseudo_count}")
        
        print("\n🎉 FEEDBACK STATUS RESET COMPLETED!")

if __name__ == "__main__":
    reset_feedback_status() 