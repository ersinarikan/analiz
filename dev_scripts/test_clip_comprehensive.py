#!/usr/bin/env python3
"""
Comprehensive CLIP Fine-tuning Test Suite
"""

import sys
import os
import requests
import json
import time
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.services.clip_training_service import CLIPTrainingService
from app.services.clip_version_service import CLIPVersionService
from app.models.clip_training import CLIPTrainingSession
from app.models.feedback import Feedback

class CLIPTestSuite:
    """Comprehensive CLIP Testing Suite"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.app = create_app('development')
        self.results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'failures': []
        }
    
    def log(self, message, level="INFO"):
        """Test log mesajı"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def assert_test(self, condition, test_name, error_msg=""):
        """Test assertion"""
        self.results['tests_run'] += 1
        if condition:
            self.results['tests_passed'] += 1
            self.log(f"✅ {test_name}", "PASS")
            return True
        else:
            self.results['tests_failed'] += 1
            self.results['failures'].append(f"{test_name}: {error_msg}")
            self.log(f"❌ {test_name}: {error_msg}", "FAIL")
            return False
    
    def test_api_endpoints(self):
        """API endpoint'lerini test et"""
        self.log("🧪 API Endpoints Test Başlıyor...")
        
        # 1. Statistics endpoint
        try:
            response = requests.get(f"{self.base_url}/api/clip-training/statistics")
            self.assert_test(
                response.status_code == 200,
                "Statistics API",
                f"Status: {response.status_code}"
            )
            
            if response.status_code == 200:
                data = response.json()
                self.assert_test(
                    'data' in data and 'total_feedbacks' in data['data'],
                    "Statistics Data Structure",
                    "Missing required fields"
                )
        except Exception as e:
            self.assert_test(False, "Statistics API", str(e))
        
        # 2. Versions endpoint
        try:
            response = requests.get(f"{self.base_url}/api/clip-training/versions")
            self.assert_test(
                response.status_code == 200,
                "Versions API",
                f"Status: {response.status_code}"
            )
        except Exception as e:
            self.assert_test(False, "Versions API", str(e))
        
        # 3. Sessions endpoint
        try:
            response = requests.get(f"{self.base_url}/api/clip-training/sessions")
            self.assert_test(
                response.status_code == 200,
                "Sessions API",
                f"Status: {response.status_code}"
            )
        except Exception as e:
            self.assert_test(False, "Sessions API", str(e))
    
    def test_services(self):
        """Service layer'ı test et"""
        self.log("🧪 Service Layer Test Başlıyor...")
        
        with self.app.app_context():
            # 1. CLIPTrainingService
            try:
                service = CLIPTrainingService()
                self.assert_test(True, "CLIPTrainingService Creation", "")
                
                # Statistics test
                stats, error = service.get_training_statistics()
                self.assert_test(
                    error is None and stats is not None,
                    "Training Statistics Service",
                    error or "Stats is None"
                )
                
                # Feedbacks test
                feedbacks, error = service.get_available_feedbacks(min_feedback_count=1)
                self.assert_test(
                    error is None and feedbacks is not None,
                    "Available Feedbacks Service",
                    error or "Feedbacks is None"
                )
                
                if feedbacks:
                    # Contrastive pairs test
                    pairs, error = service.create_contrastive_pairs(feedbacks[:5])  # Test with 5 feedbacks
                    self.assert_test(
                        error is None and isinstance(pairs, list),
                        "Contrastive Pairs Creation",
                        error or "Pairs not a list"
                    )
                
            except Exception as e:
                self.assert_test(False, "CLIPTrainingService", str(e))
            
            # 2. CLIPVersionService
            try:
                version_service = CLIPVersionService()
                self.assert_test(True, "CLIPVersionService Creation", "")
                
                # Get versions
                versions, error = version_service.get_all_versions()
                self.assert_test(
                    error is None and isinstance(versions, list),
                    "Get All Versions Service",
                    error or "Versions not a list"
                )
                
            except Exception as e:
                self.assert_test(False, "CLIPVersionService", str(e))
    
    def test_database_models(self):
        """Database model'lerini test et"""
        self.log("🧪 Database Models Test Başlıyor...")
        
        with self.app.app_context():
            # 1. CLIPTrainingSession model
            try:
                # Create test session
                session = CLIPTrainingSession(
                    version_name="test_v1",
                    feedback_count=10,
                    status='testing'
                )
                session.set_training_params({
                    'learning_rate': 0.001,
                    'batch_size': 8,
                    'epochs': 3
                })
                
                db.session.add(session)
                db.session.commit()
                
                self.assert_test(True, "CLIPTrainingSession Creation", "")
                
                # Test to_dict
                session_dict = session.to_dict()
                self.assert_test(
                    'id' in session_dict and 'version_name' in session_dict,
                    "CLIPTrainingSession to_dict",
                    "Missing required fields"
                )
                
                # Test training params
                params = session.get_training_params()
                self.assert_test(
                    params.get('learning_rate') == 0.001,
                    "Training Params Storage",
                    f"Expected 0.001, got {params.get('learning_rate')}"
                )
                
                # Clean up
                db.session.delete(session)
                db.session.commit()
                
            except Exception as e:
                self.assert_test(False, "CLIPTrainingSession Model", str(e))
            
            # 2. Feedback model integration
            try:
                feedback_count = Feedback.query.count()
                self.assert_test(
                    feedback_count > 0,
                    "Feedback Data Availability",
                    f"No feedback data found (count: {feedback_count})"
                )
                
                # Test feedback structure
                sample_feedback = Feedback.query.first()
                if sample_feedback:
                    self.assert_test(
                        hasattr(sample_feedback, 'category_feedback'),
                        "Feedback Category Structure",
                        "Missing category_feedback field"
                    )
                
            except Exception as e:
                self.assert_test(False, "Feedback Model Integration", str(e))
    
    def test_training_workflow(self):
        """Training workflow'unu test et"""
        self.log("🧪 Training Workflow Test Başlıyor...")
        
        # 1. Data preparation test
        try:
            training_params = {
                "learning_rate": 0.0001,
                "batch_size": 2,
                "epochs": 1,
                "categories": ["violence"],
                "min_feedback_count": 5
            }
            
            response = requests.post(
                f"{self.base_url}/api/clip-training/prepare-data",
                json=training_params,
                headers={'Content-Type': 'application/json'}
            )
            
            self.assert_test(
                response.status_code == 200,
                "Data Preparation API",
                f"Status: {response.status_code}"
            )
            
            if response.status_code == 200:
                data = response.json()
                self.assert_test(
                    'data' in data and 'total_pairs' in data['data'],
                    "Data Preparation Response",
                    "Missing required fields in response"
                )
        
        except Exception as e:
            self.assert_test(False, "Training Workflow", str(e))
    
    def test_file_structure(self):
        """Dosya yapısını test et"""
        self.log("🧪 File Structure Test Başlıyor...")
        
        # 1. Model directories
        required_dirs = [
            "storage/models/clip",
            "storage/models/clip/ViT-H-14-378-quickgelu_dfn5b",
            "storage/models/clip/versions",
            "storage/processed/logs"
        ]
        
        for dir_path in required_dirs:
            exists = os.path.exists(dir_path)
            self.assert_test(
                exists,
                f"Directory Exists: {dir_path}",
                f"Directory not found: {dir_path}"
            )
        
        # 2. Test image
        test_image_path = "storage/processed/frames_test/test_image.jpg"
        self.assert_test(
            os.path.exists(test_image_path),
            "Test Image Exists",
            f"Test image not found: {test_image_path}"
        )
        
        # 3. Log file
        log_file_path = "storage/processed/logs/app.log"
        self.assert_test(
            os.path.exists(log_file_path),
            "Log File Exists",
            f"Log file not found: {log_file_path}"
        )
    
    def test_performance_metrics(self):
        """Performance metriklerini test et"""
        self.log("🧪 Performance Metrics Test Başlıyor...")
        
        with self.app.app_context():
            # Check if there are any completed training sessions
            completed_sessions = CLIPTrainingSession.query.filter_by(
                status='completed'
            ).all()
            
            if completed_sessions:
                for session in completed_sessions:
                    metrics = session.get_performance_metrics()
                    self.assert_test(
                        isinstance(metrics, dict),
                        f"Performance Metrics Structure (Session {session.id})",
                        "Metrics should be a dictionary"
                    )
            else:
                self.log("⚠️ No completed training sessions found for metrics test")
    
    def run_all_tests(self):
        """Tüm testleri çalıştır"""
        self.log("🚀 CLIP Comprehensive Test Suite Başlıyor...")
        self.log("=" * 60)
        
        start_time = time.time()
        
        # Test suites
        self.test_file_structure()
        self.test_database_models()
        self.test_services()
        self.test_api_endpoints()
        self.test_training_workflow()
        self.test_performance_metrics()
        
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        # Results summary
        self.log("=" * 60)
        self.log("📊 TEST RESULTS SUMMARY")
        self.log("=" * 60)
        self.log(f"Tests Run: {self.results['tests_run']}")
        self.log(f"Tests Passed: {self.results['tests_passed']}")
        self.log(f"Tests Failed: {self.results['tests_failed']}")
        self.log(f"Success Rate: {(self.results['tests_passed']/self.results['tests_run']*100):.1f}%")
        self.log(f"Duration: {duration}s")
        
        if self.results['failures']:
            self.log("\n❌ FAILURES:")
            for failure in self.results['failures']:
                self.log(f"  - {failure}")
        
        if self.results['tests_failed'] == 0:
            self.log("\n🎉 ALL TESTS PASSED! CLIP System is ready for production.")
            return True
        else:
            self.log(f"\n⚠️ {self.results['tests_failed']} tests failed. Please review and fix.")
            return False

def main():
    """Ana test fonksiyonu"""
    test_suite = CLIPTestSuite()
    success = test_suite.run_all_tests()
    
    # Exit code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 