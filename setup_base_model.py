#!/usr/bin/env python3
"""
Base Model Setup Script
Copy current active model to base_model directory for incremental learning
"""

import os
import shutil
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.age_training_service import AgeTrainingService

def setup_base_model():
    """Copy active model to base_model directory"""
    print("🔧 SETTING UP BASE MODEL FOR INCREMENTAL LEARNING")
    print("=" * 55)
    
    app = create_app()
    with app.app_context():
        service = AgeTrainingService()
        
        # Get active model
        from app.models.content import ModelVersion
        active_model = ModelVersion.query.filter_by(
            model_type='age',
            is_active=True
        ).first()
        
        if not active_model:
            print("❌ No active model found!")
            return False
        
        print(f"📦 Active model: {active_model.version_name}")
        print(f"📅 Created: {active_model.created_at}")
        print(f"📊 Performance: MAE {active_model.metrics.get('mae', 'N/A')}")
        
        # Source path
        source_path = active_model.file_path
        print(f"📁 Source path: {source_path}")
        
        if not os.path.exists(source_path):
            print(f"❌ Source path not found: {source_path}")
            return False
        
        # Destination path
        models_folder = app.config['MODELS_FOLDER']
        base_model_dir = os.path.join(models_folder, 'age', 'custom_age_head', 'base_model')
        
        print(f"🎯 Target path: {base_model_dir}")
        
        # Create base model directory
        os.makedirs(base_model_dir, exist_ok=True)
        
        # Copy all files
        copied_files = []
        for file in os.listdir(source_path):
            src_file = os.path.join(source_path, file)
            dst_file = os.path.join(base_model_dir, file)
            
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
                copied_files.append(file)
                print(f"✅ Copied: {file}")
        
        if copied_files:
            print(f"\n🎉 BASE MODEL SETUP COMPLETED!")
            print(f"   - Files copied: {len(copied_files)}")
            print(f"   - Location: {base_model_dir}")
            print(f"   - Ready for incremental learning!")
            return True
        else:
            print("❌ No files copied")
            return False

if __name__ == "__main__":
    setup_base_model() 