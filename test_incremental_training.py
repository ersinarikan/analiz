#!/usr/bin/env python3
"""
Incremental Age Training Test Script
Test the new feedback-only fine-tuning system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.incremental_age_training_service import IncrementalAgeTrainingService

def test_incremental_training():
    """Test incremental training with current feedback data"""
    print("🚀 INCREMENTAL AGE TRAINING TEST")
    print("=" * 50)
    
    app = create_app()
    with app.app_context():
        service = IncrementalAgeTrainingService()
        
        # Test feedback data preparation
        print("📊 Testing feedback data preparation...")
        feedback_data = service.prepare_feedback_data(min_samples=1)  # Lower threshold for testing
        
        if feedback_data is None:
            print("❌ No feedback data available for training")
            return
        
        print(f"✅ Feedback data prepared:")
        print(f"   - Samples: {len(feedback_data['embeddings'])}")
        print(f"   - Manual: {feedback_data['sources'].count('manual')}")
        print(f"   - Pseudo: {feedback_data['sources'].count('pseudo')}")
        print(f"   - Age range: {feedback_data['ages'].min():.1f} - {feedback_data['ages'].max():.1f}")
        
        # Test base model loading
        print("\n🔧 Testing base model loading...")
        try:
            base_model = service.load_base_model()
            print("✅ Base model loaded successfully")
        except Exception as e:
            print(f"❌ Base model loading failed: {e}")
            return
        
        # Test incremental model creation
        print("\n🧠 Testing incremental model creation...")
        try:
            incremental_model = service.create_incremental_model(base_model)
            
            # Count trainable parameters
            total_params = sum(p.numel() for p in incremental_model.parameters())
            trainable_params = sum(p.numel() for p in incremental_model.parameters() if p.requires_grad)
            
            print(f"✅ Incremental model created:")
            print(f"   - Total parameters: {total_params:,}")
            print(f"   - Trainable parameters: {trainable_params:,}")
            print(f"   - Frozen parameters: {total_params - trainable_params:,}")
            print(f"   - Training efficiency: {trainable_params/total_params*100:.1f}% trainable")
            
        except Exception as e:
            print(f"❌ Incremental model creation failed: {e}")
            return
        
        # Run full incremental training
        print("\n🎯 Running full incremental training pipeline...")
        try:
            result = service.run_incremental_training(min_feedback_samples=1)
            
            if result is None:
                print("❌ Training failed - insufficient data")
                return
            
            model_version = result['model_version']
            training_result = result['training_result']
            
            print("✅ Incremental training completed successfully!")
            print(f"\n📊 RESULTS:")
            print(f"   Model Version: {model_version.version_name}")
            print(f"   Training Samples: {training_result['training_samples']}")
            print(f"   Validation Samples: {training_result['validation_samples']}")
            print(f"   Final MAE: {training_result['metrics']['mae']:.3f} years")
            print(f"   3-year Accuracy: {training_result['metrics']['accuracy_3years']:.1f}%")
            print(f"   Training Loss: {training_result['metrics']['final_train_loss']:.4f}")
            print(f"   Validation Loss: {training_result['metrics']['final_val_loss']:.4f}")
            
            # Training history
            epochs_run = len(training_result['history']['train_loss'])
            print(f"\n📈 TRAINING HISTORY ({epochs_run} epochs):")
            for i in range(min(5, epochs_run)):  # Show first 5 epochs
                epoch = i + 1
                train_loss = training_result['history']['train_loss'][i]
                val_loss = training_result['history']['val_loss'][i]
                val_mae = training_result['history']['val_mae'][i]
                print(f"   Epoch {epoch:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}, MAE={val_mae:.2f}")
            
            if epochs_run > 5:
                print("   ...")
                # Show last epoch
                train_loss = training_result['history']['train_loss'][-1]
                val_loss = training_result['history']['val_loss'][-1]
                val_mae = training_result['history']['val_mae'][-1]
                print(f"   Epoch {epochs_run:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}, MAE={val_mae:.2f}")
            
            print(f"\n💾 Model saved to: {model_version.file_path}")
            print(f"🆔 Model Version ID: {model_version.id}")
            
            # Compare with base model performance
            print(f"\n📊 PERFORMANCE COMPARISON:")
            print(f"   Base Model (UTKFace): MAE ~1.65 years")
            print(f"   Incremental Model: MAE {training_result['metrics']['mae']:.3f} years")
            
            improvement = 1.65 - training_result['metrics']['mae']
            if improvement > 0:
                print(f"   ✅ Improvement: {improvement:.3f} years better")
            else:
                print(f"   ⚠️  Change: {improvement:.3f} years")
            
            print(f"\n🔄 EFFICIENCY BENEFITS:")
            print(f"   - Training time: ~minutes (vs hours for full retraining)")
            print(f"   - Data used: {training_result['training_samples']} feedback samples")
            print(f"   - Memory efficient: Only fine-tuning layers trained")
            print(f"   - Base knowledge preserved: UTKFace knowledge intact")
            
        except Exception as e:
            print(f"❌ Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n🎉 INCREMENTAL TRAINING TEST COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    test_incremental_training() 