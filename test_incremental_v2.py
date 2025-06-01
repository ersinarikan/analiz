#!/usr/bin/env python3
"""
Test Improved Incremental Age Training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.incremental_age_training_service_v2 import IncrementalAgeTrainingServiceV2

def test_incremental_v2():
    """Test improved incremental training"""
    print("🚀 IMPROVED INCREMENTAL AGE TRAINING TEST")
    print("=" * 55)
    
    app = create_app()
    with app.app_context():
        service = IncrementalAgeTrainingServiceV2()
        
        # Run training
        result = service.run_incremental_training(min_feedback_samples=1)
        
        if result is None:
            print("❌ Training failed - insufficient data")
            return
        
        training_result = result['training_result']
        
        print("✅ IMPROVED INCREMENTAL TRAINING COMPLETED!")
        print(f"\n📊 RESULTS:")
        print(f"   Version: {result['version_name']}")
        print(f"   Training Samples: {training_result['training_samples']}")
        print(f"   Validation Samples: {training_result['validation_samples']}")
        print(f"   Final MAE: {training_result['metrics']['mae']:.3f} years")
        print(f"   3-year Accuracy: {training_result['metrics']['accuracy_3years']:.1f}%")
        
        # Compare with base model
        base_mae = 1.696  # From feedback enhanced model
        improvement = base_mae - training_result['metrics']['mae']
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"   Base Model (Feedback Enhanced): {base_mae:.3f} years")
        print(f"   Incremental Model V2: {training_result['metrics']['mae']:.3f} years")
        
        if improvement > 0:
            print(f"   ✅ Improvement: {improvement:.3f} years better")
        else:
            print(f"   ⚠️  Change: {improvement:.3f} years")
        
        # Show training progress (last few epochs)
        epochs = len(training_result['history']['train_loss'])
        print(f"\n📈 TRAINING PROGRESS (last 5 of {epochs} epochs):")
        
        start_idx = max(0, epochs - 5)
        for i in range(start_idx, epochs):
            epoch = i + 1
            train_loss = training_result['history']['train_loss'][i]
            val_loss = training_result['history']['val_loss'][i]
            val_mae = training_result['history']['val_mae'][i]
            print(f"   Epoch {epoch:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}, MAE={val_mae:.2f}")
        
        print(f"\n🎯 KEY IMPROVEMENTS IN V2:")
        print(f"   - Better architecture: Parallel fine-tuning branch")
        print(f"   - Sigmoid mixing weight for stable learning")
        print(f"   - Higher learning rate for faster adaptation")
        print(f"   - Frozen base model preserves knowledge")
        print(f"   - Embedding-level fine-tuning for better flexibility")
        
        print(f"\n🎉 V2 TEST COMPLETED!")

if __name__ == "__main__":
    test_incremental_v2() 