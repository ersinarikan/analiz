#!/usr/bin/env python3
"""
Test Ensemble Age Service
Lookup-based incremental learning approach
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.ensemble_age_service import EnsembleAgeService

def test_ensemble_age():
    """Test ensemble age prediction service"""
    print("🎯 ENSEMBLE AGE SERVICE TEST")
    print("=" * 40)
    
    app = create_app()
    with app.app_context():
        service = EnsembleAgeService()
        
        # Load feedback corrections
        print("📊 Loading feedback corrections...")
        correction_count = service.load_feedback_corrections()
        
        if correction_count == 0:
            print("❌ No feedback corrections available")
            return
        
        # Get statistics
        stats = service.get_statistics()
        print(f"\n📈 ENSEMBLE STATISTICS:")
        print(f"   Total people corrections: {stats['total_people_corrections']}")
        print(f"   Total embedding corrections: {stats['total_embedding_corrections']}")
        print(f"   Manual corrections: {stats['manual_corrections']}")
        print(f"   Pseudo corrections: {stats['pseudo_corrections']}")
        if 'age_range' in stats:
            print(f"   Age range: {stats['age_range']}")
            print(f"   Age mean: {stats['age_mean']}")
        
        # Test ensemble predictions
        print(f"\n🧪 TESTING ENSEMBLE PREDICTIONS:")
        test_results = service.test_ensemble_predictions()
        
        if test_results:
            print(f"\n📊 TEST RESULTS:")
            total_error = 0
            for result in test_results:
                error = abs(result['true_age'] - result['ensemble_pred'])
                total_error += error
                print(f"   Person {result['person_id'][:8]}...")
                print(f"      True age: {result['true_age']:.1f}")
                print(f"      Base pred: {result['base_pred']:.1f}")
                print(f"      Ensemble pred: {result['ensemble_pred']:.1f}")
                print(f"      Error: {error:.1f} years")
                print(f"      Method: {result['method']}")
                print(f"      Confidence: {result['confidence']:.2f}")
                print()
            
            avg_error = total_error / len(test_results)
            print(f"🎯 AVERAGE ERROR: {avg_error:.2f} years")
            
            # Method distribution
            methods = [r['method'] for r in test_results]
            method_counts = {m: methods.count(m) for m in set(methods)}
            print(f"\n📋 METHOD DISTRIBUTION:")
            for method, count in method_counts.items():
                print(f"   {method}: {count}")
        
        # Demo: How it would work in production
        print(f"\n🔮 PRODUCTION DEMO:")
        print(f"   1. Get base model prediction (e.g., 25 years)")
        print(f"   2. Check person_id in feedback corrections")
        print(f"   3. If not found, check embedding similarity")
        print(f"   4. If similar embedding found (>95% similarity):")
        print(f"      - Blend base prediction with correction")
        print(f"   5. Otherwise, use base model prediction")
        
        print(f"\n✅ KEY ADVANTAGES:")
        print(f"   - No retraining required!")
        print(f"   - Instant deployment of new corrections")
        print(f"   - Perfect memory of exact matches")
        print(f"   - Graceful degradation for unknown cases")
        print(f"   - Zero risk of catastrophic forgetting")
        print(f"   - Preserves base model knowledge")
        
        print(f"\n🎉 ENSEMBLE TEST COMPLETED!")

if __name__ == "__main__":
    test_ensemble_age() 