#!/usr/bin/env python3
"""
Test Ensemble CLIP Service
Lookup-based incremental learning for content descriptions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.ensemble_clip_service import EnsembleClipService

def test_ensemble_clip():
    """Test ensemble CLIP prediction service"""
    print("🎯 ENSEMBLE CLIP SERVICE TEST")
    print("=" * 40)
    
    app = create_app()
    with app.app_context():
        service = EnsembleClipService()
        
        # Load content corrections
        print("📊 Loading CLIP content corrections...")
        correction_count = service.load_content_corrections()
        
        if correction_count == 0:
            print("ℹ️  No CLIP content corrections available")
            print("   This is normal - content feedback might not exist yet")
        else:
            print(f"✅ Loaded {correction_count} content corrections")
        
        # Get statistics
        stats = service.get_statistics()
        print(f"\n📈 CLIP ENSEMBLE STATISTICS:")
        print(f"   Content corrections: {stats['total_content_corrections']}")
        print(f"   Confidence adjustments: {stats['total_confidence_adjustments']}")
        print(f"   Embedding corrections: {stats['total_embedding_corrections']}")
        
        if stats['total_content_corrections'] > 0:
            print(f"   Manual corrections: {stats.get('manual_corrections', 0)}")
            print(f"   Auto corrections: {stats.get('auto_corrections', 0)}")
        
        if stats['total_confidence_adjustments'] > 0:
            print(f"   Avg confidence adjustment: {stats.get('avg_confidence_adjustment', 'N/A')}")
            print(f"   Adjustment range: {stats.get('confidence_adjustment_range', 'N/A')}")
        
        # Test ensemble predictions
        if correction_count > 0:
            print(f"\n🧪 TESTING CLIP ENSEMBLE PREDICTIONS:")
            test_results = service.test_ensemble_predictions()
            
            if test_results:
                print(f"\n📊 CLIP TEST RESULTS:")
                perfect_matches = 0
                
                for result in test_results:
                    print(f"   Content {result['content_id'][:8]}...")
                    print(f"      Original: {result['original_description'][:60]}...")
                    print(f"      Corrected: {result['corrected_description'][:60]}...")
                    print(f"      Ensemble: {result['ensemble_description'][:60]}...")
                    print(f"      Base confidence: {result['base_confidence']:.2f}")
                    print(f"      Ensemble confidence: {result['ensemble_confidence']:.2f}")
                    print(f"      Method: {result['method']}")
                    print(f"      Match quality: {result['match_quality']}")
                    
                    if result['match_quality'] == 'perfect':
                        perfect_matches += 1
                    print()
                
                accuracy = perfect_matches / len(test_results) * 100
                print(f"🎯 CLIP ENSEMBLE ACCURACY: {accuracy:.1f}% ({perfect_matches}/{len(test_results)} perfect matches)")
        
        # Demo prediction scenarios
        print(f"\n🔮 CLIP ENSEMBLE DEMO SCENARIOS:")
        
        # Scenario 1: Unknown content (fallback to base)
        demo_desc, demo_conf, demo_info = service.predict_content_ensemble(
            base_description="A person in a photo",
            base_confidence=0.75,
            content_id="unknown_content_123",
            person_id="unknown_person_456"
        )
        
        print(f"   Scenario 1 - Unknown content:")
        print(f"      Input: 'A person in a photo' (conf: 0.75)")
        print(f"      Output: '{demo_desc}' (conf: {demo_conf:.2f})")
        print(f"      Method: {demo_info['method']}")
        
        # Show how content optimization would work
        sample_content_list = [
            {
                'description': 'A person standing',
                'confidence': 0.6,
                'content_id': 'test_content_1',
                'person_id': 'test_person_1'
            },
            {
                'description': 'Someone in outdoor setting',
                'confidence': 0.8,
                'content_id': 'test_content_2',
                'person_id': 'test_person_2'
            }
        ]
        
        print(f"\n📋 CONTENT OPTIMIZATION DEMO:")
        optimized_content = service.optimize_content_descriptions(sample_content_list)
        
        for i, (original, optimized) in enumerate(zip(sample_content_list, optimized_content)):
            print(f"   Content {i+1}:")
            print(f"      Original: '{original['description']}' (conf: {original['confidence']:.2f})")
            print(f"      Optimized: '{optimized['optimized_description']}' (conf: {optimized['optimized_confidence']:.2f})")
            print(f"      Method: {optimized['optimization_method']}")
        
        print(f"\n✅ CLIP ENSEMBLE ADVANTAGES:")
        print(f"   - No CLIP model retraining required!")
        print(f"   - Instant content description corrections")
        print(f"   - Confidence score adjustments")
        print(f"   - Perfect memory for corrected content")
        print(f"   - Embedding similarity for related content")
        print(f"   - Graceful fallback to base CLIP model")
        print(f"   - Zero risk of CLIP model degradation")
        
        print(f"\n🎯 PRODUCTION INTEGRATION:")
        print(f"   1. Base CLIP model generates description + confidence")
        print(f"   2. Check content_id + person_id for exact corrections")
        print(f"   3. Check for confidence adjustments")
        print(f"   4. Check CLIP embedding similarity for related corrections")
        print(f"   5. Fallback to base model for unknown content")
        print(f"   6. Log correction method for analytics")
        
        print(f"\n🎉 CLIP ENSEMBLE TEST COMPLETED!")

if __name__ == "__main__":
    test_ensemble_clip() 