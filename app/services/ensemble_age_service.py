import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from flask import current_app
from app import db
from app.models.feedback import Feedback

logger = logging.getLogger('app.ensemble_age_service')

class EnsembleAgeService:
    """
    Yaş tahmini için ensemble tabanlı artımsal öğrenme servis sınıfı.
    - Temel modeli korur, geri bildirim düzeltmelerini uygular, yaş benzerliği ile eşleştirme yapar.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu")
        self.feedback_corrections = {}  # person_id -> age_correction
        self.embedding_corrections = {}  # embedding_hash -> age_correction
        logger.info(f"EnsembleAgeService initialized")
    
    def load_feedback_corrections(self):
        """Load all feedback corrections as lookup table"""
        logger.info("Loading feedback corrections...")
        
        # Get all available feedback
        feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | 
            (Feedback.feedback_type == 'age_pseudo')
        ).filter(
            Feedback.embedding.isnot(None)
        ).all()
        
        logger.info(f"Found {len(feedbacks)} feedback records")
        
        # Process corrections
        person_corrections = {}
        embedding_corrections = {}
        
        for feedback in feedbacks:
            try:
                # Parse embedding
                if isinstance(feedback.embedding, str):
                    embedding = np.array([float(x) for x in feedback.embedding.split(',')])
                else:
                    continue
                
                # Get corrected age
                if feedback.feedback_source == 'MANUAL_USER':
                    corrected_age = feedback.corrected_age
                    confidence = 1.0
                else:
                    corrected_age = feedback.pseudo_label_original_age
                    confidence = feedback.pseudo_label_clip_confidence or 0.8
                
                if corrected_age is None:
                    continue
                
                # Store by person_id
                person_id = feedback.person_id
                if person_id:
                    if person_id not in person_corrections:
                        person_corrections[person_id] = {
                            'corrected_age': corrected_age,
                            'confidence': confidence,
                            'source': feedback.feedback_source,
                            'embedding': embedding
                        }
                    elif feedback.feedback_source == 'MANUAL_USER':
                        # Manual feedback overrides pseudo
                        person_corrections[person_id] = {
                            'corrected_age': corrected_age,
                            'confidence': confidence,
                            'source': feedback.feedback_source,
                            'embedding': embedding
                        }
                
                # Store by embedding hash (for similarity matching)
                embedding_hash = self._hash_embedding(embedding)
                embedding_corrections[embedding_hash] = {
                    'corrected_age': corrected_age,
                    'confidence': confidence,
                    'embedding': embedding,
                    'person_id': person_id
                }
                
            except Exception as e:
                logger.error(f"Error processing feedback {feedback.id}: {str(e)}")
                continue
        
        self.feedback_corrections = person_corrections
        self.embedding_corrections = embedding_corrections
        
        manual_count = sum(1 for c in person_corrections.values() if c['source'] == 'MANUAL_USER')
        pseudo_count = len(person_corrections) - manual_count
        
        logger.info(f"✅ Loaded corrections: {len(person_corrections)} people ({manual_count} manual, {pseudo_count} pseudo)")
        logger.info(f"✅ Embedding corrections: {len(embedding_corrections)} entries")
        
        return len(person_corrections)
    
    def _hash_embedding(self, embedding: np.ndarray) -> int:
        """Create hash for embedding (for lookup)"""
        # Simple hash based on first few dimensions
        return hash(tuple(embedding[:10].round(3)))
    
    def predict_age_ensemble(self, base_age: int, base_confidence: float, person_id: int = None, embedding: np.ndarray = None) -> tuple[int, float, dict]:
        """
        Ensemble prediction: Base model + Feedback corrections
        
        Args:
            base_age_prediction: Base model yaş tahmini
            embedding: Face embedding
            person_id: Person ID (if known)
            
        Returns:
            tuple: (final_age, confidence, correction_info)
        """
        
        # 1. Direct person lookup
        if person_id and person_id in self.feedback_corrections:
            correction = self.feedback_corrections[person_id]
            logger.info(f"Direct person match found: {person_id} -> {correction['corrected_age']}")
            return correction['corrected_age'], correction['confidence'], {
                'method': 'direct_person_match',
                'person_id': person_id,
                'source': correction['source']
            }
        
        # 2. Embedding similarity search
        if embedding is not None:
            embedding_hash = self._hash_embedding(embedding)
            
            # Exact embedding match
            if embedding_hash in self.embedding_corrections:
                correction = self.embedding_corrections[embedding_hash]
                logger.info(f"Exact embedding match found -> {correction['corrected_age']}")
                return correction['corrected_age'], correction['confidence'], {
                    'method': 'exact_embedding_match',
                    'person_id': correction['person_id']
                }
            
            # Similarity-based correction
            best_similarity = -1
            best_correction = None
            
            # Normalize input embedding
            embedding_norm = embedding / np.linalg.norm(embedding)
            
            for emb_hash, correction in self.embedding_corrections.items():
                stored_embedding = correction['embedding']
                stored_embedding_norm = stored_embedding / np.linalg.norm(stored_embedding)
                
                # Cosine similarity
                similarity = np.dot(embedding_norm, stored_embedding_norm)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_correction = correction
            
            # If we found a good similarity match (>0.9), use it
            if best_similarity > 0.95:  # Very high similarity threshold
                # Weighted combination: high similarity = more correction influence
                weight = best_similarity  # 0.95-1.0 range
                corrected_age = best_correction['corrected_age']
                
                # Blend base prediction with correction
                final_age = (1 - weight) * base_age + weight * corrected_age
                confidence = best_correction['confidence'] * weight
                
                logger.info(f"Similarity correction: sim={best_similarity:.3f}, "
                           f"base={base_age:.1f} -> final={final_age:.1f}")
                
                return final_age, confidence, {
                    'method': 'similarity_correction',
                    'similarity': best_similarity,
                    'base_age': base_age,
                    'corrected_age': corrected_age,
                    'person_id': best_correction['person_id']
                }
        
        # 3. No correction found - use base model
        logger.debug(f"No correction found, using base model: {base_age}")
        return base_age, 0.5, {
            'method': 'base_model_only',
            'base_age': base_age
        }
    
    def get_statistics(self):
        """Get ensemble statistics"""
        stats = {
            'total_people_corrections': len(self.feedback_corrections),
            'total_embedding_corrections': len(self.embedding_corrections),
            'manual_corrections': sum(1 for c in self.feedback_corrections.values() if c['source'] == 'MANUAL_USER'),
            'pseudo_corrections': sum(1 for c in self.feedback_corrections.values() if c['source'] != 'MANUAL_USER'),
        }
        
        if self.feedback_corrections:
            ages = [c['corrected_age'] for c in self.feedback_corrections.values()]
            stats.update({
                'age_range': f"{min(ages):.1f} - {max(ages):.1f}",
                'age_mean': f"{np.mean(ages):.1f}"
            })
        
        return stats
    
    def test_ensemble_predictions(self, test_cases=None):
        """Test ensemble on known feedback cases"""
        logger.info("Testing ensemble predictions...")
        
        if not self.feedback_corrections:
            logger.warning("No corrections loaded!")
            return
        
        # Test a few cases
        test_results = []
        
        for person_id, correction in list(self.feedback_corrections.items())[:5]:
            embedding = correction['embedding']
            corrected_age = correction['corrected_age']
            
            # Simulate base model prediction (add some noise)
            simulated_base_pred = corrected_age + np.random.normal(0, 2)  # +/- 2 years noise
            
            # Test ensemble
            ensemble_pred, confidence, info = self.predict_age_ensemble(
                simulated_base_pred, embedding, person_id
            )
            
            test_results.append({
                'person_id': person_id,
                'true_age': corrected_age,
                'base_pred': simulated_base_pred,
                'ensemble_pred': ensemble_pred,
                'confidence': confidence,
                'method': info['method']
            })
            
            logger.info(f"Test {person_id}: True={corrected_age:.1f}, "
                       f"Base={simulated_base_pred:.1f}, "
                       f"Ensemble={ensemble_pred:.1f}, "
                       f"Method={info['method']}")
        
        return test_results 