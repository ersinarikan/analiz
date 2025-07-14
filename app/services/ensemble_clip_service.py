import os
import json
import logging
import numpy as np
import torch
from datetime import datetime
from flask import current_app
from app import db
from app.models.feedback import Feedback

logger = logging.getLogger('app.ensemble_clip_service')

class EnsembleClipService:
    """
    CLIP modeli için ensemble tabanlı artımsal öğrenme servis sınıfı.
    - Temel modeli korur, geri bildirim düzeltmelerini uygular, içerik benzerliği ile eşleştirme yapar.
    """
    
    def __init__(self):
        self.content_corrections: dict[int, dict] = {}  # content_hash -> corrected_description
        self.embedding_corrections: dict[int, dict] = {}  # clip_embedding_hash -> correction
        self.confidence_adjustments: dict[int, dict] = {}  # content_hash -> confidence_adjustment
        logger.info(f"EnsembleClipService initialized")
    
    def load_content_corrections(self):
        """Load all content feedback corrections as lookup tables"""
        logger.info("Loading CLIP content corrections...")
        
        # Get all content feedback (using available fields)
        feedbacks = Feedback.query.filter(
            Feedback.feedback_type == 'content'
        ).filter(
            db.or_(
                Feedback.comment.isnot(None),  # Use comment as corrected description
                Feedback.rating.isnot(None)    # Use rating for confidence adjustment
            )
        ).all()
        
        logger.info(f"Found {len(feedbacks)} content feedback records")
        
        content_corrections = {}
        embedding_corrections = {}
        confidence_adjustments = {}
        
        for feedback in feedbacks:
            try:
                # Content hash for exact matching
                content_hash = self._hash_content(feedback)
                
                # Description correction (using comment field)
                if feedback.comment:
                    content_corrections[content_hash] = {
                        'original_description': 'Original content description',  # Placeholder
                        'corrected_description': feedback.comment,
                        'confidence': 1.0 if feedback.feedback_source == 'MANUAL_USER' else 0.8,
                        'source': feedback.feedback_source,
                        'content_id': feedback.content_id,
                        'person_id': feedback.person_id
                    }
                
                # Confidence adjustment (using rating field as adjustment)
                if feedback.rating is not None:
                    # Convert rating (1-5) to confidence adjustment (-0.4 to +0.4)
                    adjustment = (feedback.rating - 3) * 0.2  # Rating 3 = no change, 1 = -0.4, 5 = +0.4
                    confidence_adjustments[content_hash] = {
                        'adjustment': adjustment,
                        'original_confidence': 0.5,  # Placeholder
                        'source': feedback.feedback_source,
                        'content_id': feedback.content_id
                    }
                
                # Store by embedding if available
                if feedback.embedding:
                    try:
                        if isinstance(feedback.embedding, str):
                            embedding = np.array([float(x) for x in feedback.embedding.split(',')])
                            embedding_hash = self._hash_embedding(embedding)
                            
                            embedding_corrections[embedding_hash] = {
                                'corrected_description': feedback.comment,
                                'confidence_adjustment': (feedback.rating - 3) * 0.2 if feedback.rating else None,
                                'embedding': embedding,
                                'content_id': feedback.content_id
                            }
                    except:
                        pass  # Skip invalid embeddings
                
            except Exception as e:
                logger.error(f"Error processing content feedback {feedback.id}: {str(e)}")
                continue
        
        self.content_corrections = content_corrections
        self.embedding_corrections = embedding_corrections
        self.confidence_adjustments = confidence_adjustments
        
        logger.info(f"✅ Loaded content corrections: {len(content_corrections)}")
        logger.info(f"✅ Loaded confidence adjustments: {len(confidence_adjustments)}")
        logger.info(f"✅ Loaded embedding corrections: {len(embedding_corrections)}")
        
        return len(content_corrections)
    
    def _hash_content(self, feedback) -> int:
        """Create hash for content identification"""
        # Combine content_id and person_id for unique identification
        content_key = f"{feedback.content_id}_{feedback.person_id}"
        return hash(content_key)
    
    def _hash_embedding(self, embedding: np.ndarray) -> int:
        """Create hash for CLIP embedding"""
        return hash(tuple(embedding[:10].round(3)))
    
    def predict_content_ensemble(self, base_description: str, base_confidence: float, content_id: int | None = None, person_id: int | None = None, clip_embedding: np.ndarray | None = None) -> tuple[str, float, dict]:
        """
        Ensemble prediction for content description and confidence.
        
        Args:
            base_description: Base CLIP model description.
            base_confidence: Base CLIP confidence.
            content_id: Content ID (if known).
            person_id: Person ID (if known).
            clip_embedding: CLIP embedding (if available).
            
        Returns:
            tuple: (final_description, final_confidence, correction_info).
        """
        
        # 1. Direct content lookup
        if content_id and person_id:
            content_key = f"{content_id}_{person_id}"
            content_hash = hash(content_key)
            
            if content_hash in self.content_corrections:
                correction = self.content_corrections[content_hash]
                logger.info(f"Direct content match found: {content_id}_{person_id}")
                
                final_description = correction['corrected_description']
                final_confidence = correction['confidence']
                
                return final_description, final_confidence, {
                    'method': 'direct_content_match',
                    'content_id': content_id,
                    'person_id': person_id,
                    'source': correction['source']
                }
            
            # Check confidence adjustment
            if content_hash in self.confidence_adjustments:
                adjustment = self.confidence_adjustments[content_hash]
                adjusted_confidence = base_confidence + adjustment['adjustment']
                adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))  # Clamp to [0,1]
                
                logger.info(f"Confidence adjustment found: {base_confidence:.3f} -> {adjusted_confidence:.3f}")
                
                return base_description, adjusted_confidence, {
                    'method': 'confidence_adjustment',
                    'original_confidence': base_confidence,
                    'adjustment': adjustment['adjustment'],
                    'content_id': content_id
                }
        
        # 2. Embedding similarity search
        if clip_embedding is not None and len(self.embedding_corrections) > 0:
            embedding_hash = self._hash_embedding(clip_embedding)
            
            # Exact embedding match
            if embedding_hash in self.embedding_corrections:
                correction = self.embedding_corrections[embedding_hash]
                logger.info(f"Exact CLIP embedding match found")
                
                if correction['corrected_description']:
                    return correction['corrected_description'], 0.9, {
                        'method': 'exact_embedding_match',
                        'content_id': correction['content_id']
                    }
            
            # Similarity-based correction
            best_similarity = -1
            best_correction = None
            
            # Normalize input embedding
            embedding_norm = clip_embedding / np.linalg.norm(clip_embedding)
            
            for emb_hash, correction in self.embedding_corrections.items():
                stored_embedding = correction['embedding']
                stored_embedding_norm = stored_embedding / np.linalg.norm(stored_embedding)
                
                # Cosine similarity
                similarity = np.dot(embedding_norm, stored_embedding_norm)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_correction = correction
            
            # High similarity threshold for CLIP (content can be more varied)
            if best_similarity > 0.92:  # Slightly lower than age model
                corrected_desc = best_correction['corrected_description']
                
                if corrected_desc:
                    # Blend confidence based on similarity
                    blended_confidence = base_confidence * 0.3 + best_similarity * 0.7
                    
                    logger.info(f"CLIP similarity correction: sim={best_similarity:.3f}")
                    
                    return corrected_desc, blended_confidence, {
                        'method': 'similarity_correction',
                        'similarity': best_similarity,
                        'base_description': base_description,
                        'corrected_description': corrected_desc,
                        'content_id': best_correction['content_id']
                    }
        
        # 3. No correction found - use base model
        logger.debug(f"No CLIP correction found, using base model")
        return base_description, base_confidence, {
            'method': 'base_model_only',
            'base_description': base_description,
            'base_confidence': base_confidence
        }
    
    def get_statistics(self) -> dict:
        """Get ensemble statistics"""
        stats = {
            'total_content_corrections': len(self.content_corrections),
            'total_confidence_adjustments': len(self.confidence_adjustments),
            'total_embedding_corrections': len(self.embedding_corrections),
        }
        
        if self.content_corrections:
            sources = [c['source'] for c in self.content_corrections.values()]
            stats.update({
                'manual_corrections': sources.count('MANUAL_USER'),
                'auto_corrections': len(sources) - sources.count('MANUAL_USER')
            })
        
        if self.confidence_adjustments:
            adjustments = [c['adjustment'] for c in self.confidence_adjustments.values()]
            stats.update({
                'avg_confidence_adjustment': f"{np.mean(adjustments):.3f}",
                'confidence_adjustment_range': f"{min(adjustments):.3f} to {max(adjustments):.3f}"
            })
        
        return stats
    
    def test_ensemble_predictions(self) -> list[dict]:
        """Test ensemble on known content feedback cases"""
        logger.info("Testing CLIP ensemble predictions...")
        
        if not self.content_corrections and not self.confidence_adjustments:
            logger.warning("No CLIP corrections loaded!")
            return []
        
        test_results = []
        
        # Test content corrections
        for content_hash, correction in list(self.content_corrections.items())[:3]:
            original_desc = correction['original_description']
            corrected_desc = correction['corrected_description']
            content_id = correction['content_id']
            person_id = correction['person_id']
            
            # Simulate base prediction
            simulated_base_confidence = 0.7  # Typical CLIP confidence
            
            # Test ensemble
            ensemble_desc, ensemble_conf, info = self.predict_content_ensemble(
                base_description=original_desc,
                base_confidence=simulated_base_confidence,
                content_id=content_id,
                person_id=person_id
            )
            
            test_results.append({
                'content_id': content_id,
                'person_id': person_id,
                'original_description': original_desc,
                'corrected_description': corrected_desc,
                'ensemble_description': ensemble_desc,
                'base_confidence': simulated_base_confidence,
                'ensemble_confidence': ensemble_conf,
                'method': info['method'],
                'match_quality': 'perfect' if ensemble_desc == corrected_desc else 'partial'
            })
            
            logger.info(f"CLIP Test {content_id[:8]}...")
            logger.info(f"  Original: {original_desc[:50]}...")
            logger.info(f"  Corrected: {corrected_desc[:50]}...")
            logger.info(f"  Ensemble: {ensemble_desc[:50]}...")
            logger.info(f"  Method: {info['method']}")
        
        return test_results
    
    def optimize_content_descriptions(self, content_list: list[dict]) -> list[dict]:
        """
        Optimize a list of content descriptions using ensemble corrections.
        
        Args:
            content_list: List of dicts with 'description', 'confidence', 'content_id', etc.
            
        Returns:
            List of optimized content descriptions.
        """
        optimized_list = []
        
        for content in content_list:
            optimized_desc, optimized_conf, info = self.predict_content_ensemble(
                base_description=content.get('description', ''),
                base_confidence=content.get('confidence', 0.5),
                content_id=content.get('content_id'),
                person_id=content.get('person_id'),
                clip_embedding=content.get('clip_embedding')
            )
            
            optimized_content = content.copy()
            optimized_content.update({
                'optimized_description': optimized_desc,
                'optimized_confidence': optimized_conf,
                'optimization_method': info['method'],
                'optimization_info': info
            })
            
            optimized_list.append(optimized_content)
        
        return optimized_list 