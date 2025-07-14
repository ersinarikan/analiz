import os
import logging
from flask import current_app
from app.services.ensemble_age_service import EnsembleAgeService
from app.services.ensemble_clip_service import EnsembleClipService

logger = logging.getLogger('app.ensemble_integration')

class EnsembleIntegrationService:
    """
    Farklı ensemble servislerini entegre eden ana servis sınıfı.
    - CLIP ve Age ensemble servislerini bir arada yönetir.
    """
    
    def __init__(self):
        self.age_ensemble = EnsembleAgeService()
        self.clip_ensemble = EnsembleClipService()
        self.initialized = False
        logger.info("EnsembleIntegrationService initialized")
    
    def initialize(self) -> dict:
        """Load all ensemble corrections"""
        logger.info("🚀 Initializing ensemble systems...")
        
        try:
            # Load age corrections
            age_count = self.age_ensemble.load_feedback_corrections()
            logger.info(f"✅ Age ensemble: {age_count} people corrections loaded")
            
            # Load CLIP corrections
            clip_count = self.clip_ensemble.load_content_corrections()
            logger.info(f"✅ CLIP ensemble: {clip_count} content corrections loaded")
            
            self.initialized = True
            
            # Get statistics
            age_stats = self.age_ensemble.get_statistics()
            clip_stats = self.clip_ensemble.get_statistics()
            
            logger.info("📊 Ensemble system ready!")
            logger.info(f"   Age corrections: {age_stats['total_people_corrections']}")
            logger.info(f"   CLIP corrections: {clip_stats['total_content_corrections']}")
            
            return {
                'success': True,
                'age_corrections': age_count,
                'clip_corrections': clip_count,
                'age_stats': age_stats,
                'clip_stats': clip_stats
            }
            
        except Exception as e:
            logger.error(f"Ensemble initialization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_age_enhanced(self, base_age_prediction: float, embedding: list, person_id: str | None = None) -> dict:
        """
        Enhanced age prediction using ensemble
        
        Args:
            base_age_prediction: Base model prediction
            embedding: Face embedding
            person_id: Person ID if known
            
        Returns:
            dict: Enhanced prediction with metadata
        """
        if not self.initialized:
            logger.warning("Ensemble not initialized, using base prediction")
            return {
                'final_age': base_age_prediction,
                'confidence': 0.5,
                'method': 'base_model_uninitialized',
                'improvement': 0.0
            }
        
        try:
            final_age, confidence, info = self.age_ensemble.predict_age_ensemble(
                base_age_prediction, embedding, person_id
            )
            
            improvement = abs(base_age_prediction - final_age)
            
            return {
                'final_age': final_age,
                'confidence': confidence,
                'method': info['method'],
                'correction_info': info,
                'base_age': base_age_prediction,
                'improvement': improvement,
                'ensemble_active': True
            }
            
        except Exception as e:
            logger.error(f"Age ensemble prediction failed: {str(e)}")
            return {
                'final_age': base_age_prediction,
                'confidence': 0.5,
                'method': 'base_model_fallback',
                'error': str(e),
                'improvement': 0.0
            }
    
    def predict_content_enhanced(self, base_description: str, base_confidence: float, content_id: str | None = None, person_id: str | None = None, clip_embedding: list | None = None) -> dict:
        """
        Enhanced content prediction using CLIP ensemble
        
        Args:
            base_description: Base CLIP description
            base_confidence: Base CLIP confidence
            content_id: Content ID if known
            person_id: Person ID if known
            clip_embedding: CLIP embedding if available
            
        Returns:
            dict: Enhanced prediction with metadata
        """
        if not self.initialized:
            logger.warning("Ensemble not initialized, using base prediction")
            return {
                'final_description': base_description,
                'final_confidence': base_confidence,
                'method': 'base_model_uninitialized',
                'confidence_improvement': 0.0
            }
        
        try:
            final_desc, final_conf, info = self.clip_ensemble.predict_content_ensemble(
                base_description, base_confidence, content_id, person_id, clip_embedding
            )
            
            confidence_improvement = final_conf - base_confidence
            
            return {
                'final_description': final_desc,
                'final_confidence': final_conf,
                'method': info['method'],
                'correction_info': info,
                'base_description': base_description,
                'base_confidence': base_confidence,
                'confidence_improvement': confidence_improvement,
                'ensemble_active': True
            }
            
        except Exception as e:
            logger.error(f"CLIP ensemble prediction failed: {str(e)}")
            return {
                'final_description': base_description,
                'final_confidence': base_confidence,
                'method': 'base_model_fallback',
                'error': str(e),
                'confidence_improvement': 0.0
            }
    
    def analyze_image_enhanced(self, image: str, face_data: dict, content_data: dict) -> dict:
        """
        Complete enhanced analysis using both ensembles
        
        Args:
            image: Input image
            face_data: Face analysis data from base models
            content_data: Content analysis data from base models
            
        Returns:
            dict: Enhanced analysis results
        """
        results = {
            'enhanced_faces': [],
            'enhanced_content': {},
            'ensemble_stats': {
                'age_corrections_applied': 0,
                'clip_corrections_applied': 0,
                'total_improvements': 0
            }
        }
        
        # Enhance face predictions
        if 'faces' in face_data:
            for face in face_data['faces']:
                base_age = face.get('age', 25)
                embedding = face.get('embedding')
                person_id = face.get('person_id')
                
                enhanced_age = self.predict_age_enhanced(
                    base_age, embedding, person_id
                )
                
                enhanced_face = face.copy()
                enhanced_face.update({
                    'enhanced_age': enhanced_age['final_age'],
                    'age_confidence': enhanced_age['confidence'],
                    'age_method': enhanced_age['method'],
                    'age_improvement': enhanced_age['improvement'],
                    'original_age': base_age
                })
                
                results['enhanced_faces'].append(enhanced_face)
                
                if enhanced_age['method'] != 'base_model_only':
                    results['ensemble_stats']['age_corrections_applied'] += 1
        
        # Enhance content predictions
        if 'content' in content_data:
            base_desc = content_data['content'].get('description', '')
            base_conf = content_data['content'].get('confidence', 0.5)
            content_id = content_data['content'].get('content_id')
            clip_embedding = content_data['content'].get('clip_embedding')
            
            enhanced_content = self.predict_content_enhanced(
                base_desc, base_conf, content_id, None, clip_embedding
            )
            
            results['enhanced_content'] = {
                'description': enhanced_content['final_description'],
                'confidence': enhanced_content['final_confidence'],
                'method': enhanced_content['method'],
                'confidence_improvement': enhanced_content['confidence_improvement'],
                'original_description': base_desc,
                'original_confidence': base_conf
            }
            
            if enhanced_content['method'] != 'base_model_only':
                results['ensemble_stats']['clip_corrections_applied'] += 1
        
        # Calculate total improvements
        results['ensemble_stats']['total_improvements'] = (
            results['ensemble_stats']['age_corrections_applied'] + 
            results['ensemble_stats']['clip_corrections_applied']
        )
        
        return results
    
    def get_system_status(self) -> dict:
        """Get ensemble system status"""
        if not self.initialized:
            return {
                'status': 'not_initialized',
                'message': 'Ensemble system not initialized'
            }
        
        age_stats = self.age_ensemble.get_statistics()
        clip_stats = self.clip_ensemble.get_statistics()
        
        return {
            'status': 'active',
            'initialized': True,
            'age_ensemble': {
                'people_corrections': age_stats['total_people_corrections'],
                'embedding_corrections': age_stats['total_embedding_corrections'],
                'manual_corrections': age_stats['manual_corrections'],
                'pseudo_corrections': age_stats['pseudo_corrections']
            },
            'clip_ensemble': {
                'content_corrections': clip_stats['total_content_corrections'],
                'confidence_adjustments': clip_stats['total_confidence_adjustments'],
                'embedding_corrections': clip_stats['total_embedding_corrections']
            },
            'capabilities': {
                'age_correction': age_stats['total_people_corrections'] > 0,
                'content_correction': clip_stats['total_content_corrections'] > 0,
                'confidence_adjustment': clip_stats['total_confidence_adjustments'] > 0,
                'embedding_similarity': (
                    age_stats['total_embedding_corrections'] > 0 or 
                    clip_stats['total_embedding_corrections'] > 0
                )
            }
        }
    
    def refresh_corrections(self) -> dict:
        """Refresh all ensemble corrections"""
        try:
            age_result = self.age_ensemble.load_age_corrections()
            clip_result = self.clip_ensemble.load_content_corrections()
            
            return {
                'success': True,
                'age_corrections': age_result,
                'clip_corrections': clip_result,
                'age_stats': self.age_ensemble.get_correction_stats(),
                'clip_stats': self.clip_ensemble.get_correction_stats()
            }
        except Exception as e:
            logger.error(f"Refresh corrections error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def reset_ensemble_corrections(self, model_type: str) -> dict:
        """Reset ensemble corrections for specified model type"""
        try:
            if model_type == 'age':
                # Clear age corrections
                self.age_ensemble.feedback_corrections = {}
                self.age_ensemble.embedding_corrections = {}
                logger.info("Age ensemble corrections cleared")
                
                return {
                    'success': True,
                    'message': 'Age ensemble corrections cleared. Base model active.',
                    'corrections_cleared': len(self.age_ensemble.feedback_corrections)
                }
                
            elif model_type == 'content':
                # Clear CLIP corrections
                self.clip_ensemble.content_corrections = {}
                self.clip_ensemble.embedding_corrections = {}
                self.clip_ensemble.confidence_adjustments = {}
                logger.info("CLIP ensemble corrections cleared")
                
                return {
                    'success': True,
                    'message': 'CLIP ensemble corrections cleared. Base OpenCLIP active.',
                    'corrections_cleared': len(self.clip_ensemble.content_corrections)
                }
            else:
                return {
                    'success': False,
                    'error': f'Invalid model type: {model_type}'
                }
                
        except Exception as e:
            logger.error(f"Reset ensemble corrections error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_real_age_distribution(self) -> dict:
        """Get real-time age distribution from feedback database"""
        try:
            from app import db
            from app.models import Feedback
            
            # Query real feedback data - only age feedbacks
            feedbacks = db.session.query(Feedback).filter(Feedback.feedback_type == 'age').all()
            
            age_ranges = {
                "0-10": 0, "11-20": 0, "21-30": 0, "31-40": 0,
                "41-50": 0, "51-60": 0, "61+": 0
            }
            
            for feedback in feedbacks:
                corrected_age = feedback.corrected_age
                if corrected_age <= 10:
                    age_ranges["0-10"] += 1
                elif corrected_age <= 20:
                    age_ranges["11-20"] += 1
                elif corrected_age <= 30:
                    age_ranges["21-30"] += 1
                elif corrected_age <= 40:
                    age_ranges["31-40"] += 1
                elif corrected_age <= 50:
                    age_ranges["41-50"] += 1
                elif corrected_age <= 60:
                    age_ranges["51-60"] += 1
                else:
                    age_ranges["61+"] += 1
            
            return age_ranges
            
        except Exception as e:
            logger.error(f"Real age distribution error: {str(e)}")
            # Fallback to empty distribution
            return {"0-10": 0, "11-20": 0, "21-30": 0, "31-40": 0, "41-50": 0, "51-60": 0, "61+": 0}
    
    def get_real_error_distribution(self) -> dict:
        """Get real-time error distribution from feedback accuracy"""
        try:
            from app import db
            from app.models import Feedback
            
            feedbacks = db.session.query(Feedback).filter(Feedback.feedback_type == 'age').all()
            
            error_ranges = {
                "0 yaş fark": 0, "1 yaş fark": 0, 
                "2 yaş fark": 0, "3+ yaş fark": 0
            }
            
            ensemble_active = self.get_system_status()['capabilities']['age_correction']
            
            for feedback in feedbacks:
                if ensemble_active:
                    # Ensemble aktifse çoğunlukla perfect
                    person_id = feedback.person_id
                    if person_id and person_id in self.age_ensemble.feedback_corrections:
                        error_ranges["0 yaş fark"] += 1  # Perfect correction
                    else:
                        # Base model error
                        original_age = feedback.pseudo_label_original_age or 25  # fallback
                        base_error = abs(original_age - feedback.corrected_age)
                        if base_error <= 1:
                            error_ranges["1 yaş fark"] += 1
                        elif base_error <= 2:
                            error_ranges["2 yaş fark"] += 1
                        else:
                            error_ranges["3+ yaş fark"] += 1
                else:
                    # Base model accuracy
                    original_age = feedback.pseudo_label_original_age or 25  # fallback
                    error = abs(original_age - feedback.corrected_age)
                    if error == 0:
                        error_ranges["0 yaş fark"] += 1
                    elif error == 1:
                        error_ranges["1 yaş fark"] += 1
                    elif error == 2:
                        error_ranges["2 yaş fark"] += 1
                    else:
                        error_ranges["3+ yaş fark"] += 1
            
            return error_ranges
            
        except Exception as e:
            logger.error(f"Real error distribution error: {str(e)}")
            # Fallback to default
            return {"0 yaş fark": 1, "1 yaş fark": 1, "2 yaş fark": 1, "3+ yaş fark": 1}

# Global ensemble service instance
_ensemble_service = None

def get_ensemble_service() -> EnsembleIntegrationService:
    """Get global ensemble service instance"""
    global _ensemble_service
    if _ensemble_service is None:
        _ensemble_service = EnsembleIntegrationService()
        _ensemble_service.initialize()
    return _ensemble_service 