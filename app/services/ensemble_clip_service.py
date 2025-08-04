import os
import json
import logging
import numpy as np
import torch
from datetime import datetime
from flask import current_app
from app import db
from app.models.feedback import Feedback
from app.models.content import ModelVersion
from app.utils.model_utils import save_torch_model

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
        
        # Get all content feedback (using category_feedback JSON field)
        feedbacks = Feedback.query.filter(
            Feedback.feedback_type == 'content'
        ).filter(
            db.or_(
                Feedback.category_feedback.isnot(None),  # Primary: category-based feedback
                Feedback.comment.isnot(None),           # Fallback: comment as corrected description
                Feedback.rating.isnot(None)             # Fallback: rating for confidence adjustment
            )
        ).all()
        
        logger.info(f"Found {len(feedbacks)} content feedback records")
        
        content_corrections = {}
        embedding_corrections = {}
        confidence_adjustments = {}
        used_feedback_ids = []  # Track which feedbacks are used for cleanup
        
        for feedback in feedbacks:
            try:
                # Content hash for exact matching
                content_hash = self._hash_content(feedback)
                
                # PRIMARY: Category-based feedback (JSON field)
                if feedback.category_feedback:
                    try:
                        if isinstance(feedback.category_feedback, str):
                            category_data = json.loads(feedback.category_feedback)
                        else:
                            category_data = feedback.category_feedback
                        
                        for category, feedback_value in category_data.items():
                            # Convert category feedback to corrections
                            category_hash = f"{content_hash}_{category}"
                            
                            # Map feedback values to confidence adjustments
                            adjustment_map = {
                                'over_estimated': -0.3,   # Reduce confidence if over-estimated
                                'under_estimated': 0.3,   # Increase confidence if under-estimated  
                                'accurate': 0.0,          # No change if accurate
                                'wrong_category': -0.5    # Heavy penalty for wrong category
                            }
                            
                            adjustment = adjustment_map.get(feedback_value, 0.0)
                            
                            confidence_adjustments[category_hash] = {
                                'category': category,
                                'feedback_value': feedback_value,
                                'adjustment': adjustment,
                                'original_confidence': 0.5,  # Placeholder
                                'source': feedback.feedback_source,
                                'content_id': feedback.content_id,
                                'frame_path': feedback.frame_path
                            }
                            
                            # Also create content correction entry
                            content_corrections[category_hash] = {
                                'category': category,
                                'feedback_value': feedback_value,
                                'confidence': 1.0 if feedback.feedback_source == 'MANUAL_USER_CONTENT_CORRECTION' else 0.8,
                                'source': feedback.feedback_source,
                                'content_id': feedback.content_id,
                                'frame_path': feedback.frame_path
                            }
                            
                    except Exception as json_err:
                        logger.error(f"Error parsing category_feedback JSON for feedback {feedback.id}: {str(json_err)}")
                
                # FALLBACK: Description correction (using comment field)
                if feedback.comment:
                    content_corrections[content_hash] = {
                        'original_description': 'Original content description',  # Placeholder
                        'corrected_description': feedback.comment,
                        'confidence': 1.0 if feedback.feedback_source == 'MANUAL_USER' else 0.8,
                        'source': feedback.feedback_source,
                        'content_id': feedback.content_id,
                        'person_id': feedback.person_id
                    }
                
                # FALLBACK: Confidence adjustment (using rating field as adjustment)
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
        
        # Mark all used feedbacks as used_in_ensemble and collect IDs for cleanup
        try:
            for feedback in feedbacks:
                if (feedback.category_feedback or feedback.comment or feedback.rating is not None):
                    self._mark_feedback_as_used(feedback)
                    used_feedback_ids.append(feedback.id)  # Track for cleanup
            
            db.session.commit()
            logger.info(f"✅ Marked {len(feedbacks)} feedbacks as used_in_ensemble")
            
        except Exception as mark_err:
            logger.error(f"Error marking feedbacks as used: {str(mark_err)}")
            db.session.rollback()
        
        logger.info(f"✅ Loaded content corrections: {len(content_corrections)}")
        logger.info(f"✅ Loaded confidence adjustments: {len(confidence_adjustments)}")
        logger.info(f"✅ Loaded embedding corrections: {len(embedding_corrections)}")
        logger.info(f"✅ Tracked {len(used_feedback_ids)} feedback IDs for cleanup")
        
        # Return both count and IDs for cleanup
        return {
            'corrections_count': len(content_corrections),
            'used_feedback_ids': used_feedback_ids
        }
    
    def load_feedback_corrections(self) -> dict:
        """
        Veritabanından içerik düzeltmelerini yükle
        """
        logger.info("Loading content feedback corrections from database...")
        
        try:
            # İçerik geri bildirimi olan kayıtları al
            feedbacks = db.session.query(Feedback).filter(
                Feedback.category_feedback.isnot(None),
                Feedback.embedding.isnot(None)
            ).all()
            
            corrections_loaded = 0
            
            for feedback in feedbacks:
                try:
                    # Embedding'i parse et
                    embedding_str = feedback.embedding
                    if embedding_str:
                        embedding = np.array([float(x) for x in embedding_str.split(',')])
                        
                        # Content ID düzeltmesi
                        if feedback.content_id and feedback.person_id:
                            content_key = f"{feedback.content_id}_{feedback.person_id}"
                            content_hash = hash(content_key)
                            
                            # Category feedback'i parse et
                            category_feedback = feedback.category_feedback
                            if isinstance(category_feedback, str):
                                import json
                                category_feedback = json.loads(category_feedback)
                            
                            # Düzeltilmiş açıklama oluştur
                            corrected_description = self._generate_corrected_description(category_feedback)
                            
                            self.content_corrections[content_hash] = {
                                'corrected_description': corrected_description,
                                'confidence': 0.9,
                                'source': feedback.feedback_source or 'MANUAL_USER',
                                'feedback_id': feedback.id,
                                'content_id': feedback.content_id,
                                'person_id': feedback.person_id,
                                'created_at': feedback.created_at.isoformat() if feedback.created_at else None
                            }
                        
                        # Embedding düzeltmesi
                        embedding_hash = self._hash_embedding(embedding)
                        self.embedding_corrections[embedding_hash] = {
                            'corrected_description': corrected_description if 'corrected_description' in locals() else None,
                            'embedding': embedding,
                            'confidence': 0.9,
                            'content_id': feedback.content_id,
                            'person_id': feedback.person_id,
                            'source': feedback.feedback_source or 'MANUAL_USER',
                            'feedback_id': feedback.id,
                            'created_at': feedback.created_at.isoformat() if feedback.created_at else None
                        }
                        
                        corrections_loaded += 1
                        
                        # Feedback'i kullanım ile işaretle
                        self._mark_feedback_as_used(feedback)
                        
                except Exception as e:
                    logger.warning(f"Feedback parsing error for ID {feedback.id}: {str(e)}")
                    continue
            
            # Değişiklikleri kaydet
            db.session.commit()
            
            logger.info(f"✅ {corrections_loaded} content correction loaded from database")
            return {
                'corrections_loaded': corrections_loaded,
                'total_content_corrections': len(self.content_corrections),
                'total_embedding_corrections': len(self.embedding_corrections)
            }
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"❌ Error loading content feedback corrections: {str(e)}")
            return {
                'corrections_loaded': 0,
                'error': str(e)
            }
    
    def _generate_corrected_description(self, category_feedback: dict) -> str:
        """Kategori feedback'ine göre düzeltilmiş açıklama oluştur"""
        descriptions = []
        
        if category_feedback.get('violence') == 'high':
            descriptions.append("violent content")
        elif category_feedback.get('violence') == 'low':
            descriptions.append("non-violent content")
            
        if category_feedback.get('adult_content') == 'high':
            descriptions.append("adult content")
        elif category_feedback.get('adult_content') == 'low':
            descriptions.append("safe content")
            
        if category_feedback.get('harassment') == 'high':
            descriptions.append("harassment content")
        elif category_feedback.get('harassment') == 'low':
            descriptions.append("respectful content")
        
        return ", ".join(descriptions) if descriptions else "general content"
    
    def _mark_feedback_as_used(self, feedback: Feedback):
        """Feedback'i ensemble'da kullanıldı olarak işaretle"""
        from datetime import datetime
        
        # Kullanım bilgilerini güncelle
        feedback.used_in_ensemble = True
        feedback.ensemble_usage_count = (feedback.ensemble_usage_count or 0) + 1
        feedback.last_used_at = datetime.now()
        
        # Ensemble model versiyonlarını güncelle
        if feedback.ensemble_model_versions is None:
            feedback.ensemble_model_versions = []
        
        # Mevcut versiyon bilgisini ekle
        current_version = f"ensemble_clip_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if current_version not in feedback.ensemble_model_versions:
            feedback.ensemble_model_versions.append(current_version)
        
        logger.debug(f"Feedback {feedback.id} marked as used in ensemble")
    
    def _increment_usage_count(self, feedback_id: int):
        """Feedback kullanım sayısını artır"""
        try:
            feedback = db.session.query(Feedback).filter(Feedback.id == feedback_id).first()
            if feedback:
                feedback.ensemble_usage_count = (feedback.ensemble_usage_count or 0) + 1
                feedback.last_used_at = datetime.now()
                db.session.commit()
                logger.debug(f"Usage count incremented for feedback {feedback_id}")
        except Exception as e:
            logger.warning(f"Failed to increment usage count for feedback {feedback_id}: {str(e)}")
            db.session.rollback()
    
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
        İçerik tahmini için ensemble yaklaşımı
        1. Önce content_id + person_id ile doğrudan arama
        2. Sonra embedding benzerliği ile arama
        3. Son olarak base model sonucunu döndür
        """
        logger.info(f"🔍 Ensemble content prediction - Base: {base_description}, Content: {content_id}, Person: {person_id}")
        
        # 1. Direct content lookup
        if content_id and person_id:
            content_key = f"{content_id}_{person_id}"
            content_hash = hash(content_key)
            
            if content_hash in self.content_corrections:
                correction = self.content_corrections[content_hash]
                logger.info(f"✅ Direct content match found: {content_id}_{person_id}")
                
                final_description = correction['corrected_description']
                final_confidence = correction['confidence']
                
                # Kullanım sayısını artır
                if 'feedback_id' in correction:
                    self._increment_usage_count(correction['feedback_id'])
                
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
                logger.info(f"✅ Exact CLIP embedding match found")
                
                # Kullanım sayısını artır
                if 'feedback_id' in correction:
                    self._increment_usage_count(correction['feedback_id'])
                
                if correction['corrected_description']:
                    return correction['corrected_description'], 0.9, {
                        'method': 'exact_embedding_match',
                        'content_id': correction['content_id'],
                        'source': correction['source']
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
            
            # Yüksek benzerlik varsa düzeltmeyi uygula
            if best_similarity > 0.95 and best_correction and best_correction['corrected_description']:
                confidence = 0.9 * best_similarity
                
                logger.info(f"✅ Similarity match ({best_similarity:.3f}) -> {best_correction['corrected_description']}")
                
                # Kullanım sayısını artır
                if 'feedback_id' in best_correction:
                    self._increment_usage_count(best_correction['feedback_id'])
                
                return best_correction['corrected_description'], confidence, {
                    'method': 'similarity_match',
                    'similarity': best_similarity,
                    'content_id': best_correction['content_id'],
                    'source': best_correction['source']
                }
        
        # 3. Base model sonucunu döndür
        logger.info(f"📊 Using base model result: {base_description}")
        return base_description, base_confidence, {
            'method': 'base_model',
            'no_correction_found': True
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
    
    def save_ensemble_corrections_as_version(self) -> ModelVersion:
        """
        Ensemble düzeltmelerini .pth dosyası olarak kaydet ve model versiyonu oluştur
        """
        logger.info("Saving CLIP ensemble corrections as model version...")
        
        try:
            # Versiyon numarasını belirle
            last_version = ModelVersion.query.filter_by(
                model_type='content'
            ).order_by(ModelVersion.version.desc()).first()
            
            new_version_num = 1 if last_version is None else last_version.version + 1
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_name = f"ensemble_clip_v{new_version_num}_{timestamp}"
            
            # Versiyon klasörü oluştur
            version_dir = os.path.join(
                current_app.config['MODELS_FOLDER'],
                'content',
                'ensemble_versions',
                version_name
            )
            os.makedirs(version_dir, exist_ok=True)
            
            # Ensemble verilerini .pth formatında kaydet
            ensemble_data = {
                'model_type': 'ensemble_clip',
                'content_corrections': self.content_corrections,
                'embedding_corrections': self.embedding_corrections,
                'confidence_adjustments': self.confidence_adjustments,
                'version': new_version_num,
                'version_name': version_name,
                'created_at': datetime.now().isoformat(),
                'total_content_corrections': len(self.content_corrections),
                'total_embedding_corrections': len(self.embedding_corrections),
                'total_confidence_adjustments': len(self.confidence_adjustments)
            }
            
            # .pth dosyası olarak kaydet
            model_path = os.path.join(version_dir, 'ensemble_corrections.pth')
            torch.save(ensemble_data, model_path)
            logger.info(f"CLIP ensemble corrections saved to: {model_path}")
            
            # Metadata oluştur
            metadata = {
                'version': new_version_num,
                'version_name': version_name,
                'created_at': datetime.now().isoformat(),
                'model_type': 'ensemble_clip',
                'total_content_corrections': len(self.content_corrections),
                'total_embedding_corrections': len(self.embedding_corrections),
                'total_confidence_adjustments': len(self.confidence_adjustments),
                'correction_sources': self._get_content_correction_sources_stats(),
                'metrics': self._calculate_clip_ensemble_metrics()
            }
            
            # Metadata kaydet
            metadata_path = os.path.join(version_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            logger.info(f"Metadata saved to: {metadata_path}")
            
            # Tüm aktif versiyonları devre dışı bırak
            db.session.query(ModelVersion).filter_by(
                model_type='content',
                is_active=True
            ).update({ModelVersion.is_active: False})
            
            # Veritabanında yeni versiyon oluştur
            model_version = ModelVersion(
                model_type='content',
                version=new_version_num,
                version_name=version_name,
                created_at=datetime.now(),
                metrics=metadata['metrics'],
                is_active=True,
                training_samples=len(self.content_corrections),
                validation_samples=0,  # Ensemble için N/A
                epochs=0,  # Ensemble için N/A
                file_path=version_dir,
                weights_path=model_path,
                used_feedback_ids=self._get_used_clip_feedback_ids()
            )
            
            db.session.add(model_version)
            db.session.commit()
            
            logger.info(f"✅ CLIP ensemble model version created: {version_name}")
            logger.info(f"   Content corrections: {len(self.content_corrections)}")
            logger.info(f"   Embedding corrections: {len(self.embedding_corrections)}")
            logger.info(f"   Confidence adjustments: {len(self.confidence_adjustments)}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Error saving CLIP ensemble model version: {str(e)}")
            db.session.rollback()
            raise e
    
    def _get_content_correction_sources_stats(self) -> dict:
        """İçerik düzeltme kaynaklarının istatistiklerini döndür"""
        sources = {'MANUAL_USER': 0, 'AUTO_CORRECTION': 0}
        for correction in self.content_corrections.values():
            source = correction.get('source', 'UNKNOWN')
            if source in sources:
                sources[source] += 1
            else:
                sources['OTHER'] = sources.get('OTHER', 0) + 1
        return sources
    
    def _calculate_clip_ensemble_metrics(self) -> dict:
        """CLIP Ensemble metrikleri hesapla"""
        total_content = len(self.content_corrections)
        total_embedding = len(self.embedding_corrections)
        total_confidence = len(self.confidence_adjustments)
        
        # Confidence adjustment ortalaması
        if self.confidence_adjustments:
            adjustments = [adj.get('adjustment', 0.0) for adj in self.confidence_adjustments.values()]
            avg_adjustment = np.mean(adjustments) if adjustments else 0.0
        else:
            avg_adjustment = 0.0
        
        return {
            'total_content_corrections': total_content,
            'total_embedding_corrections': total_embedding,
            'total_confidence_adjustments': total_confidence,
            'average_confidence_adjustment': float(avg_adjustment),
            'coverage_ratio': float(total_embedding / max(total_content, 1))
        }
    
    def _get_used_clip_feedback_ids(self) -> list:
        """Kullanılan CLIP feedback ID'lerini döndür"""
        feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'content') | 
            (Feedback.feedback_type == 'content_rating')
        ).all()
        
        used_ids = []
        for feedback in feedbacks:
            # Content hash'e sahip feedback'leri kullan
            if hasattr(feedback, 'content_hash') and feedback.content_hash:
                if feedback.content_hash in self.content_corrections:
                    used_ids.append(feedback.id)
        
        return used_ids 

    def cleanup_used_training_data(self, used_feedback_ids: list[int], model_version_name: str = None) -> dict:
        """
        Eğitimde kullanılan content feedback'leri tamamen temizler (VT + dosyalar)
        
        Args:
            used_feedback_ids: Kullanılan feedback ID'leri  
            model_version_name: Model versiyon adı (opsiyonel)
            
        Returns:
            dict: Temizlik raporu
        """
        from flask import current_app
        import os
        
        logger.info(f"Cleaning up CLIP content training data for model {model_version_name}")
        
        cleanup_report = {
            'deleted_feedbacks': 0,
            'deleted_files': 0,
            'deleted_directories': 0,
            'errors': []
        }
        
        try:
            # 1. Önce feedback'leri al (dosya yollarını almak için)
            feedbacks_to_delete = Feedback.query.filter(
                Feedback.id.in_(used_feedback_ids),
                Feedback.feedback_type == 'content'
            ).all()
            
            # 2. İlgili dosya yollarını topla
            frame_paths = set()
            content_ids = set()
            
            for feedback in feedbacks_to_delete:
                if feedback.frame_path:
                    frame_paths.add(feedback.frame_path)
                if feedback.content_id:
                    content_ids.add(feedback.content_id)
            
            logger.info(f"Found {len(frame_paths)} frame paths and {len(content_ids)} content IDs to clean")
            
            # 3. Processed klasöründeki ilgili dosyaları sil
            processed_dir = current_app.config.get('PROCESSED_FOLDER', 'storage/processed')
            
            # Frame klasörlerini kontrol et ve sil
            if os.path.exists(processed_dir):
                for item in os.listdir(processed_dir):
                    item_path = os.path.join(processed_dir, item)
                    
                    if os.path.isdir(item_path) and item.startswith('frames_'):
                        # Bu frame klasöründe silinecek content_id'ler var mı kontrol et
                        try:
                            # Analysis ID'yi klasör adından çıkar
                            analysis_id = item.replace('frames_', '')
                            
                            # Eğer bu analysis_id silinecek content_id'ler arasındaysa
                            if int(analysis_id) in content_ids:
                                import shutil
                                shutil.rmtree(item_path)
                                cleanup_report['deleted_directories'] += 1
                                logger.info(f"Deleted directory: {item_path}")
                                
                        except (ValueError, OSError) as e:
                            cleanup_report['errors'].append(f"Error deleting directory {item_path}: {str(e)}")
                            logger.warning(f"Error deleting directory {item_path}: {str(e)}")
            
            # 4. VT'den feedback'leri sil
            for feedback in feedbacks_to_delete:
                try:
                    db.session.delete(feedback)
                    cleanup_report['deleted_feedbacks'] += 1
                    logger.info(f"Deleted content feedback: {feedback.id}")
                except Exception as e:
                    cleanup_report['errors'].append(f"Error deleting feedback {feedback.id}: {str(e)}")
                    logger.warning(f"Error deleting feedback {feedback.id}: {str(e)}")
            
            # 5. Değişiklikleri commit et
            db.session.commit()
            
            logger.info(f"Content cleanup completed. Deleted {cleanup_report['deleted_feedbacks']} feedbacks, "
                       f"{cleanup_report['deleted_files']} files, {cleanup_report['deleted_directories']} directories")
                       
        except Exception as e:
            db.session.rollback()
            cleanup_report['errors'].append(f"General cleanup error: {str(e)}")
            logger.error(f"Content cleanup error: {str(e)}")
            
        return cleanup_report 