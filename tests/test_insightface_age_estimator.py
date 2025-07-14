import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

from app.ai import insightface_age_estimator


def test_estimator_init(app):
    with app.app_context():
        estimator = insightface_age_estimator.InsightFaceAgeEstimator()
        assert estimator is not None
        assert hasattr(estimator, "model")


def test_estimator_file_not_found(monkeypatch, app):
    """Model dosyası eksik olduğunda FileNotFoundError fırlatıyor mu?"""
    with app.app_context():
        monkeypatch.setattr('os.path.exists', lambda path: False)
        with pytest.raises(FileNotFoundError):
            insightface_age_estimator.InsightFaceAgeEstimator()

def test_estimator_custom_age_head_missing(monkeypatch, app):
    """CustomAgeHead .pth dosyası yoksa age_model ve custom_age_head None oluyor mu?"""
    with app.app_context():
        # Model path var, .pth dosyası yok
        monkeypatch.setattr('os.path.exists', lambda path: True)
        monkeypatch.setattr('os.listdir', lambda path: [])
        # Model yükleme işlemlerini atla
        with patch('app.ai.insightface_age_estimator.insightface.app.FaceAnalysis') as mock_face:
            mock_face.return_value.prepare.return_value = None
            est = insightface_age_estimator.InsightFaceAgeEstimator()
            assert est.custom_age_head is None
            assert est.age_model is None

def test_estimator_clip_model_load_fail(monkeypatch, app):
    """CLIP modeli yüklenemezse fallback olarak None atanıyor mu?"""
    with app.app_context():
        monkeypatch.setattr('os.path.exists', lambda path: True)
        monkeypatch.setattr('os.listdir', lambda path: ['model.pth'])
        with patch('app.ai.insightface_age_estimator.insightface.app.FaceAnalysis') as mock_face:
            mock_face.return_value.prepare.return_value = None
            with patch('app.ai.insightface_age_estimator.open_clip.create_model_and_transforms', side_effect=Exception("clip fail")):
                est = insightface_age_estimator.InsightFaceAgeEstimator()
                assert est.clip_model is None
                assert est.clip_preprocess is None

def test_estimate_age_face_none(app):
    """estimate_age fonksiyonunda face None ise default değerler dönüyor mu?"""
    with app.app_context():
        with patch('app.ai.insightface_age_estimator.insightface.app.FaceAnalysis') as mock_face:
            mock_face.return_value.prepare.return_value = None
            est = insightface_age_estimator.InsightFaceAgeEstimator()
            age, conf, pseudo = est.estimate_age(np.zeros((100,100,3), dtype=np.uint8), None)
            assert age == 25.0
            assert conf == 0.5
            assert pseudo is None

def test_estimate_age_bbox_invalid(app):
    """estimate_age fonksiyonunda face.bbox hatalıysa default confidence dönüyor mu?"""
    class Face:
        bbox = [10, 10, 5, 5]  # x2 < x1, y2 < y1
        age = 30
        embedding = np.ones(512)
    with app.app_context():
        with patch('app.ai.insightface_age_estimator.insightface.app.FaceAnalysis') as mock_face:
            mock_face.return_value.prepare.return_value = None
            est = insightface_age_estimator.InsightFaceAgeEstimator()
            age, conf, pseudo = est.estimate_age(np.zeros((100,100,3), dtype=np.uint8), Face())
            assert age == 30.0
            assert conf == 0.5
            assert pseudo is None

def test_estimate_age_embedding_none(app):
    """estimate_age fonksiyonunda embedding None ise custom age head çalışmıyor mu?"""
    class Face:
        bbox = [0, 0, 10, 10]
        age = 40
        embedding = None
    with app.app_context():
        with patch('app.ai.insightface_age_estimator.insightface.app.FaceAnalysis') as mock_face:
            mock_face.return_value.prepare.return_value = None
            est = insightface_age_estimator.InsightFaceAgeEstimator()
            est.age_model = MagicMock()
            age, conf, pseudo = est.estimate_age(np.ones((20,20,3), dtype=np.uint8), Face())
            # Custom age head çalışmaz, sadece buffalo yaş döner
            assert age == 40.0
            assert conf in (0.1, 0.5)

def test_estimate_age_age_model_none(app):
    """estimate_age fonksiyonunda age_model None ise custom age head çalışmıyor mu?"""
    class Face:
        bbox = [0, 0, 10, 10]
        age = 50
        embedding = np.ones(512)
    with app.app_context():
        with patch('app.ai.insightface_age_estimator.insightface.app.FaceAnalysis') as mock_face:
            mock_face.return_value.prepare.return_value = None
            est = insightface_age_estimator.InsightFaceAgeEstimator()
            est.age_model = None
            age, conf, pseudo = est.estimate_age(np.ones((20,20,3), dtype=np.uint8), Face())
            # Custom age head çalışmaz, sadece buffalo yaş döner
            assert age == 50.0
            assert conf in (0.1, 0.5) 