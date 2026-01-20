# Models package
# Tüm modelleri buradan import edin
# Bu import'lar SQLAlchemy metadata'ya tablo tanımlarını ekler

from app.models.file import File
from app.models.analysis import Analysis, ContentDetection, AgeEstimation
from app.models.feedback import Feedback
from app.models.content import ModelVersion
from app.models.clip_training import CLIPTrainingSession 