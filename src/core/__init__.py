"""
Core modules for face detection and engagement classification
"""

from .face_detector import FaceDetector
from .emotion_classifier import EmotionClassifier
from .engagement_analyzer import EngagementAnalyzer

__all__ = ['FaceDetector', 'EmotionClassifier', 'EngagementAnalyzer']
