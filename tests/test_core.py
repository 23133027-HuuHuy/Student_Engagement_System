"""
Tests for Face Detector module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.face_detector import FaceDetector


class TestFaceDetector:
    """Test cases for FaceDetector class"""
    
    def test_initialization(self):
        """Test FaceDetector initialization"""
        detector = FaceDetector(method="haar")
        assert detector.method == "haar"
        assert detector.confidence_threshold == 0.5
    
    def test_invalid_method(self):
        """Test invalid detection method"""
        with pytest.raises(ValueError):
            FaceDetector(method="invalid_method")
    
    def test_detect_faces_empty_image(self):
        """Test detection on empty image"""
        detector = FaceDetector()
        faces = detector.detect_faces(None)
        assert faces == []
    
    def test_detect_faces_valid_image(self):
        """Test detection on valid image"""
        detector = FaceDetector()
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect_faces(dummy_image)
        assert isinstance(faces, list)
    
    def test_extract_face_region(self):
        """Test face region extraction"""
        detector = FaceDetector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 100, 100)
        
        face_region = detector.extract_face_region(image, bbox)
        assert face_region is not None
        assert face_region.shape[0] > 0
        assert face_region.shape[1] > 0
    
    def test_draw_faces(self):
        """Test drawing faces on image"""
        detector = FaceDetector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [
            {'bbox': (100, 100, 100, 100), 'confidence': 0.95}
        ]
        
        result = detector.draw_faces(image, faces)
        assert result is not None
        assert result.shape == image.shape


class TestEmotionClassifier:
    """Test cases for EmotionClassifier class"""
    
    def test_initialization(self):
        """Test EmotionClassifier initialization"""
        from src.core.emotion_classifier import EmotionClassifier
        
        classifier = EmotionClassifier()
        assert classifier.input_size == (48, 48)
    
    def test_preprocess_face(self):
        """Test face preprocessing"""
        from src.core.emotion_classifier import EmotionClassifier
        
        classifier = EmotionClassifier()
        face = np.zeros((100, 100, 3), dtype=np.uint8)
        
        processed = classifier.preprocess_face(face)
        assert processed.shape == (1, 48, 48, 1)
    
    def test_classify(self):
        """Test emotion classification"""
        from src.core.emotion_classifier import EmotionClassifier
        
        classifier = EmotionClassifier()
        face = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = classifier.classify(face)
        assert 'emotion' in result
        assert 'confidence' in result
        assert 'probabilities' in result


class TestEngagementAnalyzer:
    """Test cases for EngagementAnalyzer class"""
    
    def test_initialization(self):
        """Test EngagementAnalyzer initialization"""
        from src.core.engagement_analyzer import EngagementAnalyzer
        
        analyzer = EngagementAnalyzer()
        assert analyzer.emotion_weight == 0.4
        assert analyzer.attention_weight == 0.4
        assert analyzer.head_pose_weight == 0.2
    
    def test_analyze(self):
        """Test engagement analysis"""
        from src.core.engagement_analyzer import EngagementAnalyzer
        from src.core.emotion_classifier import Emotion
        
        analyzer = EngagementAnalyzer()
        emotion_result = {
            'emotion': Emotion.HAPPY,
            'confidence': 0.9
        }
        
        metrics = analyzer.analyze(emotion_result)
        assert metrics.engagement_score >= 0
        assert metrics.engagement_score <= 100
    
    def test_get_statistics(self):
        """Test getting statistics"""
        from src.core.engagement_analyzer import EngagementAnalyzer
        from src.core.emotion_classifier import Emotion
        
        analyzer = EngagementAnalyzer()
        
        # Add some samples
        for _ in range(5):
            analyzer.analyze({'emotion': Emotion.HAPPY, 'confidence': 0.9})
        
        stats = analyzer.get_statistics()
        assert stats['total_samples'] == 5
        assert 'average_score' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
