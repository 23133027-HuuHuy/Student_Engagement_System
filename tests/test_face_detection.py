"""
Tests for face detection module
"""

import pytest
import numpy as np
import cv2

from src.face_detection.detector import HaarCascadeDetector, get_detector


class TestHaarCascadeDetector:
    """Test cases for Haar Cascade detector"""
    
    def test_detector_initialization(self):
        """Test detector can be initialized"""
        detector = HaarCascadeDetector()
        assert detector is not None
        assert detector.classifier is not None
    
    def test_detect_returns_list(self):
        """Test detect method returns a list"""
        detector = HaarCascadeDetector()
        # Create a dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect(image)
        assert isinstance(result, list)
    
    def test_detect_with_grayscale(self):
        """Test detection works with grayscale images"""
        detector = HaarCascadeDetector()
        # Create a grayscale image
        image = np.zeros((100, 100), dtype=np.uint8)
        result = detector.detect(image)
        assert isinstance(result, list)


class TestGetDetector:
    """Test cases for detector factory function"""
    
    def test_get_haar_cascade_detector(self):
        """Test getting Haar Cascade detector"""
        detector = get_detector("haar_cascade")
        assert isinstance(detector, HaarCascadeDetector)
    
    def test_get_unknown_detector_raises_error(self):
        """Test getting unknown detector raises ValueError"""
        with pytest.raises(ValueError):
            get_detector("unknown_method")
