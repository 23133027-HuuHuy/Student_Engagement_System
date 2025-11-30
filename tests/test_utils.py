"""
Tests for utility modules
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.image_utils import ImageProcessor
from src.utils.visualization import Visualizer


class TestImageProcessor:
    """Test cases for ImageProcessor class"""
    
    def test_resize_no_aspect_ratio(self):
        """Test resize without keeping aspect ratio"""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = ImageProcessor.resize(image, (50, 50), keep_aspect_ratio=False)
        assert resized.shape[:2] == (50, 50)
    
    def test_resize_with_aspect_ratio(self):
        """Test resize keeping aspect ratio"""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = ImageProcessor.resize(image, (100, 100), keep_aspect_ratio=True)
        assert resized.shape[:2] == (100, 100)
    
    def test_to_grayscale(self):
        """Test grayscale conversion"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = ImageProcessor.to_grayscale(image)
        assert len(gray.shape) == 2
    
    def test_to_grayscale_already_gray(self):
        """Test grayscale on already gray image"""
        image = np.zeros((100, 100), dtype=np.uint8)
        gray = ImageProcessor.to_grayscale(image)
        assert len(gray.shape) == 2
    
    def test_normalize(self):
        """Test image normalization"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        normalized = ImageProcessor.normalize(image)
        assert normalized.dtype == np.float32
        # After normalization, values should be around 0 (depending on mean/std)
    
    def test_crop_face(self):
        """Test face cropping"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 100, 100)
        
        cropped = ImageProcessor.crop_face(image, bbox, margin=0.1)
        assert cropped is not None
        assert cropped.shape[0] > 0
        assert cropped.shape[1] > 0
    
    def test_augment_flip(self):
        """Test augmentation with flip"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[0, 0] = [255, 255, 255]  # Mark corner
        
        augmented = ImageProcessor.augment(image, flip=True)
        assert augmented.shape == image.shape


class TestVisualizer:
    """Test cases for Visualizer class"""
    
    def test_draw_face_bbox(self):
        """Test drawing face bounding box"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 100, 100)
        
        result = Visualizer.draw_face_bbox(image, bbox, label="Test")
        assert result is not None
        assert result.shape == image.shape
    
    def test_draw_engagement_overlay(self):
        """Test drawing engagement overlay"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 100, 100)
        
        result = Visualizer.draw_engagement_overlay(
            image, bbox, 'engaged', 75.0
        )
        assert result is not None
    
    def test_create_engagement_chart(self):
        """Test engagement chart creation"""
        history = [50, 60, 55, 70, 65, 80]
        
        chart = Visualizer.create_engagement_chart(history)
        assert chart is not None
        assert chart.shape == (200, 400, 3)
    
    def test_create_engagement_chart_empty(self):
        """Test engagement chart with insufficient data"""
        history = [50]
        
        chart = Visualizer.create_engagement_chart(history)
        assert chart is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
