"""
Face Detector - Phát hiện khuôn mặt
Hỗ trợ nhiều phương pháp: Haar Cascade, dlib, MediaPipe
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class BaseFaceDetector(ABC):
    """Base class for face detection"""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        pass


class HaarCascadeDetector(BaseFaceDetector):
    """Face detection using Haar Cascade Classifier"""
    
    def __init__(self, 
                 cascade_path: Optional[str] = None,
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (30, 30)):
        """
        Initialize Haar Cascade detector
        
        Args:
            cascade_path: Path to cascade XML file
            scale_factor: Scale factor for detection
            min_neighbors: Minimum neighbors for detection
            min_size: Minimum face size
        """
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.classifier = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascade"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        return [tuple(face) for face in faces]


def get_detector(method: str = "haar_cascade", **kwargs) -> BaseFaceDetector:
    """
    Factory function to get face detector
    
    Args:
        method: Detection method ("haar_cascade", "dlib", "mediapipe")
        **kwargs: Additional arguments for detector
        
    Returns:
        Face detector instance
    """
    detectors = {
        "haar_cascade": HaarCascadeDetector,
        # TODO: Add dlib and mediapipe detectors
    }
    
    if method not in detectors:
        raise ValueError(f"Unknown detector method: {method}")
    
    return detectors[method](**kwargs)
