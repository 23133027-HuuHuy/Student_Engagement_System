"""
Emotion Recognizer - Nhận dạng cảm xúc
Phân loại cảm xúc từ khuôn mặt đã được phát hiện
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


# Định nghĩa các loại cảm xúc
EMOTIONS = [
    "angry",      # Tức giận
    "disgust",    # Ghê tởm
    "fear",       # Sợ hãi
    "happy",      # Vui vẻ
    "sad",        # Buồn
    "surprise",   # Ngạc nhiên
    "neutral"     # Bình thường
]


class BaseEmotionRecognizer(ABC):
    """Base class for emotion recognition"""
    
    @abstractmethod
    def predict(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Predict emotions from face image
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Dictionary of emotion probabilities
        """
        pass
    
    def get_dominant_emotion(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Get the dominant emotion from face image
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Tuple of (emotion_name, probability)
        """
        predictions = self.predict(face_image)
        dominant = max(predictions.items(), key=lambda x: x[1])
        return dominant


class SimpleEmotionRecognizer(BaseEmotionRecognizer):
    """
    Simple emotion recognizer
    Placeholder for custom model implementation
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize emotion recognizer
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.model = None
        
        # TODO: Load actual model here
        # if model_path:
        #     self.model = load_model(model_path)
    
    def predict(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Predict emotions from face image
        
        Note: This is a placeholder implementation
        Replace with actual model prediction
        """
        # Placeholder: return equal probabilities
        # TODO: Replace with actual model prediction
        n_emotions = len(EMOTIONS)
        return {emotion: 1.0 / n_emotions for emotion in EMOTIONS}


def preprocess_face(face_image: np.ndarray, 
                    target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    """
    Preprocess face image for emotion recognition
    
    Args:
        face_image: Input face image
        target_size: Target size for model input
        
    Returns:
        Preprocessed face image
    """
    # Convert to grayscale if needed
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image
    
    # Resize to target size
    resized = cv2.resize(gray, target_size)
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    return normalized


def get_recognizer(method: str = "simple", **kwargs) -> BaseEmotionRecognizer:
    """
    Factory function to get emotion recognizer
    
    Args:
        method: Recognition method
        **kwargs: Additional arguments
        
    Returns:
        Emotion recognizer instance
    """
    recognizers = {
        "simple": SimpleEmotionRecognizer,
        # TODO: Add more recognizers
    }
    
    if method not in recognizers:
        raise ValueError(f"Unknown recognizer method: {method}")
    
    return recognizers[method](**kwargs)
