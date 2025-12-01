"""
Emotion Classification Module - Mô-đun phân loại cảm xúc
Phân loại các trạng thái cảm xúc từ khuôn mặt
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Emotion(Enum):
    """Các trạng thái cảm xúc được hỗ trợ"""
    HAPPY = "happy"           # Vui vẻ
    SAD = "sad"               # Buồn
    ANGRY = "angry"           # Tức giận
    SURPRISED = "surprised"   # Ngạc nhiên
    NEUTRAL = "neutral"       # Trung tính
    FEAR = "fear"             # Sợ hãi
    DISGUST = "disgust"       # Ghê tởm
    CONFUSED = "confused"     # Bối rối


class EmotionClassifier:
    """
    Lớp phân loại cảm xúc từ khuôn mặt
    
    Sử dụng các mô hình:
    - CNN-based (default)
    - Pre-trained models (FER, DeepFace)
    """
    
    # Ánh xạ cảm xúc sang tiếng Việt
    EMOTION_LABELS = {
        Emotion.HAPPY: "Vui vẻ",
        Emotion.SAD: "Buồn",
        Emotion.ANGRY: "Tức giận",
        Emotion.SURPRISED: "Ngạc nhiên",
        Emotion.NEUTRAL: "Trung tính",
        Emotion.FEAR: "Sợ hãi",
        Emotion.DISGUST: "Ghê tởm",
        Emotion.CONFUSED: "Bối rối"
    }
    
    def __init__(self, model_path: Optional[str] = None, input_size: Tuple[int, int] = (48, 48)):
        """
        Khởi tạo Emotion Classifier
        
        Args:
            model_path: Đường dẫn đến mô hình đã train
            input_size: Kích thước đầu vào của mô hình (width, height)
        """
        self.model_path = model_path
        self.input_size = input_size
        self.model = None
        self.is_loaded = False
        
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load mô hình phân loại cảm xúc"""
        try:
            # Placeholder: Sẽ load mô hình thực khi có file model
            # self.model = tf.keras.models.load_model(self.model_path)
            self.is_loaded = True
            logger.info(f"Đã load mô hình từ {self.model_path}")
        except Exception as e:
            logger.error(f"Lỗi khi load mô hình: {e}")
            self.is_loaded = False
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý hình ảnh khuôn mặt cho mô hình
        
        Args:
            face_image: Hình ảnh khuôn mặt (BGR)
            
        Returns:
            Hình ảnh đã được tiền xử lý
        """
        # Chuyển sang grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize về kích thước chuẩn
        resized = cv2.resize(gray, self.input_size)
        
        # Normalize
        normalized = resized / 255.0
        
        # Thêm dimension cho batch và channel
        preprocessed = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
        
        return preprocessed
    
    def classify(self, face_image: np.ndarray) -> Dict:
        """
        Phân loại cảm xúc từ hình ảnh khuôn mặt
        
        Args:
            face_image: Hình ảnh khuôn mặt (BGR)
            
        Returns:
            Dictionary chứa:
            - emotion: Cảm xúc được dự đoán
            - confidence: Độ tin cậy
            - probabilities: Xác suất cho tất cả các lớp
        """
        if face_image is None or face_image.size == 0:
            return {
                'emotion': Emotion.NEUTRAL,
                'confidence': 0.0,
                'probabilities': {}
            }
        
        # Tiền xử lý
        preprocessed = self.preprocess_face(face_image)
        
        # Dự đoán (placeholder - sẽ sử dụng model thực)
        # Tạm thời trả về kết quả mặc định
        probabilities = {
            Emotion.HAPPY: 0.1,
            Emotion.SAD: 0.1,
            Emotion.ANGRY: 0.05,
            Emotion.SURPRISED: 0.05,
            Emotion.NEUTRAL: 0.6,
            Emotion.FEAR: 0.05,
            Emotion.DISGUST: 0.025,
            Emotion.CONFUSED: 0.025
        }
        
        # Tìm cảm xúc có xác suất cao nhất
        predicted_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_emotion]
        
        return {
            'emotion': predicted_emotion,
            'emotion_label': self.EMOTION_LABELS[predicted_emotion],
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def classify_batch(self, face_images: List[np.ndarray]) -> List[Dict]:
        """
        Phân loại cảm xúc cho nhiều khuôn mặt
        
        Args:
            face_images: Danh sách hình ảnh khuôn mặt
            
        Returns:
            Danh sách kết quả phân loại
        """
        results = []
        for face in face_images:
            result = self.classify(face)
            results.append(result)
        return results
    
    def get_emotion_color(self, emotion: Emotion) -> Tuple[int, int, int]:
        """
        Lấy màu tương ứng với cảm xúc (để hiển thị)
        
        Args:
            emotion: Loại cảm xúc
            
        Returns:
            Màu BGR
        """
        color_map = {
            Emotion.HAPPY: (0, 255, 0),       # Xanh lá
            Emotion.SAD: (255, 0, 0),          # Xanh dương
            Emotion.ANGRY: (0, 0, 255),        # Đỏ
            Emotion.SURPRISED: (0, 255, 255),  # Vàng
            Emotion.NEUTRAL: (128, 128, 128),  # Xám
            Emotion.FEAR: (128, 0, 128),       # Tím
            Emotion.DISGUST: (0, 128, 0),      # Xanh đậm
            Emotion.CONFUSED: (255, 165, 0)    # Cam
        }
        return color_map.get(emotion, (255, 255, 255))
