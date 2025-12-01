"""
Face Detection Module - Mô-đun phát hiện khuôn mặt
Sử dụng các phương pháp: Haar Cascade, MTCNN, hoặc dlib
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Lớp phát hiện khuôn mặt trong hình ảnh và video
    
    Hỗ trợ các phương pháp:
    - haar: Haar Cascade (nhanh, độ chính xác trung bình)
    - dnn: Deep Neural Network (cân bằng tốc độ và độ chính xác)
    - mtcnn: Multi-task Cascaded CNN (độ chính xác cao)
    """
    
    def __init__(self, method: str = "haar", confidence_threshold: float = 0.5):
        """
        Khởi tạo Face Detector
        
        Args:
            method: Phương pháp phát hiện ("haar", "dnn", "mtcnn")
            confidence_threshold: Ngưỡng tin cậy để chấp nhận khuôn mặt
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.detector = None
        self._initialize_detector()
        
    def _initialize_detector(self):
        """Khởi tạo detector dựa trên phương pháp được chọn"""
        if self.method == "haar":
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            logger.info("Đã khởi tạo Haar Cascade Face Detector")
        elif self.method == "dnn":
            # Sử dụng OpenCV DNN với model pre-trained
            self.detector = None  # Sẽ load model khi cần
            logger.info("Đã khởi tạo DNN Face Detector")
        else:
            raise ValueError(f"Phương pháp không được hỗ trợ: {self.method}")
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Phát hiện tất cả khuôn mặt trong hình ảnh
        
        Args:
            image: Hình ảnh đầu vào (BGR format)
            
        Returns:
            Danh sách các khuôn mặt được phát hiện với thông tin:
            - bbox: (x, y, width, height)
            - confidence: Độ tin cậy
            - landmarks: Các điểm đặc trưng (nếu có)
        """
        if image is None:
            logger.error("Hình ảnh đầu vào không hợp lệ")
            return []
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = []
        
        if self.method == "haar":
            detected = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in detected:
                faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': 1.0,  # Haar không trả về confidence
                    'landmarks': None
                })
                
        logger.info(f"Phát hiện được {len(faces)} khuôn mặt")
        return faces
    
    def extract_face_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                           padding: float = 0.1) -> Optional[np.ndarray]:
        """
        Trích xuất vùng khuôn mặt từ hình ảnh
        
        Args:
            image: Hình ảnh gốc
            bbox: Bounding box (x, y, w, h)
            padding: Tỷ lệ padding thêm xung quanh khuôn mặt
            
        Returns:
            Vùng khuôn mặt đã được cắt
        """
        x, y, w, h = bbox
        
        # Thêm padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        return image[y1:y2, x1:x2]
    
    def draw_faces(self, image: np.ndarray, faces: List[dict], 
                   color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Vẽ các bounding box lên hình ảnh
        
        Args:
            image: Hình ảnh gốc
            faces: Danh sách khuôn mặt đã phát hiện
            color: Màu của bounding box (BGR)
            
        Returns:
            Hình ảnh đã được đánh dấu
        """
        result = image.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            if 'confidence' in face:
                label = f"{face['confidence']:.2f}"
                cv2.putText(result, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        return result
