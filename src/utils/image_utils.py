"""
Image Processing Utilities - Các tiện ích xử lý ảnh
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Lớp xử lý hình ảnh"""
    
    @staticmethod
    def resize(image: np.ndarray, size: Tuple[int, int], 
               keep_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize hình ảnh
        
        Args:
            image: Hình ảnh đầu vào
            size: Kích thước mới (width, height)
            keep_aspect_ratio: Giữ tỷ lệ khung hình
            
        Returns:
            Hình ảnh đã resize
        """
        if not keep_aspect_ratio:
            return cv2.resize(image, size)
        
        h, w = image.shape[:2]
        target_w, target_h = size
        
        # Tính tỷ lệ
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Padding nếu cần
        if new_w != target_w or new_h != target_h:
            result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            return result
        
        return resized
    
    @staticmethod
    def normalize(image: np.ndarray, 
                  mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                  std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
        """
        Normalize hình ảnh theo ImageNet standards
        
        Args:
            image: Hình ảnh (0-255)
            mean: Giá trị mean
            std: Giá trị std
            
        Returns:
            Hình ảnh đã normalize
        """
        image = image.astype(np.float32) / 255.0
        
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        
        return image
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Chuyển sang ảnh grayscale"""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def equalize_histogram(image: np.ndarray) -> np.ndarray:
        """
        Cân bằng histogram để cải thiện độ tương phản
        """
        if len(image.shape) == 3:
            # Chuyển sang YUV và equalize kênh Y
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return cv2.equalizeHist(image)
    
    @staticmethod
    def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Khử nhiễu hình ảnh
        
        Args:
            image: Hình ảnh đầu vào
            strength: Độ mạnh của khử nhiễu
            
        Returns:
            Hình ảnh đã khử nhiễu
        """
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        Điều chỉnh độ sáng
        
        Args:
            image: Hình ảnh đầu vào
            factor: Hệ số điều chỉnh (>1 sáng hơn, <1 tối hơn)
            
        Returns:
            Hình ảnh đã điều chỉnh
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def crop_face(image: np.ndarray, bbox: Tuple[int, int, int, int],
                  margin: float = 0.2) -> np.ndarray:
        """
        Cắt vùng khuôn mặt với margin
        
        Args:
            image: Hình ảnh gốc
            bbox: Bounding box (x, y, w, h)
            margin: Tỷ lệ margin thêm
            
        Returns:
            Vùng khuôn mặt đã cắt
        """
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]
        
        margin_w = int(w * margin)
        margin_h = int(h * margin)
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(w_img, x + w + margin_w)
        y2 = min(h_img, y + h + margin_h)
        
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def augment(image: np.ndarray, 
                flip: bool = False,
                rotate: Optional[float] = None,
                brightness: Optional[float] = None) -> np.ndarray:
        """
        Data augmentation cho hình ảnh
        
        Args:
            image: Hình ảnh đầu vào
            flip: Lật ngang
            rotate: Góc xoay (độ)
            brightness: Hệ số điều chỉnh độ sáng
            
        Returns:
            Hình ảnh đã augment
        """
        result = image.copy()
        
        if flip:
            result = cv2.flip(result, 1)
        
        if rotate is not None:
            h, w = result.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, rotate, 1.0)
            result = cv2.warpAffine(result, matrix, (w, h))
        
        if brightness is not None:
            result = ImageProcessor.adjust_brightness(result, brightness)
        
        return result
