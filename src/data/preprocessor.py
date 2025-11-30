"""
Data Preprocessor - Tiền xử lý dữ liệu
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Lớp tiền xử lý dữ liệu cho training và inference"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (48, 48),
                 grayscale: bool = True,
                 normalize: bool = True):
        """
        Khởi tạo preprocessor
        
        Args:
            target_size: Kích thước đầu ra (width, height)
            grayscale: Chuyển sang grayscale
            normalize: Normalize về [0, 1]
        """
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý một hình ảnh
        
        Args:
            image: Hình ảnh đầu vào (BGR)
            
        Returns:
            Hình ảnh đã tiền xử lý
        """
        # Chuyển grayscale nếu cần
        if self.grayscale and len(image.shape) == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            processed = image.copy()
        
        # Resize
        processed = cv2.resize(processed, self.target_size)
        
        # Normalize
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0
        
        # Thêm channel dimension nếu grayscale
        if self.grayscale and len(processed.shape) == 2:
            processed = np.expand_dims(processed, axis=-1)
        
        return processed
    
    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Tiền xử lý batch hình ảnh
        
        Args:
            images: Danh sách hình ảnh
            
        Returns:
            Numpy array shape (N, H, W, C)
        """
        processed = [self.preprocess_image(img) for img in images]
        return np.array(processed)
    
    def augment_image(self, image: np.ndarray, 
                      flip_prob: float = 0.5,
                      rotate_range: Tuple[int, int] = (-10, 10),
                      brightness_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Data augmentation cho một hình ảnh
        
        Args:
            image: Hình ảnh đầu vào
            flip_prob: Xác suất lật ngang
            rotate_range: Phạm vi góc xoay (min, max)
            brightness_range: Phạm vi điều chỉnh độ sáng
            
        Returns:
            Hình ảnh đã augment
        """
        result = image.copy()
        
        # Random horizontal flip
        if np.random.random() < flip_prob:
            result = cv2.flip(result, 1)
        
        # Random rotation
        angle = np.random.uniform(rotate_range[0], rotate_range[1])
        h, w = result.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(result, matrix, (w, h), 
                                borderMode=cv2.BORDER_REPLICATE)
        
        # Random brightness
        if len(result.shape) == 3:
            factor = np.random.uniform(brightness_range[0], brightness_range[1])
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        else:
            factor = np.random.uniform(brightness_range[0], brightness_range[1])
            result = np.clip(result * factor, 0, 255).astype(np.uint8)
        
        return result
    
    def create_augmented_dataset(self, images: List[np.ndarray], 
                                  labels: List[int],
                                  augment_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo dataset đã augment
        
        Args:
            images: Danh sách hình ảnh gốc
            labels: Nhãn tương ứng
            augment_factor: Số lần augment mỗi ảnh
            
        Returns:
            (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = []
        
        for image, label in zip(images, labels):
            # Original image
            augmented_images.append(self.preprocess_image(image))
            augmented_labels.append(label)
            
            # Augmented versions
            for _ in range(augment_factor):
                aug_image = self.augment_image(image)
                augmented_images.append(self.preprocess_image(aug_image))
                augmented_labels.append(label)
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def balance_classes(self, images: List[np.ndarray], 
                        labels: List[int]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Cân bằng các lớp bằng oversampling
        
        Args:
            images: Danh sách hình ảnh
            labels: Nhãn tương ứng
            
        Returns:
            (balanced_images, balanced_labels)
        """
        class_counts = Counter(labels)
        max_count = max(class_counts.values())
        
        balanced_images = []
        balanced_labels = []
        
        # Group by class
        class_images = {}
        for img, label in zip(images, labels):
            if label not in class_images:
                class_images[label] = []
            class_images[label].append(img)
        
        # Oversample each class
        for label, class_imgs in class_images.items():
            n = len(class_imgs)
            indices = np.random.choice(n, max_count, replace=True)
            
            for idx in indices:
                balanced_images.append(class_imgs[idx])
                balanced_labels.append(label)
        
        # Shuffle
        combined = list(zip(balanced_images, balanced_labels))
        np.random.shuffle(combined)
        balanced_images, balanced_labels = zip(*combined)
        
        return list(balanced_images), list(balanced_labels)
