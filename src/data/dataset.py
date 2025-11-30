"""
Engagement Dataset - Dataset class cho PyTorch/TensorFlow
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from pathlib import Path
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class EngagementDataset:
    """
    Dataset class cho Student Engagement System
    
    Có thể được sử dụng với PyTorch DataLoader hoặc TensorFlow Dataset
    """
    
    ENGAGEMENT_CLASSES = [
        'highly_engaged',
        'engaged',
        'neutral',
        'disengaged',
        'highly_disengaged'
    ]
    
    EMOTION_CLASSES = [
        'angry', 'disgust', 'fear', 'happy',
        'sad', 'surprised', 'neutral', 'confused'
    ]
    
    def __init__(self, 
                 images: List[np.ndarray],
                 labels: List[int],
                 transform: Optional[Callable] = None,
                 mode: str = 'emotion'):
        """
        Khởi tạo dataset
        
        Args:
            images: Danh sách hình ảnh (đã preprocessed)
            labels: Nhãn tương ứng
            transform: Hàm transform (data augmentation)
            mode: 'emotion' hoặc 'engagement'
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.mode = mode
        
        self.classes = (self.EMOTION_CLASSES if mode == 'emotion' 
                       else self.ENGAGEMENT_CLASSES)
        self.num_classes = len(self.classes)
        
        logger.info(f"Dataset initialized with {len(self.images)} samples, "
                   f"{self.num_classes} classes ({mode} mode)")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Lấy một sample
        
        Args:
            idx: Index
            
        Returns:
            (image, label)
        """
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_batch(self, batch_size: int, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lấy một batch
        
        Args:
            batch_size: Kích thước batch
            shuffle: Có shuffle không
            
        Returns:
            (batch_images, batch_labels)
        """
        n = len(self.images)
        indices = np.arange(n)
        
        if shuffle:
            np.random.shuffle(indices)
        
        batch_indices = indices[:batch_size]
        
        batch_images = [self.images[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]
        
        return np.array(batch_images), np.array(batch_labels)
    
    def get_generator(self, batch_size: int, shuffle: bool = True):
        """
        Generator cho training
        
        Args:
            batch_size: Kích thước batch
            shuffle: Có shuffle mỗi epoch không
            
        Yields:
            (batch_images, batch_labels)
        """
        n = len(self.images)
        indices = np.arange(n)
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_indices = indices[start:end]
                
                batch_images = np.array([self.images[i] for i in batch_indices])
                batch_labels = np.array([self.labels[i] for i in batch_indices])
                
                yield batch_images, batch_labels
    
    def to_one_hot(self, labels: np.ndarray) -> np.ndarray:
        """
        Chuyển labels sang one-hot encoding
        
        Args:
            labels: Labels (integers)
            
        Returns:
            One-hot encoded labels
        """
        n = len(labels)
        one_hot = np.zeros((n, self.num_classes))
        one_hot[np.arange(n), labels] = 1
        return one_hot
    
    def get_class_weights(self) -> Dict[int, float]:
        """
        Tính class weights cho imbalanced dataset
        
        Returns:
            Dictionary {class_id: weight}
        """
        counts = Counter(self.labels)
        total = len(self.labels)
        
        weights = {}
        for cls in range(self.num_classes):
            if counts[cls] > 0:
                weights[cls] = total / (self.num_classes * counts[cls])
            else:
                weights[cls] = 1.0
        
        return weights
    
    def get_statistics(self) -> Dict:
        """
        Lấy thống kê dataset
        
        Returns:
            Dictionary chứa thống kê
        """
        counts = Counter(self.labels)
        
        stats = {
            'total_samples': len(self.images),
            'num_classes': self.num_classes,
            'class_distribution': {},
            'class_names': self.classes
        }
        
        for cls_id, cls_name in enumerate(self.classes):
            count = counts.get(cls_id, 0)
            percentage = (count / len(self.images)) * 100 if self.images else 0
            stats['class_distribution'][cls_name] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        return stats
    
    @classmethod
    def from_folder(cls, folder_path: str, 
                    mode: str = 'emotion',
                    target_size: Tuple[int, int] = (48, 48)):
        """
        Tạo dataset từ thư mục (mỗi subfolder là một class)
        
        Args:
            folder_path: Đường dẫn thư mục
            mode: 'emotion' hoặc 'engagement'
            target_size: Kích thước resize
            
        Returns:
            EngagementDataset instance
        """
        import cv2
        
        folder = Path(folder_path)
        classes = (cls.EMOTION_CLASSES if mode == 'emotion' 
                  else cls.ENGAGEMENT_CLASSES)
        
        images = []
        labels = []
        
        for cls_idx, cls_name in enumerate(classes):
            cls_folder = folder / cls_name
            if not cls_folder.exists():
                logger.warning(f"Folder not found: {cls_folder}")
                continue
            
            for img_path in cls_folder.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(img, target_size)
                        img = img.astype(np.float32) / 255.0
                        img = np.expand_dims(img, axis=-1)
                        
                        images.append(img)
                        labels.append(cls_idx)
        
        return cls(images, labels, mode=mode)
