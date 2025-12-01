"""
Data Loader - Tải và quản lý dữ liệu
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Generator
import logging
import json

logger = logging.getLogger(__name__)


class DataLoader:
    """Lớp tải dữ liệu từ các nguồn khác nhau"""
    
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
    
    def __init__(self, data_dir: str = "data"):
        """
        Khởi tạo DataLoader
        
        Args:
            data_dir: Thư mục chứa dữ liệu
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Tải một hình ảnh
        
        Args:
            image_path: Đường dẫn đến hình ảnh
            
        Returns:
            Hình ảnh (BGR format) hoặc None nếu lỗi
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Không thể đọc hình ảnh: {image_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"Lỗi khi đọc hình ảnh {image_path}: {e}")
            return None
    
    def load_images_from_folder(self, folder_path: str, 
                                 limit: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
        """
        Tải tất cả hình ảnh từ thư mục
        
        Args:
            folder_path: Đường dẫn thư mục
            limit: Số lượng tối đa hình ảnh cần tải
            
        Returns:
            Danh sách (tên file, hình ảnh)
        """
        folder = Path(folder_path)
        if not folder.exists():
            logger.error(f"Thư mục không tồn tại: {folder_path}")
            return []
        
        images = []
        count = 0
        
        for file_path in sorted(folder.iterdir()):
            if file_path.suffix.lower() in self.SUPPORTED_IMAGE_FORMATS:
                image = self.load_image(str(file_path))
                if image is not None:
                    images.append((file_path.name, image))
                    count += 1
                    
                    if limit and count >= limit:
                        break
        
        logger.info(f"Đã tải {len(images)} hình ảnh từ {folder_path}")
        return images
    
    def load_labeled_dataset(self, dataset_path: str) -> Dict[str, List[np.ndarray]]:
        """
        Tải dataset có nhãn (mỗi thư mục con là một lớp)
        
        Args:
            dataset_path: Đường dẫn đến dataset
            
        Returns:
            Dictionary {label: [images]}
        """
        dataset = {}
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            logger.error(f"Dataset không tồn tại: {dataset_path}")
            return dataset
        
        for class_dir in dataset_dir.iterdir():
            if class_dir.is_dir():
                label = class_dir.name
                images = []
                
                for image_path in class_dir.iterdir():
                    if image_path.suffix.lower() in self.SUPPORTED_IMAGE_FORMATS:
                        image = self.load_image(str(image_path))
                        if image is not None:
                            images.append(image)
                
                if images:
                    dataset[label] = images
                    logger.info(f"Loaded {len(images)} images for class '{label}'")
        
        return dataset
    
    def load_annotations(self, annotation_file: str) -> Dict:
        """
        Tải file annotation (JSON format)
        
        Args:
            annotation_file: Đường dẫn file annotation
            
        Returns:
            Dictionary chứa annotations
        """
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            logger.info(f"Đã tải annotations từ {annotation_file}")
            return annotations
        except Exception as e:
            logger.error(f"Lỗi khi đọc file annotation: {e}")
            return {}
    
    def save_annotations(self, annotations: Dict, output_file: str):
        """
        Lưu annotations ra file JSON
        
        Args:
            annotations: Dictionary chứa annotations
            output_file: Đường dẫn output
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Đã lưu annotations vào {output_file}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu annotations: {e}")
    
    def image_generator(self, folder_path: str, 
                        batch_size: int = 32) -> Generator[List[np.ndarray], None, None]:
        """
        Generator để load hình ảnh theo batch
        
        Args:
            folder_path: Đường dẫn thư mục
            batch_size: Kích thước batch
            
        Yields:
            Batch hình ảnh
        """
        folder = Path(folder_path)
        image_files = [f for f in folder.iterdir() 
                      if f.suffix.lower() in self.SUPPORTED_IMAGE_FORMATS]
        
        batch = []
        for file_path in image_files:
            image = self.load_image(str(file_path))
            if image is not None:
                batch.append(image)
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        
        # Yield remaining images
        if batch:
            yield batch
    
    def split_dataset(self, dataset: Dict[str, List[np.ndarray]], 
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15) -> Tuple[Dict, Dict, Dict]:
        """
        Chia dataset thành train/val/test
        
        Args:
            dataset: Dictionary {label: [images]}
            train_ratio: Tỷ lệ train
            val_ratio: Tỷ lệ validation
            
        Returns:
            (train_set, val_set, test_set)
        """
        train_set = {}
        val_set = {}
        test_set = {}
        
        for label, images in dataset.items():
            n = len(images)
            np.random.shuffle(images)
            
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_set[label] = images[:train_end]
            val_set[label] = images[train_end:val_end]
            test_set[label] = images[val_end:]
            
            logger.info(f"Class '{label}': train={len(train_set[label])}, "
                       f"val={len(val_set[label])}, test={len(test_set[label])}")
        
        return train_set, val_set, test_set
