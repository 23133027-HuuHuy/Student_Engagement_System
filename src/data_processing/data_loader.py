"""
Data Loader - Tải và xử lý dữ liệu
Xử lý dữ liệu ảnh và video cho training và inference
"""

from typing import Tuple, List, Generator, Optional
from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (BGR format) or None if failed
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    return image


def load_images_from_folder(folder_path: str, 
                            extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
                           ) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Load all images from a folder
    
    Args:
        folder_path: Path to folder containing images
        extensions: Tuple of allowed file extensions
        
    Yields:
        Tuple of (filename, image)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    for file_path in folder.iterdir():
        if file_path.suffix.lower() in extensions:
            image = cv2.imread(str(file_path))
            if image is not None:
                yield file_path.name, image


def video_frame_generator(video_source, 
                          skip_frames: int = 0) -> Generator[np.ndarray, None, None]:
    """
    Generate frames from video source
    
    Args:
        video_source: Video file path or camera ID (int)
        skip_frames: Number of frames to skip between yields
        
    Yields:
        Video frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {video_source}")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if skip_frames == 0 or frame_count % (skip_frames + 1) == 0:
                yield frame
            
            frame_count += 1
    finally:
        cap.release()


def resize_image(image: np.ndarray, 
                 target_size: Tuple[int, int],
                 keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create output image with padding
        output = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Place resized image in center
        output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return output
    else:
        return cv2.resize(image, target_size)


def crop_face(image: np.ndarray, 
              bbox: Tuple[int, int, int, int],
              padding: float = 0.2) -> np.ndarray:
    """
    Crop face from image with optional padding
    
    Args:
        image: Input image
        bbox: Face bounding box (x, y, w, h)
        padding: Padding ratio to add around face
        
    Returns:
        Cropped face image
    """
    x, y, w, h = bbox
    
    # Calculate padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    # Calculate new coordinates with padding
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(image.shape[1], x + w + pad_w)
    y2 = min(image.shape[0], y + h + pad_h)
    
    return image[y1:y2, x1:x2]


def split_dataset(data_folder: str, 
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  random_seed: Optional[int] = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split dataset into train, validation, and test sets
    
    Args:
        data_folder: Path to data folder
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
    
    folder = Path(data_folder)
    all_files = list(folder.glob("*"))
    
    # Shuffle files with reproducible seed
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(all_files)
    
    # Calculate split indices
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = [str(f) for f in all_files[:n_train]]
    val_files = [str(f) for f in all_files[n_train:n_train + n_val]]
    test_files = [str(f) for f in all_files[n_train + n_val:]]
    
    return train_files, val_files, test_files
