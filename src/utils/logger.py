"""
Logger Configuration - Cấu hình logging
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "student_engagement",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Thiết lập logger
    
    Args:
        name: Tên logger
        level: Mức độ log (DEBUG, INFO, WARNING, ERROR)
        log_file: Đường dẫn file log (nếu muốn lưu file)
        console: Có in ra console không
        
    Returns:
        Logger đã được cấu hình
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Xóa handlers cũ
    logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "student_engagement") -> logging.Logger:
    """
    Lấy logger đã tồn tại hoặc tạo mới
    
    Args:
        name: Tên logger
        
    Returns:
        Logger
    """
    return logging.getLogger(name)
