"""
Video Processing Utilities - Các tiện ích xử lý video
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional, Callable
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Lớp xử lý video"""
    
    def __init__(self, source: int | str = 0):
        """
        Khởi tạo Video Processor
        
        Args:
            source: Nguồn video (0 cho webcam, hoặc đường dẫn file)
        """
        self.source = source
        self.cap = None
        self.is_opened = False
        self.fps = 30
        self.frame_width = 640
        self.frame_height = 480
    
    def open(self) -> bool:
        """
        Mở nguồn video
        
        Returns:
            True nếu mở thành công
        """
        self.cap = cv2.VideoCapture(self.source)
        
        if self.cap.isOpened():
            self.is_opened = True
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Đã mở video source: {self.source}")
            logger.info(f"FPS: {self.fps}, Resolution: {self.frame_width}x{self.frame_height}")
            return True
        else:
            logger.error(f"Không thể mở video source: {self.source}")
            return False
    
    def close(self):
        """Đóng nguồn video"""
        if self.cap:
            self.cap.release()
            self.is_opened = False
            logger.info("Đã đóng video source")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Đọc một frame
        
        Returns:
            Tuple (success, frame)
        """
        if not self.is_opened:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def get_frames(self, max_frames: Optional[int] = None, 
                   skip_frames: int = 0) -> Generator[np.ndarray, None, None]:
        """
        Generator để đọc các frame
        
        Args:
            max_frames: Số frame tối đa cần đọc
            skip_frames: Số frame bỏ qua giữa các lần đọc
            
        Yields:
            Các frame video
        """
        if not self.is_opened:
            if not self.open():
                return
        
        frame_count = 0
        skip_count = 0
        
        while True:
            ret, frame = self.read_frame()
            
            if not ret:
                break
            
            if skip_count < skip_frames:
                skip_count += 1
                continue
            
            skip_count = 0
            frame_count += 1
            
            yield frame
            
            if max_frames and frame_count >= max_frames:
                break
    
    def process_video(self, 
                      process_func: Callable[[np.ndarray], np.ndarray],
                      output_path: Optional[str] = None,
                      display: bool = True,
                      window_name: str = "Video") -> None:
        """
        Xử lý video với một hàm custom
        
        Args:
            process_func: Hàm xử lý mỗi frame
            output_path: Đường dẫn output (nếu muốn lưu)
            display: Hiển thị video
            window_name: Tên cửa sổ hiển thị
        """
        if not self.is_opened:
            if not self.open():
                return
        
        # Setup video writer nếu cần
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path, fourcc, self.fps,
                (self.frame_width, self.frame_height)
            )
        
        try:
            for frame in self.get_frames():
                # Xử lý frame
                processed = process_func(frame)
                
                # Lưu frame
                if writer:
                    writer.write(processed)
                
                # Hiển thị
                if display:
                    cv2.imshow(window_name, processed)
                    
                    # Nhấn 'q' để thoát
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            logger.info("Hoàn thành xử lý video")
    
    def capture_frames(self, 
                       output_dir: str,
                       interval: int = 30,
                       prefix: str = "frame") -> int:
        """
        Capture các frame từ video và lưu thành ảnh
        
        Args:
            output_dir: Thư mục lưu ảnh
            interval: Khoảng cách giữa các frame
            prefix: Tiền tố tên file
            
        Returns:
            Số frame đã capture
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for i, frame in enumerate(self.get_frames()):
            if i % interval == 0:
                filename = output_path / f"{prefix}_{count:04d}.jpg"
                cv2.imwrite(str(filename), frame)
                count += 1
                logger.debug(f"Đã lưu {filename}")
        
        logger.info(f"Đã capture {count} frames vào {output_dir}")
        return count
    
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        Lấy thông tin video
        
        Args:
            video_path: Đường dẫn video
            
        Returns:
            Dictionary chứa thông tin
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {}
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
