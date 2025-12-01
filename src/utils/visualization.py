"""
Visualization Utilities - Các tiện ích hiển thị
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """Lớp hỗ trợ visualization"""
    
    # Bảng màu cho các mức độ engagement
    ENGAGEMENT_COLORS = {
        'highly_engaged': (0, 255, 0),      # Xanh lá
        'engaged': (0, 200, 100),           # Xanh lá nhạt
        'neutral': (0, 255, 255),           # Vàng
        'disengaged': (0, 165, 255),        # Cam
        'highly_disengaged': (0, 0, 255)    # Đỏ
    }
    
    # Nhãn tiếng Việt
    ENGAGEMENT_LABELS = {
        'highly_engaged': 'Rất hứng thú',
        'engaged': 'Hứng thú',
        'neutral': 'Bình thường',
        'disengaged': 'Không hứng thú',
        'highly_disengaged': 'Rất không hứng thú'
    }
    
    @staticmethod
    def draw_face_bbox(image: np.ndarray,
                       bbox: Tuple[int, int, int, int],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2,
                       label: Optional[str] = None) -> np.ndarray:
        """
        Vẽ bounding box khuôn mặt
        
        Args:
            image: Hình ảnh
            bbox: (x, y, w, h)
            color: Màu BGR
            thickness: Độ dày đường
            label: Nhãn hiển thị
            
        Returns:
            Hình ảnh đã vẽ
        """
        result = image.copy()
        x, y, w, h = bbox
        
        # Vẽ rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        # Vẽ label nếu có
        if label:
            # Background cho text
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(result, (x, y - text_h - 10), 
                         (x + text_w + 10, y), color, -1)
            
            # Text
            cv2.putText(result, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return result
    
    @staticmethod
    def draw_engagement_overlay(image: np.ndarray,
                                bbox: Tuple[int, int, int, int],
                                engagement_level: str,
                                score: float) -> np.ndarray:
        """
        Vẽ overlay hiển thị mức độ engagement
        
        Args:
            image: Hình ảnh
            bbox: Bounding box khuôn mặt
            engagement_level: Mức độ engagement
            score: Điểm engagement
            
        Returns:
            Hình ảnh đã vẽ overlay
        """
        result = image.copy()
        x, y, w, h = bbox
        
        color = Visualizer.ENGAGEMENT_COLORS.get(engagement_level, (255, 255, 255))
        label = Visualizer.ENGAGEMENT_LABELS.get(engagement_level, engagement_level)
        
        # Vẽ bbox
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        # Vẽ engagement bar
        bar_width = w
        bar_height = 10
        bar_x = x
        bar_y = y + h + 5
        
        # Background
        cv2.rectangle(result, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # Fill dựa trên score
        fill_width = int(bar_width * score / 100)
        cv2.rectangle(result, (bar_x, bar_y),
                     (bar_x + fill_width, bar_y + bar_height),
                     color, -1)
        
        # Label
        label_text = f"{label}: {score:.0f}%"
        cv2.putText(result, label_text, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    @staticmethod
    def draw_class_stats(image: np.ndarray,
                         stats: Dict,
                         position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Vẽ thống kê lớp học lên hình ảnh
        
        Args:
            image: Hình ảnh
            stats: Dictionary chứa thống kê
            position: Vị trí bắt đầu (x, y)
            
        Returns:
            Hình ảnh đã vẽ
        """
        result = image.copy()
        x, y = position
        line_height = 25
        
        # Background panel
        panel_height = line_height * (len(stats) + 1)
        cv2.rectangle(result, (x - 5, y - 20),
                     (x + 250, y + panel_height),
                     (0, 0, 0), -1)
        cv2.rectangle(result, (x - 5, y - 20),
                     (x + 250, y + panel_height),
                     (255, 255, 255), 1)
        
        # Title
        cv2.putText(result, "THONG KE LOP HOC", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y += line_height
        
        # Stats
        for key, value in stats.items():
            if isinstance(value, float):
                text = f"{key}: {value:.1f}"
            else:
                text = f"{key}: {value}"
            
            cv2.putText(result, text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += line_height
        
        return result
    
    @staticmethod
    def create_engagement_chart(engagement_history: List[float],
                                width: int = 400,
                                height: int = 200) -> np.ndarray:
        """
        Tạo biểu đồ engagement theo thời gian
        
        Args:
            engagement_history: Lịch sử điểm engagement
            width: Chiều rộng biểu đồ
            height: Chiều cao biểu đồ
            
        Returns:
            Hình ảnh biểu đồ
        """
        chart = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        if len(engagement_history) < 2:
            return chart
        
        # Vẽ grid
        for i in range(0, height, height // 5):
            cv2.line(chart, (0, i), (width, i), (200, 200, 200), 1)
        
        # Vẽ đường engagement
        points = []
        n = len(engagement_history)
        for i, score in enumerate(engagement_history):
            x = int(i * width / max(n - 1, 1))
            y = int(height - (score / 100) * height)
            points.append((x, y))
        
        # Vẽ đường
        for i in range(len(points) - 1):
            # Chọn màu dựa trên giá trị
            score = engagement_history[i]
            if score >= 60:
                color = (0, 200, 0)
            elif score >= 40:
                color = (0, 200, 200)
            else:
                color = (0, 0, 200)
            
            cv2.line(chart, points[i], points[i + 1], color, 2)
        
        # Vẽ điểm
        for point in points:
            cv2.circle(chart, point, 3, (0, 0, 0), -1)
        
        # Labels
        cv2.putText(chart, "100%", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(chart, "0%", (5, height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return chart
    
    @staticmethod
    def create_emotion_pie_chart(emotion_counts: Dict[str, int],
                                 size: int = 200) -> np.ndarray:
        """
        Tạo biểu đồ tròn phân bố cảm xúc
        
        Args:
            emotion_counts: Dictionary {emotion: count}
            size: Kích thước biểu đồ
            
        Returns:
            Hình ảnh biểu đồ
        """
        chart = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        total = sum(emotion_counts.values())
        if total == 0:
            return chart
        
        # Màu cho các cảm xúc
        colors = {
            'happy': (0, 255, 0),
            'surprised': (0, 255, 255),
            'neutral': (128, 128, 128),
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'fear': (128, 0, 128),
            'disgust': (0, 128, 0),
            'confused': (255, 165, 0)
        }
        
        center = (size // 2, size // 2)
        radius = size // 2 - 20
        
        start_angle = 0
        for emotion, count in emotion_counts.items():
            if count == 0:
                continue
            
            angle = int(360 * count / total)
            end_angle = start_angle + angle
            
            color = colors.get(emotion, (200, 200, 200))
            cv2.ellipse(chart, center, (radius, radius),
                       0, start_angle, end_angle, color, -1)
            
            start_angle = end_angle
        
        # Vẽ viền
        cv2.circle(chart, center, radius, (0, 0, 0), 2)
        
        return chart
