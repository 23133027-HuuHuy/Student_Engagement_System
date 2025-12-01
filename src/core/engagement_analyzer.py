"""
Engagement Analyzer Module - Mô-đun phân tích mức độ hứng thú
Phân tích và đánh giá mức độ tập trung, hứng thú của sinh viên
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging

from .emotion_classifier import Emotion

logger = logging.getLogger(__name__)


class EngagementLevel(Enum):
    """Các mức độ hứng thú học tập"""
    HIGHLY_ENGAGED = "highly_engaged"     # Rất hứng thú
    ENGAGED = "engaged"                    # Hứng thú
    NEUTRAL = "neutral"                    # Bình thường
    DISENGAGED = "disengaged"             # Không hứng thú
    HIGHLY_DISENGAGED = "highly_disengaged"  # Rất không hứng thú


@dataclass
class EngagementMetrics:
    """Các chỉ số đánh giá mức độ hứng thú"""
    engagement_level: EngagementLevel
    engagement_score: float  # 0-100
    attention_score: float   # 0-100
    emotion_score: float     # 0-100
    head_pose_score: float   # 0-100 (nếu có)
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Chuyển đổi sang dictionary"""
        return {
            'engagement_level': self.engagement_level.value,
            'engagement_level_label': self.get_level_label(),
            'engagement_score': round(self.engagement_score, 2),
            'attention_score': round(self.attention_score, 2),
            'emotion_score': round(self.emotion_score, 2),
            'head_pose_score': round(self.head_pose_score, 2),
            'timestamp': self.timestamp.isoformat()
        }
    
    def get_level_label(self) -> str:
        """Lấy nhãn tiếng Việt cho mức độ hứng thú"""
        labels = {
            EngagementLevel.HIGHLY_ENGAGED: "Rất hứng thú",
            EngagementLevel.ENGAGED: "Hứng thú",
            EngagementLevel.NEUTRAL: "Bình thường",
            EngagementLevel.DISENGAGED: "Không hứng thú",
            EngagementLevel.HIGHLY_DISENGAGED: "Rất không hứng thú"
        }
        return labels.get(self.engagement_level, "Không xác định")


class EngagementAnalyzer:
    """
    Lớp phân tích mức độ hứng thú học tập của sinh viên
    
    Kết hợp nhiều yếu tố:
    - Cảm xúc khuôn mặt
    - Hướng nhìn (head pose)
    - Tần suất chớp mắt
    - Các đặc điểm khác
    """
    
    # Ánh xạ cảm xúc sang điểm hứng thú
    EMOTION_ENGAGEMENT_MAP = {
        Emotion.HAPPY: 90,
        Emotion.SURPRISED: 75,
        Emotion.NEUTRAL: 50,
        Emotion.CONFUSED: 40,
        Emotion.SAD: 30,
        Emotion.FEAR: 25,
        Emotion.ANGRY: 20,
        Emotion.DISGUST: 15
    }
    
    def __init__(self, 
                 emotion_weight: float = 0.4,
                 attention_weight: float = 0.4,
                 head_pose_weight: float = 0.2):
        """
        Khởi tạo Engagement Analyzer
        
        Args:
            emotion_weight: Trọng số cho cảm xúc (0-1)
            attention_weight: Trọng số cho sự chú ý (0-1)
            head_pose_weight: Trọng số cho hướng nhìn (0-1)
        """
        self.emotion_weight = emotion_weight
        self.attention_weight = attention_weight
        self.head_pose_weight = head_pose_weight
        
        # Lịch sử phân tích để tracking
        self.history: List[EngagementMetrics] = []
        
        logger.info("Đã khởi tạo Engagement Analyzer")
    
    def analyze(self, 
                emotion_result: Dict,
                attention_data: Optional[Dict] = None,
                head_pose_data: Optional[Dict] = None) -> EngagementMetrics:
        """
        Phân tích mức độ hứng thú từ các dữ liệu đầu vào
        
        Args:
            emotion_result: Kết quả phân loại cảm xúc
            attention_data: Dữ liệu về sự chú ý (eye tracking, etc.)
            head_pose_data: Dữ liệu về hướng nhìn
            
        Returns:
            EngagementMetrics chứa các chỉ số đánh giá
        """
        # Tính điểm cảm xúc
        emotion = emotion_result.get('emotion', Emotion.NEUTRAL)
        emotion_score = self.EMOTION_ENGAGEMENT_MAP.get(emotion, 50)
        
        # Tính điểm chú ý (placeholder)
        if attention_data:
            attention_score = attention_data.get('score', 50)
        else:
            attention_score = 50  # Mặc định
        
        # Tính điểm hướng nhìn (placeholder)
        if head_pose_data:
            head_pose_score = self._calculate_head_pose_score(head_pose_data)
        else:
            head_pose_score = 50  # Mặc định
        
        # Tính điểm tổng hợp
        engagement_score = (
            emotion_score * self.emotion_weight +
            attention_score * self.attention_weight +
            head_pose_score * self.head_pose_weight
        )
        
        # Xác định mức độ hứng thú
        engagement_level = self._score_to_level(engagement_score)
        
        # Tạo metrics
        metrics = EngagementMetrics(
            engagement_level=engagement_level,
            engagement_score=engagement_score,
            attention_score=attention_score,
            emotion_score=emotion_score,
            head_pose_score=head_pose_score,
            timestamp=datetime.now()
        )
        
        # Lưu vào lịch sử
        self.history.append(metrics)
        
        return metrics
    
    def _calculate_head_pose_score(self, head_pose_data: Dict) -> float:
        """
        Tính điểm dựa trên hướng nhìn
        
        Sinh viên nhìn thẳng về phía trước (bảng/giảng viên) = điểm cao
        Nhìn xuống, sang bên = điểm thấp hơn
        """
        yaw = head_pose_data.get('yaw', 0)    # Xoay trái/phải
        pitch = head_pose_data.get('pitch', 0)  # Ngẩng/cúi
        
        # Tính độ lệch từ vị trí nhìn thẳng
        yaw_deviation = abs(yaw)
        pitch_deviation = abs(pitch)
        
        # Điểm cao khi lệch ít
        max_deviation = 45  # độ
        yaw_score = max(0, 100 - (yaw_deviation / max_deviation) * 100)
        pitch_score = max(0, 100 - (pitch_deviation / max_deviation) * 100)
        
        return (yaw_score + pitch_score) / 2
    
    def _score_to_level(self, score: float) -> EngagementLevel:
        """Chuyển đổi điểm sang mức độ hứng thú"""
        if score >= 80:
            return EngagementLevel.HIGHLY_ENGAGED
        elif score >= 60:
            return EngagementLevel.ENGAGED
        elif score >= 40:
            return EngagementLevel.NEUTRAL
        elif score >= 20:
            return EngagementLevel.DISENGAGED
        else:
            return EngagementLevel.HIGHLY_DISENGAGED
    
    def get_level_color(self, level: EngagementLevel) -> Tuple[int, int, int]:
        """Lấy màu tương ứng với mức độ hứng thú (BGR)"""
        color_map = {
            EngagementLevel.HIGHLY_ENGAGED: (0, 255, 0),      # Xanh lá
            EngagementLevel.ENGAGED: (0, 200, 100),           # Xanh lá nhạt
            EngagementLevel.NEUTRAL: (0, 255, 255),           # Vàng
            EngagementLevel.DISENGAGED: (0, 165, 255),        # Cam
            EngagementLevel.HIGHLY_DISENGAGED: (0, 0, 255)    # Đỏ
        }
        return color_map.get(level, (255, 255, 255))
    
    def get_statistics(self) -> Dict:
        """
        Lấy thống kê từ lịch sử phân tích
        
        Returns:
            Dictionary chứa các thống kê
        """
        if not self.history:
            return {
                'total_samples': 0,
                'average_score': 0,
                'level_distribution': {}
            }
        
        scores = [m.engagement_score for m in self.history]
        
        # Đếm phân bố các mức độ
        level_counts = {}
        for level in EngagementLevel:
            count = sum(1 for m in self.history if m.engagement_level == level)
            level_counts[level.value] = count
        
        return {
            'total_samples': len(self.history),
            'average_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'std_score': np.std(scores),
            'level_distribution': level_counts
        }
    
    def get_realtime_status(self, window_size: int = 10) -> Dict:
        """
        Lấy trạng thái real-time (dựa trên các mẫu gần nhất)
        
        Args:
            window_size: Số mẫu gần nhất để tính toán
            
        Returns:
            Trạng thái hiện tại
        """
        if not self.history:
            return {
                'current_level': EngagementLevel.NEUTRAL.value,
                'current_score': 50,
                'trend': 'stable'
            }
        
        recent = self.history[-window_size:]
        recent_scores = [m.engagement_score for m in recent]
        
        current_score = np.mean(recent_scores)
        current_level = self._score_to_level(current_score)
        
        # Xác định xu hướng
        if len(recent_scores) >= 3:
            first_half = np.mean(recent_scores[:len(recent_scores)//2])
            second_half = np.mean(recent_scores[len(recent_scores)//2:])
            
            if second_half - first_half > 5:
                trend = 'increasing'
            elif first_half - second_half > 5:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'current_level': current_level.value,
            'current_level_label': EngagementMetrics(
                current_level, current_score, 0, 0, 0, datetime.now()
            ).get_level_label(),
            'current_score': round(current_score, 2),
            'trend': trend
        }
    
    def clear_history(self):
        """Xóa lịch sử phân tích"""
        self.history.clear()
        logger.info("Đã xóa lịch sử phân tích")
