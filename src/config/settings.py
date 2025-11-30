"""
Settings - Cấu hình ứng dụng
"""

from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
import os


@dataclass
class ModelSettings:
    """Cấu hình mô hình"""
    # Emotion model
    emotion_model_path: str = "data/models/emotion_model.h5"
    emotion_input_size: tuple = (48, 48)
    emotion_num_classes: int = 8
    
    # Engagement model
    engagement_model_path: str = "data/models/engagement_model.h5"
    engagement_use_temporal: bool = True
    engagement_sequence_length: int = 10


@dataclass
class DetectionSettings:
    """Cấu hình phát hiện khuôn mặt"""
    method: str = "haar"  # haar, dnn, mtcnn
    confidence_threshold: float = 0.5
    min_face_size: int = 30
    scale_factor: float = 1.1


@dataclass
class EngagementSettings:
    """Cấu hình đánh giá engagement"""
    emotion_weight: float = 0.4
    attention_weight: float = 0.4
    head_pose_weight: float = 0.2
    
    # Ngưỡng các mức độ
    highly_engaged_threshold: float = 80
    engaged_threshold: float = 60
    neutral_threshold: float = 40
    disengaged_threshold: float = 20


@dataclass
class VideoSettings:
    """Cấu hình xử lý video"""
    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    skip_frames: int = 0


@dataclass
class Settings:
    """Cấu hình tổng hợp"""
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    
    # Sub-settings
    model: ModelSettings = field(default_factory=ModelSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    engagement: EngagementSettings = field(default_factory=EngagementSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    
    # General
    debug: bool = False
    log_level: str = "INFO"
    language: str = "vi"
    
    def __post_init__(self):
        """Khởi tạo các thư mục"""
        self.data_dir = self.base_dir / self.data_dir
        self.output_dir = self.base_dir / self.output_dir
        self.log_dir = self.base_dir / self.log_dir
    
    def ensure_directories(self):
        """Tạo các thư mục cần thiết"""
        dirs = [
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "models",
            self.output_dir,
            self.log_dir
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary"""
        return {
            'base_dir': str(self.base_dir),
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'log_dir': str(self.log_dir),
            'model': {
                'emotion_model_path': self.model.emotion_model_path,
                'emotion_input_size': self.model.emotion_input_size,
                'emotion_num_classes': self.model.emotion_num_classes,
                'engagement_model_path': self.model.engagement_model_path,
                'engagement_use_temporal': self.model.engagement_use_temporal,
                'engagement_sequence_length': self.model.engagement_sequence_length
            },
            'detection': {
                'method': self.detection.method,
                'confidence_threshold': self.detection.confidence_threshold,
                'min_face_size': self.detection.min_face_size,
                'scale_factor': self.detection.scale_factor
            },
            'engagement': {
                'emotion_weight': self.engagement.emotion_weight,
                'attention_weight': self.engagement.attention_weight,
                'head_pose_weight': self.engagement.head_pose_weight
            },
            'video': {
                'camera_id': self.video.camera_id,
                'frame_width': self.video.frame_width,
                'frame_height': self.video.frame_height,
                'fps': self.video.fps
            },
            'debug': self.debug,
            'log_level': self.log_level,
            'language': self.language
        }
    
    def save(self, filepath: str):
        """Lưu cấu hình ra file JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'Settings':
        """Tải cấu hình từ file JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        settings = cls()
        
        if 'detection' in data:
            settings.detection = DetectionSettings(**data['detection'])
        if 'engagement' in data:
            settings.engagement = EngagementSettings(**data['engagement'])
        if 'video' in data:
            settings.video = VideoSettings(**data['video'])
        if 'debug' in data:
            settings.debug = data['debug']
        if 'log_level' in data:
            settings.log_level = data['log_level']
        
        return settings


# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Lấy settings instance (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings(config_path: Optional[str] = None) -> Settings:
    """Reload settings từ file"""
    global _settings
    
    if config_path and os.path.exists(config_path):
        _settings = Settings.load(config_path)
    else:
        _settings = Settings()
    
    return _settings
