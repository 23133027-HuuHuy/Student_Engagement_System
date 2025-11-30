"""
Student Engagement System - Main Application
Hệ thống phân loại mức độ hứng thú học tập của sinh viên

Chạy: python main.py [options]
"""

import argparse
import cv2
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import FaceDetector, EmotionClassifier, EngagementAnalyzer
from src.utils import setup_logger, VideoProcessor, Visualizer
from src.config import get_settings


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Student Engagement System - Phân loại mức độ hứng thú học tập'
    )
    
    parser.add_argument(
        '--mode', 
        choices=['camera', 'video', 'image'],
        default='camera',
        help='Chế độ xử lý: camera (webcam), video (file video), image (ảnh)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Nguồn video: 0 cho webcam, hoặc đường dẫn file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Đường dẫn lưu output video'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Bật chế độ debug'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Không hiển thị video (chỉ lưu output)'
    )
    
    return parser.parse_args()


def process_frame(frame, face_detector, emotion_classifier, engagement_analyzer):
    """
    Xử lý một frame video
    
    Args:
        frame: Frame hình ảnh
        face_detector: FaceDetector instance
        emotion_classifier: EmotionClassifier instance
        engagement_analyzer: EngagementAnalyzer instance
        
    Returns:
        Frame đã được xử lý với annotations
    """
    result = frame.copy()
    
    # Phát hiện khuôn mặt
    faces = face_detector.detect_faces(frame)
    
    for face in faces:
        bbox = face['bbox']
        
        # Trích xuất vùng khuôn mặt
        face_region = face_detector.extract_face_region(frame, bbox)
        
        if face_region is not None and face_region.size > 0:
            # Phân loại cảm xúc
            emotion_result = emotion_classifier.classify(face_region)
            
            # Phân tích engagement
            metrics = engagement_analyzer.analyze(emotion_result)
            
            # Vẽ overlay
            result = Visualizer.draw_engagement_overlay(
                result, bbox,
                metrics.engagement_level.value,
                metrics.engagement_score
            )
    
    # Vẽ thống kê
    realtime_status = engagement_analyzer.get_realtime_status()
    stats = {
        'Số sinh viên': len(faces),
        'Điểm TB': realtime_status['current_score'],
        'Mức độ': realtime_status.get('current_level_label', 'N/A'),
        'Xu hướng': realtime_status['trend']
    }
    result = Visualizer.draw_class_stats(result, stats)
    
    return result


def run_camera_mode(args):
    """Chạy với webcam"""
    logger = logging.getLogger(__name__)
    settings = get_settings()
    
    # Khởi tạo các thành phần
    face_detector = FaceDetector(method=settings.detection.method)
    emotion_classifier = EmotionClassifier()
    engagement_analyzer = EngagementAnalyzer(
        emotion_weight=settings.engagement.emotion_weight,
        attention_weight=settings.engagement.attention_weight,
        head_pose_weight=settings.engagement.head_pose_weight
    )
    
    # Mở camera
    source = int(args.source) if args.source.isdigit() else args.source
    video_processor = VideoProcessor(source)
    
    if not video_processor.open():
        logger.error("Không thể mở camera!")
        return
    
    logger.info("Bắt đầu xử lý video. Nhấn 'q' để thoát.")
    
    try:
        # Xử lý video với callback
        def process_callback(frame):
            return process_frame(
                frame, face_detector, 
                emotion_classifier, engagement_analyzer
            )
        
        video_processor.process_video(
            process_callback,
            output_path=args.output,
            display=not args.no_display,
            window_name="Student Engagement System"
        )
    finally:
        video_processor.close()
        
        # In thống kê cuối cùng
        stats = engagement_analyzer.get_statistics()
        logger.info(f"Thống kê: {stats}")


def run_image_mode(args):
    """Xử lý một ảnh"""
    logger = logging.getLogger(__name__)
    settings = get_settings()
    
    # Khởi tạo
    face_detector = FaceDetector(method=settings.detection.method)
    emotion_classifier = EmotionClassifier()
    engagement_analyzer = EngagementAnalyzer()
    
    # Đọc ảnh
    image = cv2.imread(args.source)
    if image is None:
        logger.error(f"Không thể đọc ảnh: {args.source}")
        return
    
    # Xử lý
    result = process_frame(
        image, face_detector, 
        emotion_classifier, engagement_analyzer
    )
    
    # Hiển thị hoặc lưu
    if args.output:
        cv2.imwrite(args.output, result)
        logger.info(f"Đã lưu kết quả vào: {args.output}")
    
    if not args.no_display:
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(level=log_level)
    
    logger.info("=" * 50)
    logger.info("STUDENT ENGAGEMENT SYSTEM")
    logger.info("Hệ thống phân loại mức độ hứng thú học tập")
    logger.info("=" * 50)
    
    try:
        if args.mode == 'camera' or args.mode == 'video':
            run_camera_mode(args)
        elif args.mode == 'image':
            run_image_mode(args)
    except KeyboardInterrupt:
        logger.info("Dừng bởi người dùng")
    except Exception as e:
        logger.exception(f"Lỗi: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
