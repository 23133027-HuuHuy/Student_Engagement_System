"""
Main entry point - Điểm khởi chạy chính
Chạy hệ thống phân loại mức độ hứng thú học tập
"""

import cv2
import argparse
from typing import Optional

from src.face_detection.detector import get_detector
from src.emotion_recognition.recognizer import get_recognizer, preprocess_face
from src.engagement_classifier.classifier import EngagementClassifier
from src.visualization.visualizer import visualize_results
from utils.helpers import load_config


def process_frame(frame, face_detector, emotion_recognizer, engagement_classifier, config):
    """
    Process a single frame
    
    Args:
        frame: Input frame
        face_detector: Face detector instance
        emotion_recognizer: Emotion recognizer instance
        engagement_classifier: Engagement classifier instance
        config: Configuration dictionary
        
    Returns:
        Processed frame with annotations
    """
    # Detect faces
    faces = face_detector.detect(frame)
    
    engagement_levels = []
    emotion_probs_list = []
    
    for face_bbox in faces:
        x, y, w, h = face_bbox
        
        # Crop face
        face_img = frame[y:y+h, x:x+w]
        
        if face_img.shape[0] == 0 or face_img.shape[1] == 0:
            continue
        
        # Recognize emotions
        emotion_probs = emotion_recognizer.predict(face_img)
        emotion_probs_list.append(emotion_probs)
        
        # Classify engagement
        level, _ = engagement_classifier.classify_from_emotion_probs(emotion_probs)
        engagement_levels.append(level)
    
    # Visualize results
    result = visualize_results(
        frame, faces, engagement_levels, emotion_probs_list,
        show_emotions=config.get('visualization', {}).get('show_emotions', True),
        language='vi'
    )
    
    return result


def run_camera(config: dict, camera_id: int = 0):
    """
    Run real-time engagement detection from camera
    
    Args:
        config: Configuration dictionary
        camera_id: Camera device ID
    """
    # Initialize components
    face_detector = get_detector(
        config.get('face_detection', {}).get('model', 'haar_cascade')
    )
    emotion_recognizer = get_recognizer('simple')
    engagement_classifier = EngagementClassifier()
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return
    
    # Set camera properties
    camera_config = config.get('camera', {})
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.get('width', 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.get('height', 480))
    
    print("Press 'q' to quit, 's' to save screenshot")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Cannot read frame")
            break
        
        # Process frame
        result = process_frame(
            frame, face_detector, emotion_recognizer, 
            engagement_classifier, config
        )
        
        # Display result
        cv2.imshow('Student Engagement System', result)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('screenshot.png', result)
            print("Screenshot saved!")
    
    cap.release()
    cv2.destroyAllWindows()


def run_image(config: dict, image_path: str, output_path: Optional[str] = None):
    """
    Run engagement detection on a single image
    
    Args:
        config: Configuration dictionary
        image_path: Path to input image
        output_path: Path to save output image (optional)
    """
    # Initialize components
    face_detector = get_detector(
        config.get('face_detection', {}).get('model', 'haar_cascade')
    )
    emotion_recognizer = get_recognizer('simple')
    engagement_classifier = EngagementClassifier()
    
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    # Process image
    result = process_frame(
        image, face_detector, emotion_recognizer,
        engagement_classifier, config
    )
    
    # Save or display result
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Result saved to {output_path}")
    else:
        cv2.imshow('Student Engagement System', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Student Engagement Classification System'
    )
    parser.add_argument(
        '--mode', choices=['camera', 'image'], default='camera',
        help='Run mode: camera (real-time) or image (single image)'
    )
    parser.add_argument(
        '--config', default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input', help='Input image path (for image mode)'
    )
    parser.add_argument(
        '--output', help='Output image path (for image mode)'
    )
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera device ID (for camera mode)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.mode == 'camera':
        run_camera(config, args.camera)
    else:
        if not args.input:
            print("Error: --input is required for image mode")
            return
        run_image(config, args.input, args.output)


if __name__ == '__main__':
    main()
