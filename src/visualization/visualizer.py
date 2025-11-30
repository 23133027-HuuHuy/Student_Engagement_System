"""
Visualization Module - Hiển thị kết quả
Vẽ các kết quả phát hiện khuôn mặt và phân loại hứng thú
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

from ..engagement_classifier.classifier import EngagementLevel, get_engagement_label


# Colors for different engagement levels (BGR format)
ENGAGEMENT_COLORS = {
    EngagementLevel.HIGHLY_ENGAGED: (0, 255, 0),      # Green
    EngagementLevel.ENGAGED: (0, 200, 100),           # Light green
    EngagementLevel.NEUTRAL: (0, 255, 255),           # Yellow
    EngagementLevel.DISENGAGED: (0, 165, 255),        # Orange
    EngagementLevel.HIGHLY_DISENGAGED: (0, 0, 255),   # Red
}


def draw_face_box(image: np.ndarray,
                  bbox: Tuple[int, int, int, int],
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box around face
    
    Args:
        image: Input image
        bbox: Bounding box (x, y, w, h)
        color: Box color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with drawn box
    """
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image


def draw_engagement_label(image: np.ndarray,
                          bbox: Tuple[int, int, int, int],
                          level: EngagementLevel,
                          language: str = "vi",
                          font_scale: float = 0.6) -> np.ndarray:
    """
    Draw engagement label above face box
    
    Args:
        image: Input image
        bbox: Face bounding box
        level: Engagement level
        language: Label language
        font_scale: Font scale
        
    Returns:
        Image with label
    """
    x, y, w, h = bbox
    label = get_engagement_label(level, language)
    color = ENGAGEMENT_COLORS.get(level, (255, 255, 255))
    
    # Draw background rectangle for text
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
    )
    
    # Draw text background
    cv2.rectangle(image, 
                  (x, y - text_h - 10), 
                  (x + text_w + 10, y),
                  color, -1)
    
    # Draw text
    cv2.putText(image, label, 
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                (255, 255, 255), 2)
    
    return image


def draw_emotion_bar(image: np.ndarray,
                     emotion_probs: Dict[str, float],
                     position: Tuple[int, int] = (10, 10),
                     bar_width: int = 150,
                     bar_height: int = 15) -> np.ndarray:
    """
    Draw emotion probability bars
    
    Args:
        image: Input image
        emotion_probs: Dictionary of emotion probabilities
        position: Starting position (x, y)
        bar_width: Maximum bar width
        bar_height: Bar height
        
    Returns:
        Image with emotion bars
    """
    x, y = position
    
    for i, (emotion, prob) in enumerate(sorted(emotion_probs.items())):
        # Draw label
        cv2.putText(image, f"{emotion}:", 
                    (x, y + i * (bar_height + 5) + bar_height - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw bar background
        bar_x = x + 70
        bar_y = y + i * (bar_height + 5)
        cv2.rectangle(image, 
                      (bar_x, bar_y), 
                      (bar_x + bar_width, bar_y + bar_height),
                      (100, 100, 100), -1)
        
        # Draw bar fill
        fill_width = int(bar_width * prob)
        cv2.rectangle(image,
                      (bar_x, bar_y),
                      (bar_x + fill_width, bar_y + bar_height),
                      (0, 200, 100), -1)
        
        # Draw percentage
        cv2.putText(image, f"{prob*100:.1f}%",
                    (bar_x + bar_width + 5, bar_y + bar_height - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    return image


def draw_class_statistics(image: np.ndarray,
                          stats: Dict,
                          position: Tuple[int, int] = (10, 10)) -> np.ndarray:
    """
    Draw class engagement statistics
    
    Args:
        image: Input image
        stats: Statistics dictionary from EngagementClassifier.get_class_statistics
        position: Starting position
        
    Returns:
        Image with statistics overlay
    """
    x, y = position
    
    if not stats:
        return image
    
    # Draw title
    cv2.putText(image, "Class Engagement:",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y += 25
    
    # Draw percentages for each level
    percentages = stats.get("percentages", {})
    for level_name, percentage in percentages.items():
        cv2.putText(image, f"{level_name}: {percentage:.1f}%",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
    
    # Draw average score
    avg_score = stats.get("average_score", 0)
    cv2.putText(image, f"Avg Score: {avg_score:.2f}",
                (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return image


def visualize_results(image: np.ndarray,
                      faces: List[Tuple[int, int, int, int]],
                      engagement_levels: List[EngagementLevel],
                      emotion_probs: Optional[List[Dict[str, float]]] = None,
                      show_emotions: bool = True,
                      language: str = "vi") -> np.ndarray:
    """
    Main visualization function - draw all results on image
    
    Args:
        image: Input image
        faces: List of face bounding boxes
        engagement_levels: List of engagement levels for each face
        emotion_probs: Optional list of emotion probabilities for each face
        show_emotions: Whether to show emotion bars
        language: Label language
        
    Returns:
        Annotated image
    """
    result = image.copy()
    
    for i, (bbox, level) in enumerate(zip(faces, engagement_levels)):
        color = ENGAGEMENT_COLORS.get(level, (255, 255, 255))
        
        # Draw face box
        result = draw_face_box(result, bbox, color)
        
        # Draw engagement label
        result = draw_engagement_label(result, bbox, level, language)
        
        # Draw emotion bars if available
        if show_emotions and emotion_probs and i < len(emotion_probs):
            x, y, w, h = bbox
            result = draw_emotion_bar(result, emotion_probs[i], 
                                       position=(x + w + 10, y))
    
    return result
