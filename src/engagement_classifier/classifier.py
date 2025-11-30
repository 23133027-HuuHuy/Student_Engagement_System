"""
Engagement Classifier - Phân loại mức độ hứng thú
Phân loại mức độ hứng thú học tập dựa trên cảm xúc
"""

from typing import Dict, Tuple, List
from enum import Enum


class EngagementLevel(Enum):
    """Các mức độ hứng thú học tập"""
    HIGHLY_ENGAGED = "highly_engaged"      # Rất hứng thú
    ENGAGED = "engaged"                     # Hứng thú
    NEUTRAL = "neutral"                     # Bình thường
    DISENGAGED = "disengaged"              # Không hứng thú
    HIGHLY_DISENGAGED = "highly_disengaged"  # Rất không hứng thú


# Mapping từ cảm xúc sang mức độ hứng thú
DEFAULT_EMOTION_MAPPING = {
    "happy": EngagementLevel.HIGHLY_ENGAGED,
    "surprise": EngagementLevel.ENGAGED,
    "neutral": EngagementLevel.NEUTRAL,
    "sad": EngagementLevel.DISENGAGED,
    "angry": EngagementLevel.HIGHLY_DISENGAGED,
    "fear": EngagementLevel.DISENGAGED,
    "disgust": EngagementLevel.DISENGAGED
}

# Điểm số cho mỗi mức độ hứng thú (dùng để tính trung bình)
ENGAGEMENT_SCORES = {
    EngagementLevel.HIGHLY_ENGAGED: 1.0,
    EngagementLevel.ENGAGED: 0.75,
    EngagementLevel.NEUTRAL: 0.5,
    EngagementLevel.DISENGAGED: 0.25,
    EngagementLevel.HIGHLY_DISENGAGED: 0.0
}


class EngagementClassifier:
    """Classifier for student engagement levels"""
    
    def __init__(self, emotion_mapping: Dict[str, EngagementLevel] = None):
        """
        Initialize engagement classifier
        
        Args:
            emotion_mapping: Custom emotion to engagement mapping
        """
        self.emotion_mapping = emotion_mapping or DEFAULT_EMOTION_MAPPING
    
    def classify_from_emotion(self, emotion: str) -> EngagementLevel:
        """
        Classify engagement level from detected emotion
        
        Args:
            emotion: Detected emotion name
            
        Returns:
            EngagementLevel enum
        """
        return self.emotion_mapping.get(emotion, EngagementLevel.NEUTRAL)
    
    def classify_from_emotion_probs(self, 
                                     emotion_probs: Dict[str, float]) -> Tuple[EngagementLevel, float]:
        """
        Classify engagement level from emotion probabilities
        
        Args:
            emotion_probs: Dictionary of emotion probabilities
            
        Returns:
            Tuple of (EngagementLevel, confidence)
        """
        # Calculate weighted engagement score
        total_score = 0.0
        total_weight = 0.0
        
        for emotion, prob in emotion_probs.items():
            if emotion in self.emotion_mapping:
                level = self.emotion_mapping[emotion]
                score = ENGAGEMENT_SCORES[level]
                total_score += score * prob
                total_weight += prob
        
        if total_weight > 0:
            avg_score = total_score / total_weight
        else:
            avg_score = 0.5
        
        # Convert score to engagement level
        if avg_score >= 0.9:
            level = EngagementLevel.HIGHLY_ENGAGED
        elif avg_score >= 0.7:
            level = EngagementLevel.ENGAGED
        elif avg_score >= 0.4:
            level = EngagementLevel.NEUTRAL
        elif avg_score >= 0.2:
            level = EngagementLevel.DISENGAGED
        else:
            level = EngagementLevel.HIGHLY_DISENGAGED
        
        return level, avg_score
    
    def get_class_statistics(self, 
                              engagement_history: List[EngagementLevel]) -> Dict[str, float]:
        """
        Get engagement statistics for a class/group
        
        Args:
            engagement_history: List of engagement levels
            
        Returns:
            Dictionary with engagement statistics
        """
        if not engagement_history:
            return {}
        
        # Count each level
        counts = {}
        for level in EngagementLevel:
            counts[level.value] = sum(1 for e in engagement_history if e == level)
        
        # Calculate percentages
        total = len(engagement_history)
        percentages = {k: v / total * 100 for k, v in counts.items()}
        
        # Calculate average score
        scores = [ENGAGEMENT_SCORES[e] for e in engagement_history]
        avg_score = sum(scores) / len(scores)
        
        return {
            "counts": counts,
            "percentages": percentages,
            "average_score": avg_score,
            "total_samples": total
        }


def get_engagement_label(level: EngagementLevel, language: str = "vi") -> str:
    """
    Get human-readable label for engagement level
    
    Args:
        level: EngagementLevel enum
        language: Language code ("vi" for Vietnamese, "en" for English)
        
    Returns:
        Human-readable label
    """
    labels = {
        "vi": {
            EngagementLevel.HIGHLY_ENGAGED: "Rất hứng thú",
            EngagementLevel.ENGAGED: "Hứng thú",
            EngagementLevel.NEUTRAL: "Bình thường",
            EngagementLevel.DISENGAGED: "Không hứng thú",
            EngagementLevel.HIGHLY_DISENGAGED: "Rất không hứng thú"
        },
        "en": {
            EngagementLevel.HIGHLY_ENGAGED: "Highly Engaged",
            EngagementLevel.ENGAGED: "Engaged",
            EngagementLevel.NEUTRAL: "Neutral",
            EngagementLevel.DISENGAGED: "Disengaged",
            EngagementLevel.HIGHLY_DISENGAGED: "Highly Disengaged"
        }
    }
    
    return labels.get(language, labels["en"]).get(level, level.value)
