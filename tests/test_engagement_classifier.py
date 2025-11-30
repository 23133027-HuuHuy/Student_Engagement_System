"""
Tests for engagement classifier module
"""

import pytest

from src.engagement_classifier.classifier import (
    EngagementClassifier, 
    EngagementLevel,
    get_engagement_label,
    ENGAGEMENT_SCORES
)


class TestEngagementClassifier:
    """Test cases for EngagementClassifier"""
    
    def test_classifier_initialization(self):
        """Test classifier can be initialized"""
        classifier = EngagementClassifier()
        assert classifier is not None
        assert classifier.emotion_mapping is not None
    
    def test_classify_from_emotion_happy(self):
        """Test classification of happy emotion"""
        classifier = EngagementClassifier()
        result = classifier.classify_from_emotion("happy")
        assert result == EngagementLevel.HIGHLY_ENGAGED
    
    def test_classify_from_emotion_sad(self):
        """Test classification of sad emotion"""
        classifier = EngagementClassifier()
        result = classifier.classify_from_emotion("sad")
        assert result == EngagementLevel.DISENGAGED
    
    def test_classify_from_emotion_unknown(self):
        """Test classification of unknown emotion returns neutral"""
        classifier = EngagementClassifier()
        result = classifier.classify_from_emotion("unknown_emotion")
        assert result == EngagementLevel.NEUTRAL
    
    def test_classify_from_emotion_probs(self):
        """Test classification from emotion probabilities"""
        classifier = EngagementClassifier()
        probs = {"happy": 0.8, "neutral": 0.1, "sad": 0.1}
        level, score = classifier.classify_from_emotion_probs(probs)
        assert level in [EngagementLevel.HIGHLY_ENGAGED, EngagementLevel.ENGAGED]
        assert 0 <= score <= 1
    
    def test_get_class_statistics_empty(self):
        """Test statistics with empty history"""
        classifier = EngagementClassifier()
        stats = classifier.get_class_statistics([])
        assert stats == {}
    
    def test_get_class_statistics(self):
        """Test statistics with engagement history"""
        classifier = EngagementClassifier()
        history = [
            EngagementLevel.HIGHLY_ENGAGED,
            EngagementLevel.ENGAGED,
            EngagementLevel.NEUTRAL
        ]
        stats = classifier.get_class_statistics(history)
        assert "counts" in stats
        assert "percentages" in stats
        assert "average_score" in stats
        assert stats["total_samples"] == 3


class TestGetEngagementLabel:
    """Test cases for get_engagement_label function"""
    
    def test_vietnamese_labels(self):
        """Test Vietnamese labels"""
        label = get_engagement_label(EngagementLevel.HIGHLY_ENGAGED, "vi")
        assert label == "Rất hứng thú"
    
    def test_english_labels(self):
        """Test English labels"""
        label = get_engagement_label(EngagementLevel.HIGHLY_ENGAGED, "en")
        assert label == "Highly Engaged"
    
    def test_default_language(self):
        """Test default language is English when unknown"""
        label = get_engagement_label(EngagementLevel.NEUTRAL, "unknown")
        assert label == "Neutral"
