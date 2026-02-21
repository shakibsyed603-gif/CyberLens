"""
Severity Classifier Module - Classifies threats into severity levels
Provides visual indicators and color coding for threat levels
"""

from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SeverityClassifier:
    """Classifies anomalies into severity levels"""
    
    # Severity thresholds (Isolation Forest: -1 to +1, lower = more anomalous)
    HIGH_THRESHOLD = -0.7       # score < -0.7 (severe anomaly)
    MEDIUM_THRESHOLD = -0.3     # -0.7 <= score < -0.3 (moderate anomaly)
    # LOW: score >= -0.3 (near normal)
    
    # Color scheme
    COLORS = {
        'HIGH': '#ff6b6b',      # Red
        'MEDIUM': '#ffd93d',    # Yellow
        'LOW': '#6bcf7f'        # Green
    }
    
    # Emoji indicators
    EMOJIS = {
        'HIGH': '🔴',
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }
    
    # Display names
    DISPLAY_NAMES = {
        'HIGH': 'High Severity',
        'MEDIUM': 'Medium Severity',
        'LOW': 'Low Severity'
    }
    
    @staticmethod
    def classify(anomaly_score: float) -> str:
        """
        Classify anomaly score into severity level
        
        Args:
            anomaly_score: Score from Isolation Forest (-1 to 1, lower = more anomalous)
        
        Returns:
            Severity level: 'HIGH', 'MEDIUM', or 'LOW'
        """
        if anomaly_score < SeverityClassifier.HIGH_THRESHOLD:
            return 'HIGH'
        elif anomaly_score < SeverityClassifier.MEDIUM_THRESHOLD:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    @staticmethod
    def get_color(severity: str) -> str:
        """Get color code for severity level"""
        return SeverityClassifier.COLORS.get(severity, '#cccccc')
    
    @staticmethod
    def get_emoji(severity: str) -> str:
        """Get emoji indicator for severity level"""
        return SeverityClassifier.EMOJIS.get(severity, '❓')
    
    @staticmethod
    def get_display_name(severity: str) -> str:
        """Get display name for severity level"""
        return SeverityClassifier.DISPLAY_NAMES.get(severity, 'Unknown')
    
    @staticmethod
    def get_severity_info(anomaly_score: float) -> Dict:
        """
        Get complete severity information
        
        Args:
            anomaly_score: Anomaly score
        
        Returns:
            Dictionary with severity, color, emoji, and display name
        """
        severity = SeverityClassifier.classify(anomaly_score)
        
        return {
            'severity': severity,
            'color': SeverityClassifier.get_color(severity),
            'emoji': SeverityClassifier.get_emoji(severity),
            'display_name': SeverityClassifier.get_display_name(severity),
            'score': anomaly_score
        }
    
    @staticmethod
    def get_confidence(anomaly_score: float) -> float:
        """
        Calculate confidence score based on anomaly score
        
        Args:
            anomaly_score: Anomaly score
        
        Returns:
            Confidence score (0 to 1)
        """
        # FIXED: Proper confidence calculation based on Isolation Forest score range [-1, +1]
        # More negative = higher confidence it's an anomaly
        if anomaly_score < -0.7:
            # High confidence for strong anomalies
            return min(1.0, max(0.0, abs(anomaly_score) * 1.2))
        elif anomaly_score < -0.3:
            # Medium confidence for moderate anomalies
            return min(1.0, max(0.0, abs(anomaly_score) * 0.9))
        else:
            # Low confidence for normal patterns
            return min(1.0, max(0.0, abs(anomaly_score) * 0.5))
    
    @staticmethod
    def get_risk_level(severity: str) -> int:
        """
        Get numeric risk level for sorting
        
        Args:
            severity: Severity level
        
        Returns:
            Risk level (3=HIGH, 2=MEDIUM, 1=LOW)
        """
        risk_levels = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        return risk_levels.get(severity, 0)
    
    @staticmethod
    def format_severity_badge(severity: str) -> str:
        """
        Format severity as HTML badge
        
        Args:
            severity: Severity level
        
        Returns:
            HTML badge string
        """
        color = SeverityClassifier.get_color(severity)
        emoji = SeverityClassifier.get_emoji(severity)
        display_name = SeverityClassifier.get_display_name(severity)
        
        return f'<span style="background-color: {color}; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold;">{emoji} {display_name}</span>'
    
    @staticmethod
    def get_severity_description(severity: str) -> str:
        """Get description for severity level"""
        descriptions = {
            'HIGH': 'Strong anomaly detected. Immediate investigation recommended.',
            'MEDIUM': 'Moderate anomaly detected. Review and monitor closely.',
            'LOW': 'Minor anomaly detected. Normal operation with slight deviation.'
        }
        return descriptions.get(severity, 'Unknown severity level')
    
    @staticmethod
    def get_recommended_action(severity: str) -> str:
        """Get recommended action for severity level"""
        actions = {
            'HIGH': '🚨 IMMEDIATE ACTION: Block connection, investigate source, review logs',
            'MEDIUM': '⚠️ MONITOR: Increase monitoring, prepare response plan',
            'LOW': '✓ OBSERVE: Continue normal monitoring'
        }
        return actions.get(severity, 'No action recommended')


class SeverityStatistics:
    """Statistics for threat severity levels"""
    
    def __init__(self):
        """Initialize statistics tracker"""
        self.counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        self.total = 0
    
    def add_threat(self, severity: str):
        """Add threat to statistics"""
        if severity in self.counts:
            self.counts[severity] += 1
            self.total += 1
    
    def get_counts(self) -> Dict[str, int]:
        """Get threat counts by severity"""
        return self.counts.copy()
    
    def get_percentages(self) -> Dict[str, float]:
        """Get threat percentages by severity"""
        if self.total == 0:
            return {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        return {
            'HIGH': (self.counts['HIGH'] / self.total) * 100,
            'MEDIUM': (self.counts['MEDIUM'] / self.total) * 100,
            'LOW': (self.counts['LOW'] / self.total) * 100
        }
    
    def get_total(self) -> int:
        """Get total threat count"""
        return self.total
    
    def reset(self):
        """Reset statistics"""
        self.counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        self.total = 0
    
    def get_summary(self) -> Dict:
        """Get complete statistics summary"""
        return {
            'counts': self.get_counts(),
            'percentages': self.get_percentages(),
            'total': self.get_total(),
            'high_percentage': self.get_percentages()['HIGH'],
            'medium_percentage': self.get_percentages()['MEDIUM'],
            'low_percentage': self.get_percentages()['LOW']
        }
