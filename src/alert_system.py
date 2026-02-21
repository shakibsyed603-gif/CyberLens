"""
Alert System Module - Handles notifications for detected threats
Supports dashboard alerts and email notifications
"""

import smtplib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

logger = logging.getLogger(__name__)


class AlertSystem:
    """Manages threat alerts and notifications"""
    
    def __init__(self, email_config: Optional[Dict] = None):
        """
        Initialize alert system
        
        Args:
            email_config: Dictionary with SMTP configuration
                {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'sender_email': 'your_email@gmail.com',
                    'sender_password': 'your_password',
                    'recipient_emails': ['admin@example.com']
                }
        """
        self.email_config = email_config or {}
        self.alert_history = {}  # Track recent alerts to prevent duplicates
        self.alert_cooldown = 300  # 5 minutes cooldown between same alerts
    
    def should_alert(self, threat_key: str) -> bool:
        """
        Check if alert should be sent (prevent duplicates)
        
        Args:
            threat_key: Unique identifier for threat
        
        Returns:
            True if alert should be sent
        """
        now = datetime.now()
        
        if threat_key not in self.alert_history:
            self.alert_history[threat_key] = now
            return True
        
        last_alert = self.alert_history[threat_key]
        if (now - last_alert).total_seconds() > self.alert_cooldown:
            self.alert_history[threat_key] = now
            return True
        
        return False
    
    def create_dashboard_alert(self, threat_data: Dict) -> Dict:
        """
        Create dashboard alert notification
        
        Args:
            threat_data: Threat information
        
        Returns:
            Alert dictionary for Streamlit notification
        """
        severity = threat_data.get('severity', 'UNKNOWN')
        anomaly_score = threat_data.get('anomaly_score', 0)
        timestamp = threat_data.get('timestamp', datetime.now().isoformat())
        
        alert_type = "error" if severity == "HIGH" else "warning" if severity == "MEDIUM" else "info"
        
        message = f"""
🚨 **{severity} SEVERITY THREAT DETECTED**

⏰ Timestamp: {timestamp}
📊 Anomaly Score: {anomaly_score:.4f}
🎯 Severity: {severity}

Details:
- Protocol: {threat_data.get('protocol_type', 'N/A')}
- Service: {threat_data.get('service', 'N/A')}
- Confidence: {threat_data.get('confidence', 0):.2%}

Top Contributing Features:
{threat_data.get('shap_summary', 'N/A')}
        """
        
        return {
            'type': alert_type,
            'message': message,
            'timestamp': timestamp,
            'severity': severity
        }
    
    def send_email_alert(
        self,
        threat_data: Dict,
        recipient_emails: Optional[List[str]] = None
    ) -> bool:
        """
        Send email alert for critical threats
        
        Args:
            threat_data: Threat information
            recipient_emails: List of email addresses
        
        Returns:
            True if email sent successfully
        """
        if not self.email_config:
            logger.warning("Email configuration not set")
            return False
        
        recipients = recipient_emails or self.email_config.get('recipient_emails', [])
        
        if not recipients:
            logger.warning("No recipient emails configured")
            return False
        
        try:
            # Create email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🚨 {threat_data.get('severity')} SEVERITY THREAT DETECTED"
            msg['From'] = self.email_config.get('sender_email')
            msg['To'] = ', '.join(recipients)
            
            # Create plain text and HTML versions
            text = self._create_email_text(threat_data)
            html = self._create_email_html(threat_data)
            
            msg.attach(MIMEText(text, 'plain'))
            msg.attach(MIMEText(html, 'html'))
            
            # Send email
            server = smtplib.SMTP(
                self.email_config.get('smtp_server', 'smtp.gmail.com'),
                self.email_config.get('smtp_port', 587)
            )
            server.starttls()
            server.login(
                self.email_config.get('sender_email'),
                self.email_config.get('sender_password')
            )
            server.sendmail(
                self.email_config.get('sender_email'),
                recipients,
                msg.as_string()
            )
            server.quit()
            
            logger.info(f"Email alert sent to {recipients}")
            return True
        
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False
    
    def _create_email_text(self, threat_data: Dict) -> str:
        """Create plain text email content"""
        return f"""
CYBERLENS THREAT ALERT
{'='*50}

SEVERITY: {threat_data.get('severity')}
TIMESTAMP: {threat_data.get('timestamp')}
ANOMALY SCORE: {threat_data.get('anomaly_score'):.4f}
CONFIDENCE: {threat_data.get('confidence'):.2%}

CONNECTION DETAILS:
- Protocol: {threat_data.get('protocol_type')}
- Service: {threat_data.get('service')}
- Source Bytes: {threat_data.get('src_bytes')}
- Destination Bytes: {threat_data.get('dst_bytes')}

THREAT ANALYSIS:
{threat_data.get('shap_summary', 'N/A')}

TOP CONTRIBUTING FEATURES:
{threat_data.get('top_features', 'N/A')}

{'='*50}
This is an automated alert from CyberLens
"""
    
    def _create_email_html(self, threat_data: Dict) -> str:
        """Create HTML email content"""
        severity = threat_data.get('severity', 'UNKNOWN')
        color = '#ff6b6b' if severity == 'HIGH' else '#ffd93d' if severity == 'MEDIUM' else '#6bcf7f'
        
        return f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; background-color: #f5f5f5; }}
        .container {{ max-width: 600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; }}
        .header {{ background-color: {color}; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .section {{ margin-bottom: 20px; }}
        .section h2 {{ color: #333; font-size: 16px; border-bottom: 2px solid {color}; padding-bottom: 10px; }}
        .detail {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }}
        .detail-label {{ font-weight: bold; color: #666; }}
        .detail-value {{ color: #333; }}
        .features {{ background-color: #f9f9f9; padding: 10px; border-left: 4px solid {color}; }}
        .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚨 {severity} SEVERITY THREAT DETECTED</h1>
        </div>
        
        <div class="section">
            <h2>Alert Details</h2>
            <div class="detail">
                <span class="detail-label">Timestamp:</span>
                <span class="detail-value">{threat_data.get('timestamp')}</span>
            </div>
            <div class="detail">
                <span class="detail-label">Severity:</span>
                <span class="detail-value">{severity}</span>
            </div>
            <div class="detail">
                <span class="detail-label">Anomaly Score:</span>
                <span class="detail-value">{threat_data.get('anomaly_score'):.4f}</span>
            </div>
            <div class="detail">
                <span class="detail-label">Confidence:</span>
                <span class="detail-value">{threat_data.get('confidence'):.2%}</span>
            </div>
        </div>
        
        <div class="section">
            <h2>Connection Details</h2>
            <div class="detail">
                <span class="detail-label">Protocol:</span>
                <span class="detail-value">{threat_data.get('protocol_type')}</span>
            </div>
            <div class="detail">
                <span class="detail-label">Service:</span>
                <span class="detail-value">{threat_data.get('service')}</span>
            </div>
            <div class="detail">
                <span class="detail-label">Source Bytes:</span>
                <span class="detail-value">{threat_data.get('src_bytes')}</span>
            </div>
            <div class="detail">
                <span class="detail-label">Destination Bytes:</span>
                <span class="detail-value">{threat_data.get('dst_bytes')}</span>
            </div>
        </div>
        
        <div class="section">
            <h2>Threat Analysis</h2>
            <div class="features">
                {threat_data.get('shap_summary', 'N/A')}
            </div>
        </div>
        
        <div class="section">
            <h2>Top Contributing Features</h2>
            <div class="features">
                {threat_data.get('top_features', 'N/A')}
            </div>
        </div>
        
        <div class="footer">
            <p>This is an automated alert from CyberLens Threat Detection System</p>
        </div>
    </div>
</body>
</html>
"""
    
    def create_alert_summary(self, threat_data: Dict) -> str:
        """Create brief alert summary"""
        return f"{threat_data.get('severity')} threat detected at {threat_data.get('timestamp')} (Score: {threat_data.get('anomaly_score'):.4f})"
