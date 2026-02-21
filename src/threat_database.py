"""
Threat Database Module - SQLite storage for threat logs and history
Handles persistent storage of detected threats with filtering and export capabilities
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ThreatDatabase:
    """SQLite database for storing and managing threat logs"""
    
    def __init__(self, db_path: str = "data/threat_logs.db"):
        """
        Initialize threat database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create threats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    anomaly_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    protocol_type TEXT,
                    service TEXT,
                    src_bytes INTEGER,
                    dst_bytes INTEGER,
                    num_failed_logins INTEGER,
                    root_shell INTEGER,
                    su_attempted INTEGER,
                    shap_summary TEXT,
                    top_features TEXT,
                    is_acknowledged INTEGER DEFAULT 0,
                    acknowledged_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    threat_id INTEGER NOT NULL,
                    alert_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    message TEXT,
                    sent_to TEXT,
                    status TEXT DEFAULT 'sent',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (threat_id) REFERENCES threats(id)
                )
            ''')
            
            # Create indices for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON threats(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON threats(severity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomaly_score ON threats(anomaly_score)')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
        
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def add_threat(self, threat_data: Dict) -> int:
        """
        Add a threat to the database
        
        Args:
            threat_data: Dictionary with threat information
        
        Returns:
            Threat ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO threats (
                    timestamp, severity, anomaly_score, confidence,
                    protocol_type, service, src_bytes, dst_bytes,
                    num_failed_logins, root_shell, su_attempted,
                    shap_summary, top_features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                threat_data.get('timestamp', datetime.now().isoformat()),
                threat_data.get('severity'),
                threat_data.get('anomaly_score'),
                threat_data.get('confidence'),
                threat_data.get('protocol_type'),
                threat_data.get('service'),
                threat_data.get('src_bytes'),
                threat_data.get('dst_bytes'),
                threat_data.get('num_failed_logins'),
                threat_data.get('root_shell'),
                threat_data.get('su_attempted'),
                threat_data.get('shap_summary'),
                threat_data.get('top_features')
            ))
            
            conn.commit()
            threat_id = cursor.lastrowid
            conn.close()
            
            logger.info(f"Threat {threat_id} added to database")
            return threat_id
        
        except Exception as e:
            logger.error(f"Error adding threat: {e}")
            raise
    
    def get_threats(
        self,
        limit: int = 100,
        offset: int = 0,
        severity: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None
    ) -> List[Dict]:
        """
        Get threats with optional filtering
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            severity: Filter by severity (HIGH, MEDIUM, LOW)
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            min_score: Minimum anomaly score
            max_score: Maximum anomaly score
        
        Returns:
            List of threat dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM threats WHERE 1=1"
            params = []
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            if min_score is not None:
                query += " AND anomaly_score >= ?"
                params.append(min_score)
            
            if max_score is not None:
                query += " AND anomaly_score <= ?"
                params.append(max_score)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
        
        except Exception as e:
            logger.error(f"Error getting threats: {e}")
            return []
    
    def get_threat_count(
        self,
        severity: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """Get total count of threats with optional filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT COUNT(*) FROM threats WHERE 1=1"
            params = []
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            cursor.execute(query, params)
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
        
        except Exception as e:
            logger.error(f"Error getting threat count: {e}")
            return 0
    
    def get_severity_counts(self, hours: int = 24) -> Dict[str, int]:
        """Get threat counts by severity for last N hours"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor.execute('''
                SELECT severity, COUNT(*) as count
                FROM threats
                WHERE timestamp >= ?
                GROUP BY severity
            ''', (cutoff_time,))
            
            results = cursor.fetchall()
            conn.close()
            
            counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for severity, count in results:
                if severity in counts:
                    counts[severity] = count
            
            return counts
        
        except Exception as e:
            logger.error(f"Error getting severity counts: {e}")
            return {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    
    def acknowledge_threat(self, threat_id: int) -> bool:
        """Mark threat as acknowledged"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE threats
                SET is_acknowledged = 1, acknowledged_at = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), threat_id))
            
            conn.commit()
            conn.close()
            
            return True
        
        except Exception as e:
            logger.error(f"Error acknowledging threat: {e}")
            return False
    
    def export_to_csv(
        self,
        output_path: str = "threat_logs.csv",
        severity: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> bool:
        """
        Export threats to CSV file
        
        Args:
            output_path: Path to save CSV file
            severity: Filter by severity
            start_date: Filter by start date
            end_date: Filter by end date
        
        Returns:
            True if successful
        """
        try:
            threats = self.get_threats(
                limit=10000,
                severity=severity,
                start_date=start_date,
                end_date=end_date
            )
            
            if not threats:
                logger.warning("No threats to export")
                return False
            
            df = pd.DataFrame(threats)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Exported {len(threats)} threats to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def get_timeline_data(self, hours: int = 24) -> pd.DataFrame:
        """Get threat timeline data for visualization"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            query = '''
                SELECT 
                    DATE(timestamp) as date,
                    HOUR(timestamp) as hour,
                    severity,
                    COUNT(*) as count
                FROM threats
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp), HOUR(timestamp), severity
                ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(query, conn, params=(cutoff_time,))
            conn.close()
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting timeline data: {e}")
            return pd.DataFrame()
    
    def add_alert(self, threat_id: int, alert_type: str, message: str, sent_to: str = "") -> bool:
        """Add alert record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (threat_id, alert_type, timestamp, message, sent_to)
                VALUES (?, ?, ?, ?, ?)
            ''', (threat_id, alert_type, datetime.now().isoformat(), message, sent_to))
            
            conn.commit()
            conn.close()
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
            return False
    
    def get_recent_threats(self, minutes: int = 60) -> List[Dict]:
        """Get threats from last N minutes"""
        cutoff_time = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        return self.get_threats(
            limit=1000,
            start_date=cutoff_time
        )
    
    def cleanup_old_data(self, days: int = 90) -> int:
        """Delete threats older than N days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute('DELETE FROM threats WHERE timestamp < ?', (cutoff_time,))
            deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted {deleted} old threat records")
            return deleted
        
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
