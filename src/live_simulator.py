"""
Live Data Simulator for CyberLens
Simulates real-time network traffic by processing data row-by-row
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import threading
import queue
import time
from datetime import datetime, timedelta
from collections import deque

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_PATH, SCALER_PATH, PROCESSED_DATA_DIR


class LiveDataSimulator:
    """Simulates live network traffic data processing"""
    
    def __init__(self, max_buffer_size=10):
        """
        Initialize the live data simulator
        
        Args:
            max_buffer_size: Maximum number of recent detections to keep
        """
        self.model = None
        self.scaler = None
        self.data_source = None
        self.current_index = 0
        self.max_buffer_size = max_buffer_size
        self.detection_buffer = deque(maxlen=max_buffer_size)
        self.is_running = False
        self.processing_thread = None
        self.detection_queue = queue.Queue()
        self.total_processed = 0
        self.total_anomalies = 0
        self.start_time = None
        
    def load_model_and_data(self):
        """Load the trained model, scaler, and data source"""
        try:
            # Load model
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            self.model = joblib.load(MODEL_PATH)
            
            # Load scaler
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
            self.scaler = joblib.load(SCALER_PATH)
            
            # Load data source
            processed_data_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
            if not os.path.exists(processed_data_path):
                raise FileNotFoundError(f"Processed data not found at {processed_data_path}")
            
            processed_data = joblib.load(processed_data_path)
            self.data_source = processed_data['X'].copy()
            self.current_index = 0
            
            return True
        except Exception as e:
            print(f"Error loading model and data: {str(e)}")
            return False
    
    def get_threat_level(self, anomaly_score):
        """Determine threat level based on anomaly score"""
        if anomaly_score < -0.5:
            return "High"
        elif anomaly_score < -0.2:
            return "Medium"
        else:
            return "Low"
    
    def get_attack_type(self, anomaly_score):
        """Simulate attack type based on anomaly score"""
        attack_types = ['DoS', 'Probe', 'R2L', 'U2R', 'Unknown']
        # Use anomaly score to influence attack type selection
        seed_value = int(abs(anomaly_score) * 1000) % len(attack_types)
        return attack_types[seed_value]
    
    def process_next_row(self):
        """Process the next row of data and return detection info"""
        if self.data_source is None or self.current_index >= len(self.data_source):
            # Cycle back to beginning
            self.current_index = 0
        
        # Get the next row
        row = self.data_source.iloc[self.current_index].values.reshape(1, -1)
        self.current_index += 1
        self.total_processed += 1
        
        # Make prediction
        prediction = self.model.predict(row)[0]
        anomaly_score = self.model.decision_function(row)[0]
        
        is_anomaly = prediction == -1
        if is_anomaly:
            self.total_anomalies += 1
        
        # Create detection record
        detection = {
            'timestamp': datetime.now(),
            'is_anomaly': is_anomaly,
            'anomaly_score': float(anomaly_score),
            'threat_level': self.get_threat_level(anomaly_score),
            'attack_type': self.get_attack_type(anomaly_score) if is_anomaly else 'Normal',
            'packet_size': int(np.random.randint(50, 1500)),
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP']),
            'source_ip': f"{np.random.randint(1, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}",
            'destination_ip': f"{np.random.randint(1, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}",
        }
        
        # Add to buffer
        self.detection_buffer.append(detection)
        
        return detection
    
    def start_monitoring(self, interval=2.5):
        """
        Start the monitoring thread
        
        Args:
            interval: Time in seconds between processing each row
        """
        if self.is_running:
            return False
        
        if self.model is None:
            self.load_model_and_data()
        
        if self.model is None:
            return False
        
        self.is_running = True
        self.start_time = datetime.now()
        self.processing_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.processing_thread.start()
        return True
    
    def _monitoring_loop(self, interval):
        """Internal loop for processing data"""
        while self.is_running:
            try:
                detection = self.process_next_row()
                self.detection_queue.put(detection)
                time.sleep(interval)
            except Exception as e:
                print(f"Error in monitoring loop: {str(e)}")
                self.is_running = False
                break
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
    
    def get_latest_detections(self, count=10):
        """Get the latest detections from the buffer"""
        return list(self.detection_buffer)[-count:]
    
    def get_statistics(self):
        """Get current monitoring statistics"""
        stats = {
            'total_processed': self.total_processed,
            'total_anomalies': self.total_anomalies,
            'anomaly_rate': (self.total_anomalies / max(self.total_processed, 1)) * 100,
            'is_running': self.is_running,
            'start_time': self.start_time,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'buffer_size': len(self.detection_buffer)
        }
        return stats
    
    def get_detection_from_queue(self, timeout=0.1):
        """Get a detection from the queue (non-blocking)"""
        try:
            return self.detection_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def reset(self):
        """Reset the simulator"""
        self.stop_monitoring()
        self.current_index = 0
        self.total_processed = 0
        self.total_anomalies = 0
        self.detection_buffer.clear()
        self.start_time = None
