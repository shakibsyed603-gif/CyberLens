try:
    import scapy.all as scapy
    SCAPY_AVAILABLE = True
except Exception:
    # scapy not installed or failed to import (e.g., on systems without native libs)
    SCAPY_AVAILABLE = False

    # Provide a lightweight stub object with the minimal attributes used by the
    # rest of the code so importing this module won't crash the app. Live
    # capture functionality will be disabled when scapy is unavailable.
    class _ScapyStub:
        class _Proto:
            pass

        IP = _Proto()
        TCP = _Proto()
        UDP = _Proto()
        ICMP = _Proto()

        class conf:
            iface = None

        @staticmethod
        def sniff(*args, **kwargs):
            raise RuntimeError("scapy not available; live capture disabled")

        @staticmethod
        def get_if_list():
            return []

    scapy = _ScapyStub()
import pandas as pd
import numpy as np
import joblib
import os
import sys
import threading
import queue
from collections import defaultdict
from datetime import datetime
import shap

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_PATH, SCALER_PATH, PROCESSED_DATA_DIR

class LivePacketCapture:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.explainer = None
        self.packet_queue = queue.Queue()
        self.is_capturing = False
        self.packet_buffer = []
        self.flow_stats = defaultdict(lambda: {
            'duration': 0,
            'src_bytes': 0,
            'dst_bytes': 0,
            'count': 0,
            'srv_count': 0,
            'serror_rate': 0,
            'srv_serror_rate': 0,
            'rerror_rate': 0,
            'srv_rerror_rate': 0,
            'same_srv_rate': 0,
            'diff_srv_rate': 0,
            'srv_diff_host_rate': 0,
            'dst_host_count': 0,
            'dst_host_srv_count': 0,
            'dst_host_same_srv_rate': 0,
            'dst_host_diff_srv_rate': 0,
            'dst_host_same_src_port_rate': 0,
            'dst_host_srv_diff_host_rate': 0,
            'dst_host_serror_rate': 0,
            'dst_host_srv_serror_rate': 0,
            'dst_host_rerror_rate': 0,
            'dst_host_srv_rerror_rate': 0,
        })
        
    def load_model_and_scaler(self):
        """Load the trained model and scaler"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
        
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        
        # Load feature columns
        processed_data_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
        if os.path.exists(processed_data_path):
            processed_data = joblib.load(processed_data_path)
            self.feature_columns = processed_data['feature_columns']
        
        return True
    
    def initialize_explainer(self, background_size=50):
        """Initialize SHAP explainer for live packets"""
        try:
            processed_data_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
            if not os.path.exists(processed_data_path):
                return False
            
            processed_data = joblib.load(processed_data_path)
            X = processed_data['X']
            
            # Use a small background sample for faster SHAP calculations
            background_data = X.sample(min(background_size, len(X)), random_state=42)
            
            self.explainer = shap.KernelExplainer(
                self.model.decision_function,
                background_data
            )
            return True
        except Exception as e:
            print(f"Error initializing explainer: {e}")
            return False
    
    def extract_packet_features(self, packet):
        """Extract network features from a packet"""
        features = {
            'protocol_type': 0,  # TCP=0, UDP=1, ICMP=2
            'service': 0,  # Port-based service
            'flag': 0,  # TCP flags
            'src_bytes': 0,
            'dst_bytes': 0,
            'land': 0,
            'wrong_fragment': 0,
            'urgent': 0,
            'hot': 0,
            'num_failed_logins': 0,
            'logged_in': 0,
            'num_compromised': 0,
            'root_shell': 0,
            'su_attempted': 0,
            'num_root': 0,
            'num_file_creations': 0,
            'num_shells': 0,
            'num_access_files': 0,
            'num_outbound_cmds': 0,
            'is_host_login': 0,
            'is_guest_login': 0,
            'count': 1,
            'srv_count': 1,
            'serror_rate': 0,
            'srv_serror_rate': 0,
            'rerror_rate': 0,
            'srv_rerror_rate': 0,
            'same_srv_rate': 0,
            'diff_srv_rate': 0,
            'srv_diff_host_rate': 0,
            'dst_host_count': 1,
            'dst_host_srv_count': 1,
            'dst_host_same_srv_rate': 0,
            'dst_host_diff_srv_rate': 0,
            'dst_host_same_src_port_rate': 0,
            'dst_host_srv_diff_host_rate': 0,
            'dst_host_serror_rate': 0,
            'dst_host_srv_serror_rate': 0,
            'dst_host_rerror_rate': 0,
            'dst_host_srv_rerror_rate': 0,
        }
        
        try:
            # Extract IP layer info
            if scapy.IP in packet:
                ip_layer = packet[scapy.IP]
                features['src_bytes'] = len(packet)
                
                # Determine protocol
                if scapy.TCP in packet:
                    features['protocol_type'] = 0
                    tcp_layer = packet[scapy.TCP]
                    features['dst_bytes'] = len(packet)
                    # Extract TCP flags
                    flags = tcp_layer.flags
                    if flags & 0x01:  # FIN
                        features['flag'] = 1
                    elif flags & 0x02:  # SYN
                        features['flag'] = 2
                    elif flags & 0x04:  # RST
                        features['flag'] = 3
                    elif flags & 0x10:  # ACK
                        features['flag'] = 4
                    
                elif scapy.UDP in packet:
                    features['protocol_type'] = 1
                    features['dst_bytes'] = len(packet)
                    
                elif scapy.ICMP in packet:
                    features['protocol_type'] = 2
                    features['dst_bytes'] = len(packet)
                
                # Check for land attack (src == dst)
                if ip_layer.src == ip_layer.dst:
                    features['land'] = 1
        
        except Exception as e:
            pass
        
        return features
    
    def preprocess_packet_features(self, features_dict):
        """Preprocess extracted features to match KDD format"""
        try:
            # Create DataFrame with features in the correct order
            feature_df = pd.DataFrame([features_dict])
            
            # Ensure all feature columns exist
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            
            # Select only the required features in correct order
            feature_df = feature_df[self.feature_columns]
            
            # Scale features
            features_scaled = self.scaler.transform(feature_df)
            
            return pd.DataFrame(features_scaled, columns=self.feature_columns)
        except Exception as e:
            print(f"Error preprocessing features: {e}")
            return None
    
    def predict_anomaly(self, features_df):
        """Predict if packet is anomalous"""
        try:
            prediction = self.model.predict(features_df)[0]
            anomaly_score = self.model.decision_function(features_df)[0]
            
            return {
                'is_anomaly': prediction == -1,
                'anomaly_score': float(anomaly_score),
                'prediction': int(prediction)
            }
        except Exception as e:
            print(f"Error predicting anomaly: {e}")
            return None
    
    def get_shap_explanation(self, features_df):
        """Generate SHAP explanation for a packet"""
        try:
            if self.explainer is None:
                return None
            
            shap_values = self.explainer.shap_values(features_df)
            
            # Get top contributing features
            feature_contributions = pd.DataFrame({
                'feature': self.feature_columns,
                'value': features_df.iloc[0].values,
                'shap_value': shap_values[0]
            })
            
            feature_contributions['abs_shap'] = np.abs(feature_contributions['shap_value'])
            feature_contributions = feature_contributions.sort_values('abs_shap', ascending=False)
            
            return feature_contributions.head(10)
        except Exception as e:
            print(f"Error generating SHAP explanation: {e}")
            return None
    
    def packet_callback(self, packet):
        """Callback function for packet processing"""
        if self.is_capturing:
            try:
                # Extract features from packet
                features = self.extract_packet_features(packet)
                
                # Preprocess features
                features_df = self.preprocess_packet_features(features)
                
                if features_df is not None:
                    # Predict anomaly
                    prediction = self.predict_anomaly(features_df)
                    
                    if prediction is not None:
                        packet_info = {
                            'timestamp': datetime.now(),
                            'features': features,
                            'features_df': features_df,
                            'prediction': prediction,
                            'shap_explanation': None
                        }
                        
                        # Generate SHAP explanation if anomaly detected
                        if prediction['is_anomaly'] and self.explainer is not None:
                            packet_info['shap_explanation'] = self.get_shap_explanation(features_df)
                        
                        self.packet_queue.put(packet_info)
            except Exception as e:
                print(f"Error in packet callback: {e}")
    
    def start_capture(self, interface=None, packet_count=0):
        """Start live packet capture"""
        self.is_capturing = True
        try:
            # If no interface specified, use default
            if interface is None:
                interface = scapy.conf.iface
            
            # Start sniffing in a separate thread
            scapy.sniff(
                iface=interface,
                prn=self.packet_callback,
                store=False,
                count=packet_count if packet_count > 0 else 0
            )
        except Exception as e:
            print(f"Error starting capture: {e}")
            self.is_capturing = False
    
    def stop_capture(self):
        """Stop packet capture"""
        self.is_capturing = False
    
    def get_latest_packets(self, max_packets=10):
        """Get latest captured packets from queue"""
        packets = []
        try:
            while len(packets) < max_packets:
                packet = self.packet_queue.get_nowait()
                packets.append(packet)
        except queue.Empty:
            pass
        
        return packets
    
    def get_network_interfaces(self):
        """Get available network interfaces"""
        try:
            return scapy.get_if_list()
        except Exception as e:
            print(f"Error getting interfaces: {e}")
            return []
