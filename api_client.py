"""
CyberLens API Client - Example usage and utilities for interacting with the API
"""

import requests
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CyberLensAPIClient:
    """Client for interacting with CyberLens API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client
        
        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "CyberLens-Client/1.0"
        })
    
    def health_check(self) -> Dict:
        """
        Check API health status
        
        Returns:
            Dictionary with health status
        """
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def detect_threat(self, features: Dict) -> Dict:
        """
        Detect threat in a single network connection
        
        Args:
            features: Dictionary of network features
        
        Returns:
            Detection result with threat status and anomaly score
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/detect",
                json=features,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Threat detection failed: {e}")
            return {"error": str(e)}
    
    def detect_threats_batch(self, records: List[Dict], return_details: bool = False) -> Dict:
        """
        Detect threats in multiple network connections
        
        Args:
            records: List of network feature dictionaries
            return_details: Whether to include detailed anomaly scores
        
        Returns:
            Batch detection results
        """
        try:
            payload = {
                "records": records,
                "return_details": return_details
            }
            response = self.session.post(
                f"{self.base_url}/api/detect-batch",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch detection failed: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict:
        """
        Get API statistics and model information
        
        Returns:
            API statistics
        """
        try:
            response = self.session.get(f"{self.base_url}/api/stats")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


# ============================================================================
# Example Network Features (KDD Dataset Format)
# ============================================================================

def create_normal_connection() -> Dict:
    """Create example normal network connection"""
    return {
        "duration": 0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "src_bytes": 181,
        "dst_bytes": 5450,
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 1,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": 8,
        "srv_count": 8,
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,
        "dst_host_count": 9,
        "dst_host_srv_count": 9,
        "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.11,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    }


def create_suspicious_connection() -> Dict:
    """Create example suspicious network connection"""
    return {
        "duration": 0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "S0",
        "src_bytes": 0,
        "dst_bytes": 0,
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 5,
        "logged_in": 0,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 1,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": 1,
        "srv_count": 1,
        "serror_rate": 1.0,
        "srv_serror_rate": 1.0,
        "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0,
        "same_srv_rate": 0.0,
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,
        "dst_host_count": 1,
        "dst_host_srv_count": 1,
        "dst_host_same_srv_rate": 0.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 1.0,
        "dst_host_srv_serror_rate": 1.0,
        "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    }


# ============================================================================
# Example Usage Functions
# ============================================================================

def example_single_detection():
    """Example: Detect threat in a single connection"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Connection Threat Detection")
    print("="*70)
    
    client = CyberLensAPIClient()
    
    # Check health first
    print("\n1. Checking API health...")
    health = client.health_check()
    print(f"   Status: {health.get('status')}")
    print(f"   Model loaded: {health.get('model_loaded')}")
    
    # Test with normal connection
    print("\n2. Testing with NORMAL connection...")
    normal_conn = create_normal_connection()
    result = client.detect_threat(normal_conn)
    print(f"   Is Threat: {result.get('is_threat')}")
    print(f"   Severity: {result.get('severity')}")
    print(f"   Anomaly Score: {result.get('anomaly_score'):.4f}")
    print(f"   Confidence: {result.get('confidence'):.4f}")
    print(f"   Processing Time: {result.get('processing_time_ms'):.2f}ms")
    
    # Test with suspicious connection
    print("\n3. Testing with SUSPICIOUS connection...")
    suspicious_conn = create_suspicious_connection()
    result = client.detect_threat(suspicious_conn)
    print(f"   Is Threat: {result.get('is_threat')}")
    print(f"   Severity: {result.get('severity')}")
    print(f"   Anomaly Score: {result.get('anomaly_score'):.4f}")
    print(f"   Confidence: {result.get('confidence'):.4f}")
    print(f"   Processing Time: {result.get('processing_time_ms'):.2f}ms")


def example_batch_detection():
    """Example: Detect threats in multiple connections"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Threat Detection")
    print("="*70)
    
    client = CyberLensAPIClient()
    
    # Create batch of mixed connections
    print("\n1. Creating batch of 5 connections (3 normal, 2 suspicious)...")
    records = [
        create_normal_connection(),
        create_normal_connection(),
        create_normal_connection(),
        create_suspicious_connection(),
        create_suspicious_connection(),
    ]
    
    print(f"   Total records: {len(records)}")
    
    # Send batch request
    print("\n2. Sending batch detection request...")
    result = client.detect_threats_batch(records, return_details=True)
    
    print(f"   Total records processed: {result.get('total_records')}")
    print(f"   Threats detected: {result.get('threats_detected')}")
    print(f"   Processing time: {result.get('processing_time_ms'):.2f}ms")
    
    # Show detailed results
    print("\n3. Detailed results:")
    for res in result.get('results', []):
        idx = res.get('record_index')
        threat = res.get('is_threat')
        severity = res.get('severity')
        score = res.get('anomaly_score', 'N/A')
        print(f"   Record {idx}: {'🔴 THREAT' if threat else '🟢 NORMAL'} | "
              f"Severity: {severity} | Score: {score}")


def example_performance_test():
    """Example: Test API performance"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Performance Test")
    print("="*70)
    
    client = CyberLensAPIClient()
    
    # Test single request performance
    print("\n1. Testing single request performance (10 requests)...")
    times = []
    for i in range(10):
        conn = create_normal_connection()
        start = time.time()
        result = client.detect_threat(conn)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"   Request {i+1}: {elapsed:.2f}ms")
    
    print(f"\n   Average: {sum(times)/len(times):.2f}ms")
    print(f"   Min: {min(times):.2f}ms")
    print(f"   Max: {max(times):.2f}ms")
    
    # Test batch performance
    print("\n2. Testing batch performance (100 records)...")
    records = [create_normal_connection() for _ in range(100)]
    start = time.time()
    result = client.detect_threats_batch(records)
    elapsed = (time.time() - start) * 1000
    
    print(f"   Total time: {elapsed:.2f}ms")
    print(f"   Time per record: {elapsed/100:.2f}ms")
    print(f"   Throughput: {100/(elapsed/1000):.0f} records/sec")


def example_api_stats():
    """Example: Get API statistics"""
    print("\n" + "="*70)
    print("EXAMPLE 4: API Statistics")
    print("="*70)
    
    client = CyberLensAPIClient()
    
    print("\n1. Fetching API statistics...")
    stats = client.get_stats()
    
    print(f"   Model Version: {stats.get('model_version')}")
    print(f"   Model Type: {stats.get('model_type')}")
    print(f"   Model Loaded: {stats.get('model_loaded')}")
    print(f"   Feature Count: {stats.get('feature_count')}")
    print(f"   Dataset: {stats.get('dataset')}")


# ============================================================================
# Integration Example: Real-time Monitoring
# ============================================================================

def example_real_time_monitoring():
    """Example: Simulate real-time monitoring"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Real-time Monitoring Simulation")
    print("="*70)
    
    client = CyberLensAPIClient()
    
    print("\nSimulating real-time network monitoring (20 connections)...")
    print("Timestamp          | Type      | Severity | Score    | Status")
    print("-" * 70)
    
    threat_count = 0
    normal_count = 0
    
    for i in range(20):
        # Randomly choose normal or suspicious
        if i % 4 == 0:  # 25% suspicious
            conn = create_suspicious_connection()
            conn_type = "Suspicious"
        else:
            conn = create_normal_connection()
            conn_type = "Normal"
        
        result = client.detect_threat(conn)
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        severity = result.get('severity', 'N/A')
        score = result.get('anomaly_score', 0)
        is_threat = result.get('is_threat', False)
        
        status = "🔴 ALERT" if is_threat else "🟢 OK"
        
        print(f"{timestamp} | {conn_type:9} | {severity:8} | {score:8.4f} | {status}")
        
        if is_threat:
            threat_count += 1
        else:
            normal_count += 1
        
        time.sleep(0.1)  # Small delay between requests
    
    print("-" * 70)
    print(f"Summary: {threat_count} threats detected, {normal_count} normal connections")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CyberLens API Client - Example Usage")
    print("="*70)
    print("\nMake sure the API server is running:")
    print("  python api.py")
    print("  OR")
    print("  uvicorn api:app --reload --host 0.0.0.0 --port 8000")
    
    try:
        # Run examples
        example_single_detection()
        example_batch_detection()
        example_performance_test()
        example_api_stats()
        example_real_time_monitoring()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")
    
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"\nError: {e}")
        print("\nMake sure the API server is running on http://localhost:8000")
