"""
CyberLens API Test Suite
Comprehensive testing for the REST API endpoints
"""

import requests
import json
import time
import sys
from typing import Dict, List
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 10

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class APITester:
    """Test suite for CyberLens API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")
    
    def print_test(self, name: str, passed: bool, message: str = ""):
        """Print test result"""
        status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
        print(f"{status} | {name}")
        if message:
            print(f"       {Colors.YELLOW}{message}{Colors.RESET}")
        
        self.results.append((name, passed))
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        percentage = (self.passed / total * 100) if total > 0 else 0
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}Test Summary{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"Total Tests:  {total}")
        print(f"Passed:       {Colors.GREEN}{self.passed}{Colors.RESET}")
        print(f"Failed:       {Colors.RED}{self.failed}{Colors.RESET}")
        print(f"Success Rate: {Colors.BOLD}{percentage:.1f}%{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")
    
    # ========================================================================
    # Test Cases
    # ========================================================================
    
    def test_api_health(self):
        """Test API health endpoint"""
        self.print_header("Test 1: API Health Check")
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/health",
                timeout=API_TIMEOUT
            )
            
            self.print_test(
                "Health endpoint accessible",
                response.status_code == 200,
                f"Status: {response.status_code}"
            )
            
            data = response.json()
            
            self.print_test(
                "Health response has 'status' field",
                'status' in data,
                f"Status: {data.get('status')}"
            )
            
            self.print_test(
                "Model is loaded",
                data.get('model_loaded') == True,
                f"Model loaded: {data.get('model_loaded')}"
            )
            
            self.print_test(
                "Scaler is loaded",
                data.get('scaler_loaded') == True,
                f"Scaler loaded: {data.get('scaler_loaded')}"
            )
        
        except Exception as e:
            self.print_test("Health check", False, str(e))
    
    def test_single_detection(self):
        """Test single threat detection"""
        self.print_header("Test 2: Single Threat Detection")
        
        features = self._get_normal_features()
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/detect",
                json=features,
                timeout=API_TIMEOUT
            )
            elapsed = (time.time() - start_time) * 1000
            
            self.print_test(
                "Detection endpoint accessible",
                response.status_code == 200,
                f"Status: {response.status_code}"
            )
            
            data = response.json()
            
            self.print_test(
                "Response has 'is_threat' field",
                'is_threat' in data,
                f"is_threat: {data.get('is_threat')}"
            )
            
            self.print_test(
                "Response has 'severity' field",
                'severity' in data and data['severity'] in ['HIGH', 'MEDIUM', 'LOW'],
                f"severity: {data.get('severity')}"
            )
            
            self.print_test(
                "Response has 'anomaly_score' field",
                'anomaly_score' in data and isinstance(data['anomaly_score'], (int, float)),
                f"anomaly_score: {data.get('anomaly_score')}"
            )
            
            self.print_test(
                "Response has 'confidence' field",
                'confidence' in data and 0 <= data['confidence'] <= 1,
                f"confidence: {data.get('confidence')}"
            )
            
            self.print_test(
                "Response time < 500ms",
                elapsed < 500,
                f"Processing time: {elapsed:.2f}ms"
            )
            
            self.print_test(
                "Response has 'processing_time_ms' field",
                'processing_time_ms' in data,
                f"processing_time_ms: {data.get('processing_time_ms')}"
            )
        
        except Exception as e:
            self.print_test("Single detection", False, str(e))
    
    def test_batch_detection(self):
        """Test batch threat detection"""
        self.print_header("Test 3: Batch Threat Detection")
        
        records = [
            self._get_normal_features(),
            self._get_suspicious_features(),
            self._get_normal_features(),
        ]
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/detect-batch",
                json={"records": records, "return_details": True},
                timeout=API_TIMEOUT
            )
            elapsed = (time.time() - start_time) * 1000
            
            self.print_test(
                "Batch endpoint accessible",
                response.status_code == 200,
                f"Status: {response.status_code}"
            )
            
            data = response.json()
            
            self.print_test(
                "Response has 'total_records' field",
                data.get('total_records') == len(records),
                f"total_records: {data.get('total_records')}"
            )
            
            self.print_test(
                "Response has 'threats_detected' field",
                'threats_detected' in data and isinstance(data['threats_detected'], int),
                f"threats_detected: {data.get('threats_detected')}"
            )
            
            self.print_test(
                "Response has 'results' field",
                'results' in data and isinstance(data['results'], list),
                f"results count: {len(data.get('results', []))}"
            )
            
            self.print_test(
                "Results count matches records",
                len(data.get('results', [])) == len(records),
                f"Expected {len(records)}, got {len(data.get('results', []))}"
            )
            
            self.print_test(
                "Batch processing time reasonable",
                elapsed < 2000,
                f"Processing time: {elapsed:.2f}ms"
            )
        
        except Exception as e:
            self.print_test("Batch detection", False, str(e))
    
    def test_invalid_input(self):
        """Test error handling with invalid input"""
        self.print_header("Test 4: Invalid Input Handling")
        
        # Missing required fields
        invalid_features = {"duration": 0}
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/detect",
                json=invalid_features,
                timeout=API_TIMEOUT
            )
            
            self.print_test(
                "Invalid input returns error",
                response.status_code != 200,
                f"Status: {response.status_code}"
            )
        
        except Exception as e:
            self.print_test("Invalid input handling", False, str(e))
        
        # Invalid protocol type
        features = self._get_normal_features()
        features['protocol_type'] = 'invalid_protocol'
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/detect",
                json=features,
                timeout=API_TIMEOUT
            )
            
            self.print_test(
                "Invalid protocol type returns error",
                response.status_code != 200,
                f"Status: {response.status_code}"
            )
        
        except Exception as e:
            self.print_test("Invalid protocol handling", False, str(e))
    
    def test_api_stats(self):
        """Test API statistics endpoint"""
        self.print_header("Test 5: API Statistics")
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/stats",
                timeout=API_TIMEOUT
            )
            
            self.print_test(
                "Stats endpoint accessible",
                response.status_code == 200,
                f"Status: {response.status_code}"
            )
            
            data = response.json()
            
            self.print_test(
                "Response has 'model_version' field",
                'model_version' in data,
                f"model_version: {data.get('model_version')}"
            )
            
            self.print_test(
                "Response has 'model_type' field",
                'model_type' in data,
                f"model_type: {data.get('model_type')}"
            )
            
            self.print_test(
                "Response has 'feature_count' field",
                data.get('feature_count') == 42,
                f"feature_count: {data.get('feature_count')}"
            )
        
        except Exception as e:
            self.print_test("API stats", False, str(e))
    
    def test_performance(self):
        """Test API performance"""
        self.print_header("Test 6: Performance Testing")
        
        features = self._get_normal_features()
        times = []
        
        try:
            for i in range(10):
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/api/detect",
                    json=features,
                    timeout=API_TIMEOUT
                )
                elapsed = (time.time() - start_time) * 1000
                times.append(elapsed)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            self.print_test(
                "Average response time < 100ms",
                avg_time < 100,
                f"Average: {avg_time:.2f}ms"
            )
            
            self.print_test(
                "Min response time < 50ms",
                min_time < 50,
                f"Min: {min_time:.2f}ms"
            )
            
            self.print_test(
                "Max response time < 500ms",
                max_time < 500,
                f"Max: {max_time:.2f}ms"
            )
            
            print(f"\nPerformance Metrics:")
            print(f"  Average: {avg_time:.2f}ms")
            print(f"  Min:     {min_time:.2f}ms")
            print(f"  Max:     {max_time:.2f}ms")
            print(f"  Throughput: {1000/(avg_time):.0f} req/sec")
        
        except Exception as e:
            self.print_test("Performance testing", False, str(e))
    
    def test_threat_classification(self):
        """Test threat classification logic"""
        self.print_header("Test 7: Threat Classification")
        
        # Test normal connection
        normal_features = self._get_normal_features()
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/detect",
                json=normal_features,
                timeout=API_TIMEOUT
            )
            data = response.json()
            
            self.print_test(
                "Normal connection classified correctly",
                data['severity'] in ['LOW', 'MEDIUM'],
                f"Severity: {data['severity']}, Score: {data['anomaly_score']:.4f}"
            )
        
        except Exception as e:
            self.print_test("Normal connection classification", False, str(e))
        
        # Test suspicious connection
        suspicious_features = self._get_suspicious_features()
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/detect",
                json=suspicious_features,
                timeout=API_TIMEOUT
            )
            data = response.json()
            
            self.print_test(
                "Suspicious connection detected",
                data['is_threat'] == True,
                f"Is Threat: {data['is_threat']}, Severity: {data['severity']}"
            )
        
        except Exception as e:
            self.print_test("Suspicious connection detection", False, str(e))
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _get_normal_features(self) -> Dict:
        """Get normal connection features"""
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
    
    def _get_suspicious_features(self) -> Dict:
        """Get suspicious connection features"""
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
    
    def run_all_tests(self):
        """Run all tests"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}")
        print("╔" + "═"*68 + "╗")
        print("║" + "CyberLens API Test Suite".center(68) + "║")
        print("║" + f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(68) + "║")
        print("╚" + "═"*68 + "╝")
        print(Colors.RESET)
        
        self.test_api_health()
        self.test_single_detection()
        self.test_batch_detection()
        self.test_invalid_input()
        self.test_api_stats()
        self.test_performance()
        self.test_threat_classification()
        
        self.print_summary()
        
        return self.failed == 0


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CyberLens API Test Suite")
    print("="*70)
    print("\nMake sure the API is running:")
    print("  python api.py")
    print("\nOr:")
    print("  uvicorn api:app --reload --host 0.0.0.0 --port 8000")
    print("\n" + "="*70 + "\n")
    
    tester = APITester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)
