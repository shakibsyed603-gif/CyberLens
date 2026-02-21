"""
CyberLens API Server - REST API for Real-time Threat Detection
Provides endpoints for anomaly detection using the trained Isolation Forest model
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
import uvicorn
import threading
import time
from typing import Any
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import MODEL_PATH, SCALER_PATH, PROCESSED_DATA_DIR, MODEL_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class NetworkFeatures(BaseModel):
    """Network features matching KDD dataset structure"""
    duration: float = Field(..., description="Connection duration in seconds")
    protocol_type: str = Field(..., description="Protocol type (tcp, udp, icmp)")
    service: str = Field(..., description="Network service (http, ftp, etc.)")
    flag: str = Field(..., description="Connection status flag")
    src_bytes: float = Field(..., description="Bytes sent from source")
    dst_bytes: float = Field(..., description="Bytes sent to destination")
    land: int = Field(0, description="1 if connection is from/to same host")
    wrong_fragment: int = Field(0, description="Number of wrong fragments")
    urgent: int = Field(0, description="Number of urgent packets")
    hot: int = Field(0, description="Number of hot indicators")
    num_failed_logins: int = Field(0, description="Number of failed login attempts")
    logged_in: int = Field(0, description="1 if successfully logged in")
    num_compromised: int = Field(0, description="Number of compromised conditions")
    root_shell: int = Field(0, description="1 if root shell is obtained")
    su_attempted: int = Field(0, description="1 if su command attempted")
    num_root: int = Field(0, description="Number of root accesses")
    num_file_creations: int = Field(0, description="Number of file creation operations")
    num_shells: int = Field(0, description="Number of shell prompts")
    num_access_files: int = Field(0, description="Number of access to file operations")
    num_outbound_cmds: int = Field(0, description="Number of outbound commands")
    is_host_login: int = Field(0, description="1 if the login belongs to the host list")
    is_guest_login: int = Field(0, description="1 if the login is a guest login")
    count: int = Field(..., description="Number of connections to same host")
    srv_count: int = Field(..., description="Number of connections to same service")
    serror_rate: float = Field(..., description="% of connections with SYN errors")
    srv_serror_rate: float = Field(..., description="% of connections with SYN errors to service")
    rerror_rate: float = Field(..., description="% of connections with REJ errors")
    srv_rerror_rate: float = Field(..., description="% of connections with REJ errors to service")
    same_srv_rate: float = Field(..., description="% of connections to same service")
    diff_srv_rate: float = Field(..., description="% of connections to different services")
    srv_diff_host_rate: float = Field(..., description="% of connections to different hosts")
    dst_host_count: int = Field(..., description="Count of connections to destination host")
    dst_host_srv_count: int = Field(..., description="Count of connections to destination host/service")
    dst_host_same_srv_rate: float = Field(..., description="% of connections to same service")
    dst_host_diff_srv_rate: float = Field(..., description="% of connections to different services")
    dst_host_same_src_port_rate: float = Field(..., description="% of connections from same source port")
    dst_host_srv_diff_host_rate: float = Field(..., description="% of connections to different hosts")
    dst_host_serror_rate: float = Field(..., description="% of connections with SYN errors")
    dst_host_srv_serror_rate: float = Field(..., description="% of connections with SYN errors to service")
    dst_host_rerror_rate: float = Field(..., description="% of connections with REJ errors")
    dst_host_srv_rerror_rate: float = Field(..., description="% of connections with REJ errors to service")

    @validator('protocol_type')
    def validate_protocol(cls, v):
        valid_protocols = ['tcp', 'udp', 'icmp']
        if v.lower() not in valid_protocols:
            raise ValueError(f'Protocol must be one of {valid_protocols}')
        return v.lower()

    @validator('duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count', pre=True)
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v


class ThreatDetectionResponse(BaseModel):
    """Response model for threat detection"""
    is_threat: bool = Field(..., description="Whether the connection is flagged as a threat")
    severity: str = Field(..., description="Threat severity level: HIGH, MEDIUM, LOW")
    anomaly_score: float = Field(..., description="Anomaly score (-1 to 1, lower = more anomalous)")
    confidence: float = Field(..., description="Confidence score (0 to 1)")
    timestamp: str = Field(..., description="ISO format timestamp of detection")
    processing_time_ms: float = Field(..., description="Time taken to process request in milliseconds")
    model_version: str = Field(..., description="Version of the model used")


class BatchDetectionRequest(BaseModel):
    """Request model for batch threat detection"""
    records: List[NetworkFeatures] = Field(..., description="List of network records to analyze")
    return_details: bool = Field(False, description="Include detailed anomaly scores")


class BatchDetectionResponse(BaseModel):
    """Response model for batch detection"""
    total_records: int = Field(..., description="Total records processed")
    threats_detected: int = Field(..., description="Number of threats detected")
    processing_time_ms: float = Field(..., description="Total processing time")
    results: List[Dict] = Field(..., description="Detection results for each record")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    scaler_loaded: bool = Field(..., description="Whether the scaler is loaded")
    timestamp: str = Field(..., description="Current timestamp")


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages model and scaler loading"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_version = "1.0"
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model file not found at {MODEL_PATH}")
                return False
            
            if not os.path.exists(SCALER_PATH):
                logger.error(f"Scaler file not found at {SCALER_PATH}")
                return False
            
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            
            # Load feature columns from processed data
            processed_data_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
            if os.path.exists(processed_data_path):
                data = joblib.load(processed_data_path)
                if isinstance(data, dict) and 'feature_columns' in data:
                    self.feature_columns = data['feature_columns']
            
            logger.info("Model and scaler loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if model and scaler are loaded"""
        return self.model is not None and self.scaler is not None
    
    def predict(self, features_dict: Dict) -> Dict:
        """
        Predict anomaly for given features
        
        Args:
            features_dict: Dictionary of network features
        
        Returns:
            Dictionary with is_threat, severity, and anomaly_score
        """
        if not self.is_ready():
            raise RuntimeError("Model or scaler not loaded")
        
        try:
            # Convert to DataFrame for consistent processing
            df = pd.DataFrame([features_dict])
            
            # Get feature columns in correct order
            if self.feature_columns:
                feature_cols = [col for col in self.feature_columns if col in df.columns]
            else:
                # Fallback: use all numeric columns except label columns
                exclude = ['Label', 'is_anomaly', 'difficulty']
                feature_cols = [col for col in df.columns if col not in exclude]
            
            # Scale features
            X = self.scaler.transform(df[feature_cols])
            
            # Get anomaly score (use decision_function for consistency with app)
            anomaly_score = self.model.decision_function(X)[0]
            
            # Classify threat level
            # FIXED: Correct confidence calculation based on anomaly score range
            if anomaly_score < -0.7:
                severity = "HIGH"
                # Confidence: more negative = more confident it's an anomaly
                confidence = min(1.0, max(0.0, abs(anomaly_score) * 1.2))
            elif anomaly_score < -0.3:
                severity = "MEDIUM"
                confidence = min(1.0, max(0.0, abs(anomaly_score) * 0.9))
            else:
                severity = "LOW"
                confidence = min(1.0, max(0.0, abs(anomaly_score) * 0.5))
            
            is_threat = severity in ["HIGH", "MEDIUM"]
            
            return {
                "is_threat": is_threat,
                "severity": severity,
                "anomaly_score": float(anomaly_score),
                "confidence": float(confidence)
            }
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="CyberLens API",
    description="REST API for real-time network threat detection using Isolation Forest",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve a tiny static folder for assets (favicon etc.)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    try:
        os.makedirs(static_dir, exist_ok=True)
    except Exception:
        # best-effort: if creating the dir fails, continue without static files
        pass

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return a small favicon if present in the static folder."""
    svg_path = os.path.join(static_dir, "favicon.svg")
    ico_path = os.path.join(static_dir, "favicon.ico")
    png_path = os.path.join(static_dir, "favicon.png")
    if os.path.exists(ico_path):
        return FileResponse(ico_path, media_type="image/x-icon")
    if os.path.exists(svg_path):
        return FileResponse(svg_path, media_type="image/svg+xml")
    if os.path.exists(png_path):
        return FileResponse(png_path, media_type="image/png")
    raise HTTPException(status_code=404)

# Initialize model manager
model_manager = ModelManager()


# ==========================================================================
# Background Training Manager
# ==========================================================================

class TrainingManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_training = False
        self.current_round = 0
        self.total_rounds = 10
        self.accuracy = 0.0
        self.mae = 0.0
        self.rmse = 0.0
        self.status = "Idle"
        self._stop_flag = False
        self._thread = None

    def start(self):
        with self.lock:
            if self.is_training:
                return False
            self.is_training = True
            self._stop_flag = False
            self.current_round = 0
            self.status = "Training Started"
            self._thread = threading.Thread(target=self._run_training, daemon=True)
            self._thread.start()
            return True

    def stop(self):
        with self.lock:
            if not self.is_training:
                return False
            self._stop_flag = True
            self.status = "Stopping"
            return True

    def reset(self):
        with self.lock:
            self._stop_flag = True
            self.is_training = False
            self.current_round = 0
            self.accuracy = 0.0
            self.mae = 0.0
            self.rmse = 0.0
            self.status = "Reset"
            return True

    def get_status(self) -> dict:
        with self.lock:
            return {
                "round": self.current_round,
                "total_rounds": self.total_rounds,
                "accuracy": round(self.accuracy, 4),
                "mae": round(self.mae, 4),
                "rmse": round(self.rmse, 4),
                "is_training": self.is_training,
                "status": self.status,
            }

    def _run_training(self):
        try:
            processed_data_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
            if not os.path.exists(processed_data_path):
                with self.lock:
                    self.status = "Processed data not found"
                    self.is_training = False
                return

            data = joblib.load(processed_data_path)
            X = data['X']
            y = data['y']

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            normal_data = X_train[y_train == 0]

            for r in range(1, self.total_rounds + 1):
                with self.lock:
                    if self._stop_flag:
                        self.status = "Stopped"
                        self.is_training = False
                        return
                    self.current_round = r
                    self.status = f"Training round {r}/{self.total_rounds}"

                from sklearn.ensemble import IsolationForest
                cfg = MODEL_CONFIG.copy()
                cfg['random_state'] = cfg.get('random_state', 42) + r
                model = IsolationForest(**cfg)
                model.fit(normal_data)

                y_pred = model.predict(X_test)
                y_pred_binary = (y_pred == -1).astype(int)

                acc = float(accuracy_score(y_test, y_pred_binary))
                mae = float(mean_absolute_error(y_test, y_pred_binary))
                rmse = float(mean_squared_error(y_test, y_pred_binary, squared=False))

                with self.lock:
                    self.accuracy = acc
                    self.mae = mae
                    self.rmse = rmse

                if r == self.total_rounds:
                    try:
                        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                        joblib.dump(model, MODEL_PATH)
                    except Exception:
                        pass

                # brief pause to simulate longer work and allow UI polling
                for _ in range(10):
                    time.sleep(0.2)
                    with self.lock:
                        if self._stop_flag:
                            self.status = "Stopped"
                            self.is_training = False
                            return

            with self.lock:
                self.status = "Training Completed"
                self.is_training = False

        except Exception as e:
            with self.lock:
                self.status = f"Error: {str(e)}"
                self.is_training = False


training_manager = TrainingManager()



# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthResponse with service status and model availability
    """
    return HealthResponse(
        status="healthy" if model_manager.is_ready() else "degraded",
        model_loaded=model_manager.model is not None,
        scaler_loaded=model_manager.scaler is not None,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/api/detect", response_model=ThreatDetectionResponse, tags=["Detection"])
async def detect_threat(features: NetworkFeatures):
    """
    Detect threats in a single network connection
    
    This endpoint analyzes a single network connection and returns whether it's
    flagged as a threat based on the Isolation Forest model.
    
    Args:
        features: NetworkFeatures object with 42 network features
    
    Returns:
        ThreatDetectionResponse with threat status and anomaly score
    
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if not model_manager.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files exist and restart the API."
        )
    
    try:
        import time
        start_time = time.time()
        
        # Convert Pydantic model to dict
        features_dict = features.dict()
        
        # Get prediction
        prediction = model_manager.predict(features_dict)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return ThreatDetectionResponse(
            is_threat=prediction["is_threat"],
            severity=prediction["severity"],
            anomaly_score=prediction["anomaly_score"],
            confidence=prediction["confidence"],
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=processing_time,
            model_version=model_manager.model_version
        )
    
    except Exception as e:
        logger.error(f"Error in threat detection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.post("/api/detect-batch", response_model=BatchDetectionResponse, tags=["Detection"])
async def detect_threats_batch(request: BatchDetectionRequest):
    """
    Detect threats in multiple network connections (batch processing)
    
    This endpoint analyzes multiple network connections in a single request.
    Useful for processing large datasets or periodic batch analysis.
    
    Args:
        request: BatchDetectionRequest with list of network features
    
    Returns:
        BatchDetectionResponse with aggregated results
    
    Raises:
        HTTPException: If model is not loaded or batch processing fails
    """
    if not model_manager.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files exist and restart the API."
        )
    
    if not request.records:
        raise HTTPException(
            status_code=400,
            detail="No records provided in request"
        )
    
    if len(request.records) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 10000 records"
        )
    
    try:
        import time
        start_time = time.time()
        
        results = []
        threats_count = 0
        
        for idx, features in enumerate(request.records):
            try:
                features_dict = features.dict()
                prediction = model_manager.predict(features_dict)
                
                result = {
                    "record_index": idx,
                    "is_threat": prediction["is_threat"],
                    "severity": prediction["severity"],
                    "confidence": prediction["confidence"]
                }
                
                if request.return_details:
                    result["anomaly_score"] = prediction["anomaly_score"]
                
                if prediction["is_threat"]:
                    threats_count += 1
                
                results.append(result)
            
            except Exception as e:
                logger.error(f"Error processing record {idx}: {e}")
                results.append({
                    "record_index": idx,
                    "error": str(e)
                })
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchDetectionResponse(
            total_records=len(request.records),
            threats_detected=threats_count,
            processing_time_ms=processing_time,
            results=results
        )
    
    except Exception as e:
        logger.error(f"Error in batch threat detection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch: {str(e)}"
        )


@app.post("/api/train-model", tags=["Training"])
async def start_training(background_tasks: BackgroundTasks):
    """Start a background training job that runs for multiple rounds.

    Returns immediately with current training status.
    """
    started = training_manager.start()
    if not started:
        return {"status": "Already running", **training_manager.get_status()}

    return {"status": "Started", **training_manager.get_status()}


@app.post("/api/train-stop", tags=["Training"])
async def stop_training():
    stopped = training_manager.stop()
    if not stopped:
        raise HTTPException(status_code=400, detail="Training not running")
    return {"status": "Stopping", **training_manager.get_status()}


@app.post("/api/train-reset", tags=["Training"])
async def reset_training():
    training_manager.reset()
    return {"status": "Reset", **training_manager.get_status()}


@app.get("/api/train-status", tags=["Training"])
async def train_status():
    return training_manager.get_status()



@app.get("/api/stats", tags=["Info"])
async def get_stats():
    """
    Get API statistics and model information
    
    Returns:
        Dictionary with model info and API statistics
    """
    return {
        "model_version": model_manager.model_version,
        "model_type": "Isolation Forest",
        "model_loaded": model_manager.is_ready(),
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "feature_count": 42,
        "dataset": "KDD Cup 1999",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "CyberLens API",
        "version": "1.0.0",
        "description": "REST API for real-time network threat detection",
        "docs": "/api/docs",
        "health": "/api/health",
        "endpoints": {
            "detect": "/api/detect (POST)",
            "detect_batch": "/api/detect-batch (POST)",
            "health": "/api/health (GET)",
            "stats": "/api/stats (GET)"
        }
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("CyberLens API starting up...")
    if model_manager.is_ready():
        logger.info("Model and scaler loaded successfully")
    else:
        logger.warning("Model or scaler not loaded - API will return 503 errors")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("CyberLens API shutting down...")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run with: python api.py
    # Or: uvicorn api:app --reload --host 0.0.0.0 --port 8000
    
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting CyberLens API on {host}:{port}")
    logger.info(f"Swagger docs available at http://localhost:{port}/api/docs")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
