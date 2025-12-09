from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os
import time
from collections import deque
from datetime import datetime
from windowing_features import EMGBuffer, get_feature_names

# ============================================== Informations =========================================================
# Commande to run server: python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
# Doc auto generated http://127.0.0.1:8000/docs
# POST http://127.0.0.1:8000/predict
# BODY {"features": {"EMG1": 51, "EMG2": 8, "EMG3": 2, "EMG4": -3, "EMG5": -25, "EMG6": 12, "EMG7": -7, "EMG8": -26}}
#
# For windowed models:
# - Client must send samples at consistent rate (e.g., every 20ms for 50Hz sampling)
# - Each client gets a session_id to maintain separate buffers
# - POST to /predict_windowed with session_id in body
# - First N samples will return "buffering" until window is full
# =====================================================================================================================


# ===== Load model artifacts =====
base_dir = "models"
models = sorted(os.listdir(base_dir))
if not models:
    print(f"\033[91m> Error: No model found in '{base_dir}' directory\033[0m")
    exit(1)

# Use the latest model directory
latest_model = models[-1]
MODEL_DIR = f"{base_dir}/{latest_model}"
print(f"\033[95m> Loading models from: {MODEL_DIR}\033[0m")

# Try to load gate model (for two-stage prediction)
gate_pipe = None
gate_cfg = None
if os.path.exists(f"{MODEL_DIR}/gate_pipeline.joblib"):
    print(f"\033[96m  - Loading gate model (Rest vs Gesture)\033[0m")
    gate_pipe = joblib.load(f"{MODEL_DIR}/gate_pipeline.joblib")
    with open(f"{MODEL_DIR}/gate_config.json") as f:
        gate_cfg = json.load(f)
    print(f"\033[92m  ✓ Gate model loaded\033[0m")
else:
    print(f"\033[93m  ⚠ No gate model found - running single-stage prediction\033[0m")

# Load gesture classifier model
if os.path.exists(f"{MODEL_DIR}/gesture_pipeline.joblib"):
    print(f"\033[96m  - Loading gesture classifier\033[0m")
    pipe = joblib.load(f"{MODEL_DIR}/gesture_pipeline.joblib")
    with open(f"{MODEL_DIR}/gesture_config.json") as f:
        cfg = json.load(f)
    print(f"\033[92m  ✓ Gesture classifier loaded\033[0m")
else:
    # Fallback to old naming convention (pipeline.joblib)
    print(f"\033[96m  - Loading model (legacy format)\033[0m")
    pipe = joblib.load(f"{MODEL_DIR}/pipeline.joblib")
    with open(f"{MODEL_DIR}/config.json") as f:
        cfg = json.load(f)
    print(f"\033[92m  ✓ Model loaded\033[0m")

if gate_pipe:
    print(f"\033[94m> Two-stage prediction enabled: Gate → Gesture Classifier\033[0m")
else:
    print(f"\033[94m> Single-stage prediction (no gate model)\033[0m")


# ======= Define request schema =======
class PredictRequest(BaseModel):
    features: dict

class PredictWindowedRequest(BaseModel):
    features: dict
    session_id: str  # Unique identifier for each client session

class BatchPredictWindowedRequest(BaseModel):
    batch: list[dict]  # List of EMG samples (ordered chronologically)
    session_id: str  # Unique identifier for each client session

class BatchPredictRequest(BaseModel):
    batch: list[dict]  # List of feature dictionaries


# ======= Create FastAPI app =======
app = FastAPI(title="Gesture Classifier API")

# ======= Request statistics tracking =======
request_stats = {
    "predict": deque(maxlen=100),  # Last 100 requests
    "predict_batch": deque(maxlen=100),
    "predict_windowed": deque(maxlen=100),
    "batch_predict_windowed": deque(maxlen=100),
    "total_requests": 0,
    "total_samples_processed": 0
}

# ======= Session management for windowed predictions =======
# Each client session maintains its own EMG buffer and maxEMG for normalization
session_buffers = {}  # {session_id: EMGBuffer}
session_max_emg = {}  # {session_id: float} - for normalizing features

def get_or_create_buffer(session_id: str) -> EMGBuffer:
    """Get existing buffer for session or create new one."""
    if session_id not in session_buffers:
        window_size = cfg.get("window_size", 10)
        session_buffers[session_id] = EMGBuffer(window_size=window_size)
    return session_buffers[session_id]

@app.middleware("http")
async def add_timing_stats(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000  # Convert to ms
    
    # Log request with timing
    endpoint = request.url.path
    if endpoint in ["/predict", "/predict_batch", "/predict_windowed", "/batch_predict_windowed"]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {endpoint}: {duration:.2f}ms")
        
        # Track stats
        request_stats["total_requests"] += 1
        if endpoint == "/predict":
            request_stats["predict"].append(duration)
            request_stats["total_samples_processed"] += 1
        elif endpoint == "/predict_batch":
            request_stats["predict_batch"].append(duration)
        elif endpoint == "/predict_windowed":
            request_stats["predict_windowed"].append(duration)
            request_stats["total_samples_processed"] += 1
        elif endpoint == "/batch_predict_windowed":
            request_stats["batch_predict_windowed"].append(duration)
    
    return response


def process_prediction(probs, cfg, features=None, raw_emg_features=None):
    """
    Two-stage prediction: Gate model (rest vs gesture) -> Gesture classifier
    
    Args:
        probs: Model prediction probabilities from gesture classifier
        cfg: Model configuration
        features: Optional - extracted features dict (for windowed models)
        raw_emg_features: Optional - raw EMG dict for gate model (for non-windowed)
    """
    
    # STAGE 1: Gate model - check if user is at rest or attempting a gesture
    if gate_pipe is not None:
        # Prepare features for gate model
        if raw_emg_features is not None:
            # Non-windowed: use raw EMG values
            gate_features = [raw_emg_features.get(f'EMG{i}', 0) for i in range(1, 9)]
        elif features is not None:
            # Windowed: use MAV features (mean absolute value represents intensity)
            gate_features = [features.get(f'EMG{ch}_MAV', 0) for ch in range(1, 9)]
        else:
            # Fallback: can't use gate model without features
            gate_features = None
        
        if gate_features is not None:
            # Predict with gate model
            gate_X = pd.DataFrame([gate_features], columns=gate_cfg["feature_names"])
            gate_probs = gate_pipe.predict_proba(gate_X)[0]
            gate_pred = gate_pipe.predict(gate_X)[0]
            
            rest_prob = float(gate_probs[0])  # Probability of Rest
            gesture_prob = float(gate_probs[1])  # Probability of Gesture
            
            # If gate model predicts REST, return Neutral immediately
            # Use a threshold to balance sensitivity (default: 0.5)
            gate_threshold = cfg.get("gate_threshold", 0.5)
            
            if rest_prob > gate_threshold:
                return {
                    "label": "Neutral",
                    "prob": rest_prob,
                    "topk": [{"label": "Neutral", "prob": rest_prob}],
                    "detection_method": "gate_model",
                    "gate_rest_prob": rest_prob,
                    "gate_gesture_prob": gesture_prob
                }
            
            # Gate detected gesture activity, proceed to gesture classifier
            # (Continue to STAGE 2 below)
    
    # STAGE 2: Gesture classifier - identify specific gesture
    top_idx = probs.argmax()
    top_prob = float(probs[top_idx])
    predicted_label = cfg["class_names"][top_idx]
    
    label = predicted_label

    # Not confident
    if top_prob < cfg.get("abstain_threshold", 0.90):
        label = "Unknown"
    
    # Top-K
    top_k = cfg.get("top_k", 2)
    topk_idx = probs.argsort()[-top_k:][::-1]
    topk = [{"label": cfg["class_names"][i], "prob": float(probs[i])} for i in topk_idx]
    
    result = {
        "label": label, 
        "prob": top_prob, 
        "topk": topk,
        "detection_method": "gesture_model"
    }
    
    # Add gate model info if available
    if gate_pipe is not None and 'gesture_prob' in locals():
        result["gate_rest_prob"] = rest_prob
        result["gate_gesture_prob"] = gesture_prob
    
    return result


# ======= Define API endpoints =======
@app.post("/predict")
def predict(req: PredictRequest):
    """
    Single prediction endpoint for non-windowed models.
    Two-stage: Gate model (rest detection) -> Gesture classifier
    """
    check_param(req)

    # Reorder features according to the model's expectations
    ordered_features = [req.features[f] for f in cfg["feature_names"]]
    X = pd.DataFrame([ordered_features], columns=cfg["feature_names"])
    probs = pipe.predict_proba(X)[0]
    
    return process_prediction(probs, cfg, raw_emg_features=req.features)


@app.post("/predict_windowed")
def predict_windowed(req: PredictWindowedRequest):
    """
    Windowed prediction endpoint for temporal feature models.
    Buffers samples per session and extracts features from windows.
    
    Client requirements:
    - Send samples at consistent rate (e.g., 50Hz = every 20ms)
    - Include unique session_id to maintain buffer state
    - First N samples will return status="buffering" until window fills
    """
    # Check if model uses windowing
    if not cfg.get("uses_windowing", False):
        raise HTTPException(
            status_code=400, 
            detail="This endpoint requires a windowed model. Use /predict for non-windowed models."
        )
    
    # Validate features
    emg_channels = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8']
    for ch in emg_channels:
        if ch not in req.features:
            raise HTTPException(status_code=422, detail=f"Missing feature: {ch}")
    
    # Get or create buffer for this session
    buffer = get_or_create_buffer(req.session_id)
    
    # Add sample to buffer
    buffer.add_sample(req.features)
    
    # Check if buffer is ready
    if not buffer.is_ready():
        samples_needed = cfg.get("window_size", 10) - len(buffer.buffer)
        return {
            "status": "buffering",
            "message": f"Collecting samples... need {samples_needed} more",
            "buffer_size": len(buffer.buffer),
            "window_size": cfg.get("window_size", 10)
        }
    
    # Extract features from window
    features = buffer.extract_features()
    
    # Normalize features by session's maxEMG if available
    # This makes EMG intensity comparable across different calibration sessions
    if req.session_id in session_max_emg:
        max_emg = session_max_emg[req.session_id]
        # Normalize intensity-based features (MAV, WL)
        for ch in range(1, 9):
            if f'EMG{ch}_MAV' in features:
                features[f'EMG{ch}_MAV'] /= max_emg
            if f'EMG{ch}_WL' in features:
                features[f'EMG{ch}_WL'] /= max_emg
    
    # Reorder features according to model's expectations
    ordered_features = [features[f] for f in cfg["feature_names"]]
    X = pd.DataFrame([ordered_features], columns=cfg["feature_names"])
    probs = pipe.predict_proba(X)[0]
    
    result = process_prediction(probs, cfg, features=features)  # Pass features for rest detection
    result["status"] = "predicted"
    return result


@app.post("/batch_predict_windowed")
def batch_predict_windowed(req: BatchPredictWindowedRequest):
    """
    Batch windowed prediction endpoint - processes multiple samples as a stream.
    Maintains temporal continuity by adding samples to buffer sequentially.
    
    Use this for efficient batch processing while maintaining windowing benefits.
    Client sends multiple consecutive EMG samples in chronological order.
    
    Returns predictions for each sample that has a full window available.
    """
    # Check if model uses windowing
    if not cfg.get("uses_windowing", False):
        raise HTTPException(
            status_code=400, 
            detail="This endpoint requires a windowed model. Use /predict_batch for non-windowed models."
        )
    
    if not req.batch:
        raise HTTPException(status_code=422, detail="Batch cannot be empty.")
    
    # Validate features
    emg_channels = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8']
    for idx, sample in enumerate(req.batch):
        for ch in emg_channels:
            if ch not in sample:
                raise HTTPException(
                    status_code=422, 
                    detail=f"Missing feature '{ch}' in batch item {idx}"
                )
    
    # Get or create buffer for this session
    buffer = get_or_create_buffer(req.session_id)
    window_size = cfg.get("window_size", 10)
    
    predictions = []
    batch_size = len(req.batch)
    request_stats["total_samples_processed"] += batch_size
    
    # Collect all features for ready samples first (more efficient)
    features_to_predict = []
    sample_indices = []
    
    # Process each sample in the batch sequentially (as a stream)
    for idx, sample in enumerate(req.batch):
        # Add sample to buffer
        buffer.add_sample(sample)
        
        # Check if buffer is ready for this sample
        if buffer.is_ready():
            # Extract features from current window
            features = buffer.extract_features()
            features_to_predict.append(features)
            sample_indices.append(idx)
        else:
            # Still buffering for this sample
            samples_needed = window_size - len(buffer.buffer)
            predictions.append({
                "status": "buffering",
                "sample_index": idx,
                "samples_needed": samples_needed,
                "buffer_size": len(buffer.buffer)
            })
    
    # Batch predict all ready samples at once (MUCH faster)
    if features_to_predict:
        # Normalize features by session's maxEMG if available
        if req.session_id in session_max_emg:
            max_emg = session_max_emg[req.session_id]
            for features in features_to_predict:
                # Normalize intensity-based features (MAV, WL)
                for ch in range(1, 9):
                    if f'EMG{ch}_MAV' in features:
                        features[f'EMG{ch}_MAV'] /= max_emg
                    if f'EMG{ch}_WL' in features:
                        features[f'EMG{ch}_WL'] /= max_emg
        
        # Build feature matrix for all predictions
        feature_matrix = []
        for features in features_to_predict:
            ordered_features = [features[f] for f in cfg["feature_names"]]
            feature_matrix.append(ordered_features)
        
        # Single prediction call for all samples
        X = pd.DataFrame(feature_matrix, columns=cfg["feature_names"])
        probs_batch = pipe.predict_proba(X)
        
        # Process results
        for idx, (probs, sample_idx, features) in enumerate(zip(probs_batch, sample_indices, features_to_predict)):
            result = process_prediction(probs, cfg, features=features)  # Pass features for rest detection
            result["status"] = "predicted"
            result["sample_index"] = sample_idx
            predictions.append(result)
    
    return {
        "session_id": req.session_id,
        "total_samples": batch_size,
        "predictions": predictions,
        "buffer_ready": buffer.is_ready()
    }


@app.post("/predict_batch")
def predict_batch(req: BatchPredictRequest):
    if not req.batch:
        raise HTTPException(status_code=422, detail="Batch cannot be empty.")
    
    batch_size = len(req.batch)
    request_stats["total_samples_processed"] += batch_size
    
    results = []
    for idx, features in enumerate(req.batch):
        # Validate each sample
        for key in cfg["feature_names"]:
            if key not in features:
                raise HTTPException(
                    status_code=422, 
                    detail=f"Missing feature '{key}' in batch item {idx}"
                )
        
        # Reorder features according to the model's expectations
        ordered_features = [features[f] for f in cfg["feature_names"]]
        results.append(ordered_features)
    
    # Batch prediction
    X = pd.DataFrame(results, columns=cfg["feature_names"])
    probs_batch = pipe.predict_proba(X)
    
    # Process each prediction
    predictions = []
    for idx, probs in enumerate(probs_batch):
        predictions.append(process_prediction(probs, cfg, raw_emg_features=req.batch[idx]))
    
    return {"predictions": predictions}


@app.get("/stats")
def get_stats():
    """Get runtime statistics about server performance"""
    predict_times = list(request_stats["predict"])
    batch_times = list(request_stats["predict_batch"])
    windowed_times = list(request_stats["predict_windowed"])
    batch_windowed_times = list(request_stats["batch_predict_windowed"])
    
    return {
        "total_requests": request_stats["total_requests"],
        "total_samples_processed": request_stats["total_samples_processed"],
        "active_sessions": len(session_buffers),
        "predict": {
            "count": len(predict_times),
            "avg_ms": sum(predict_times) / len(predict_times) if predict_times else 0,
            "min_ms": min(predict_times) if predict_times else 0,
            "max_ms": max(predict_times) if predict_times else 0,
        },
        "predict_batch": {
            "count": len(batch_times),
            "avg_ms": sum(batch_times) / len(batch_times) if batch_times else 0,
            "min_ms": min(batch_times) if batch_times else 0,
            "max_ms": max(batch_times) if batch_times else 0,
        },
        "predict_windowed": {
            "count": len(windowed_times),
            "avg_ms": sum(windowed_times) / len(windowed_times) if windowed_times else 0,
            "min_ms": min(windowed_times) if windowed_times else 0,
            "max_ms": max(windowed_times) if windowed_times else 0,
        },
        "batch_predict_windowed": {
            "count": len(batch_windowed_times),
            "avg_ms": sum(batch_windowed_times) / len(batch_windowed_times) if batch_windowed_times else 0,
            "min_ms": min(batch_windowed_times) if batch_windowed_times else 0,
            "max_ms": max(batch_windowed_times) if batch_windowed_times else 0,
        }
    }


@app.post("/set_gate_threshold")
def set_gate_threshold(gate_threshold: float):
    """
    Set the gate model threshold for rest detection (0-1 scale).
    Default is 0.5 (balanced).
    
    Examples:
    - 0.3 = more sensitive to gestures (easier to trigger gesture detection)
    - 0.7 = more sensitive to rest (easier to trigger Neutral)
    """
    if gate_pipe is None:
        raise HTTPException(status_code=400, detail="No gate model loaded")
    cfg["gate_threshold"] = gate_threshold
    return {"status": "success", "gate_threshold": gate_threshold}

@app.get("/get_gate_threshold")
def get_gate_threshold():
    """Get current gate model threshold."""
    if gate_pipe is None:
        return {"error": "No gate model loaded"}
    return {"gate_threshold": cfg.get("gate_threshold", 0.5)}

@app.get("/get_model_info")
def get_model_info():
    """Get information about loaded models."""
    return {
        "model_directory": MODEL_DIR,
        "gate_enabled": gate_pipe is not None,
        "gesture_classes": cfg.get("class_names", []),
        "gate_threshold": cfg.get("gate_threshold", 0.5),
        "abstain_threshold": cfg.get("abstain_threshold", 0.90),
        "uses_windowing": cfg.get("uses_windowing", False)
    }

@app.post("/set_max_emg")
def set_max_emg(session_id: str, max_emg: float):
    """Set the maxEMG value for a session (from Unity calibration)."""
    session_max_emg[session_id] = max_emg
    return {"status": "success", "session_id": session_id, "max_emg": max_emg}

@app.post("/clear_session")
def clear_session(session_id: str):
    """Clear the buffer and maxEMG for a specific session."""
    if session_id in session_buffers:
        session_buffers[session_id].clear()
    if session_id in session_max_emg:
        del session_max_emg[session_id]
    return {"status": "cleared", "session_id": session_id}


@app.get("/ping")
def ping():
    return {"status": "ok"}

def check_param(req: PredictRequest):
    if not req.features:
        raise HTTPException(status_code=422, detail="Missing features in request.")
    for key in cfg["feature_names"]:
        if key not in req.features:
            raise HTTPException(status_code=422, detail=f"Missing feature: {key}")