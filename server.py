from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os
import time
from collections import deque
from datetime import datetime
from windowing_features import EMGBuffer, EMG_CHANNELS

# ============================================== Informations =========================================================
# Commande to run server: python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
# Doc auto generated http://127.0.0.1:8000/docs
# POST http://127.0.0.1:8000/predict
# BODY {"features": {"EMG1": 51, "EMG2": 8, "EMG3": 2, "EMG4": -3, "EMG5": -25, "EMG6": 12, "EMG7": -7, "EMG8": -26}}
#
# Two-Stage Windowed Prediction:
# 1. Client sends EMG samples at consistent rate (e.g., 50Hz = every 20ms)
# 2. Server buffers samples per session_id
# 3. Gate model detects Rest vs Gesture
# 4. If gesture detected, MLP classifier identifies specific gesture
# =====================================================================================================================


# ===== Load model artifacts =====
base_dir = "models"
models = sorted(os.listdir(base_dir))
if not models:
    print(f"\033[91m> Error: No model found in '{base_dir}' directory\033[0m")
    exit(1)

MODEL_DIR = f"{base_dir}/{models[-1]}"
print(f"\033[95m> Loading models from: {MODEL_DIR}\033[0m")

# Load gate model (Rest vs Gesture)
gate_pipe = None
gate_cfg = None
if os.path.exists(f"{MODEL_DIR}/gate_pipeline.joblib"):
    print(f"\033[96m  - Loading gate model\033[0m")
    gate_pipe = joblib.load(f"{MODEL_DIR}/gate_pipeline.joblib")
    with open(f"{MODEL_DIR}/gate_config.json") as f:
        gate_cfg = json.load(f)
    print(f"\033[92m  ✓ Gate model loaded\033[0m")
else:
    print(f"\033[93m  ⚠ No gate model - single-stage mode\033[0m")

# Load gesture classifier
gesture_file = f"{MODEL_DIR}/gesture_pipeline.joblib"
config_file = f"{MODEL_DIR}/gesture_config.json"

# Fallback to legacy naming if needed
if not os.path.exists(gesture_file):
    gesture_file = f"{MODEL_DIR}/pipeline.joblib"
    config_file = f"{MODEL_DIR}/config.json"

print(f"\033[96m  - Loading gesture classifier\033[0m")
pipe = joblib.load(gesture_file)
with open(config_file) as f:
    cfg = json.load(f)
print(f"\033[92m  ✓ Gesture classifier loaded\033[0m")

if gate_pipe:
    print(f"\033[94m> Two-stage prediction: Gate → Gesture\033[0m")
else:
    print(f"\033[94m> Single-stage prediction\033[0m")


# ======= Request Schemas =======
class PredictRequest(BaseModel):
    features: dict
    session_id: str

class BatchPredictRequest(BaseModel):
    batch: list[dict]  # List of EMG samples (chronological order)
    session_id: str


# ======= FastAPI App =======
app = FastAPI(title="EMG Gesture Recognition API")

# ======= Statistics =======
request_stats = {
    "predict": deque(maxlen=100),
    "batch_predict": deque(maxlen=100),
    "total_requests": 0,
    "total_samples_processed": 0
}

# ======= Session Management =======
session_buffers = {}  # {session_id: EMGBuffer}
session_max_emg = {}  # {session_id: float}

def get_or_create_buffer(session_id: str) -> EMGBuffer:
    """Get or create buffer for session."""
    if session_id not in session_buffers:
        window_size = cfg.get("window_size", 10)
        session_buffers[session_id] = EMGBuffer(window_size=window_size)
    return session_buffers[session_id]


def normalize_features(features: dict, max_emg: float) -> None:
    """Normalize intensity-based features (MAV, WL) by maxEMG in-place."""
    for ch in range(1, 9):
        if f'EMG{ch}_MAV' in features:
            features[f'EMG{ch}_MAV'] /= max_emg
        if f'EMG{ch}_WL' in features:
            features[f'EMG{ch}_WL'] /= max_emg


@app.middleware("http")
async def add_timing_stats(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000
    
    endpoint = request.url.path
    if endpoint in ["/predict", "/batch_predict"]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {endpoint}: {duration:.2f}ms")
        request_stats["total_requests"] += 1
    
    return response


def process_prediction(probs, features):
    """
    Two-stage prediction: Gate model → Gesture classifier
    
    Args:
        probs: Gesture classifier probabilities
        features: Extracted window features (dict) - all 32 features
    """
    
    # STAGE 1: Gate model (Rest vs Gesture)
    if gate_pipe is not None:
        # Gate model uses ALL features (same as gesture classifier)
        # Build feature vector in correct order
        gate_features = [features[f] for f in gate_cfg["feature_names"]]
        gate_X = pd.DataFrame([gate_features], columns=gate_cfg["feature_names"])
        gate_probs = gate_pipe.predict_proba(gate_X)[0]
        
        rest_prob = float(gate_probs[0])
        gesture_prob = float(gate_probs[1])
        gate_threshold = cfg.get("gate_threshold", 0.85)
        
        # If at rest, return Neutral immediately
        if rest_prob > gate_threshold:
            return {
                "label": "Neutral",
                "prob": rest_prob,
                "topk": [{"label": "Neutral", "prob": rest_prob}],
                "detection_method": "gate",
                "gate_rest_prob": rest_prob,
                "gate_gesture_prob": gesture_prob
            }
    
    # STAGE 2: Gesture classifier
    top_idx = probs.argmax()
    top_prob = float(probs[top_idx])
    predicted_label = cfg["class_names"][top_idx]
    
    # Check confidence threshold
    abstain_threshold = cfg.get("abstain_threshold", 0.90)
    label = predicted_label if top_prob >= abstain_threshold else "Unknown"
    
    # Top-K results
    top_k = cfg.get("top_k", 2)
    topk_idx = probs.argsort()[-top_k:][::-1]
    topk = [{"label": cfg["class_names"][i], "prob": float(probs[i])} for i in topk_idx]
    
    result = {
        "label": label,
        "prob": top_prob,
        "topk": topk,
        "detection_method": "gesture"
    }
    
    # Add gate info if available
    if gate_pipe is not None:
        result["gate_rest_prob"] = rest_prob
        result["gate_gesture_prob"] = gesture_prob
    
    return result


# ======= API Endpoints =======
@app.post("/predict")
def predict(req: PredictRequest):
    """
    Single sample prediction with windowing.
    
    - Buffers samples per session
    - Returns "buffering" until window is full
    - Returns prediction with two-stage classification
    """
    # Validate EMG channels
    for ch in EMG_CHANNELS:
        if ch not in req.features:
            raise HTTPException(status_code=422, detail=f"Missing feature: {ch}")
    
    # Get buffer and add sample
    buffer = get_or_create_buffer(req.session_id)
    buffer.add_sample(req.features)
    
    request_stats["total_samples_processed"] += 1
    
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
    
    # Normalize by session maxEMG if available
    if req.session_id in session_max_emg:
        normalize_features(features, session_max_emg[req.session_id])
    
    # Predict
    ordered_features = [features[f] for f in cfg["feature_names"]]
    X = pd.DataFrame([ordered_features], columns=cfg["feature_names"])
    probs = pipe.predict_proba(X)[0]
    
    result = process_prediction(probs, features)
    result["status"] = "predicted"
    return result


@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest):
    """
    Batch prediction with windowing.
    
    - Processes multiple samples sequentially (maintains temporal order)
    - More efficient than calling /predict multiple times
    - Returns predictions for all samples that have full windows
    """
    if not req.batch:
        raise HTTPException(status_code=422, detail="Batch cannot be empty")
    
    # Validate all samples
    for idx, sample in enumerate(req.batch):
        for ch in EMG_CHANNELS:
            if ch not in sample:
                raise HTTPException(status_code=422, detail=f"Missing '{ch}' in sample {idx}")
    
    buffer = get_or_create_buffer(req.session_id)
    window_size = cfg.get("window_size", 10)
    
    batch_size = len(req.batch)
    request_stats["total_samples_processed"] += batch_size
    
    predictions = []
    features_to_predict = []
    sample_indices = []
    
    # Process each sample sequentially
    for idx, sample in enumerate(req.batch):
        buffer.add_sample(sample)
        
        if buffer.is_ready():
            features = buffer.extract_features()
            features_to_predict.append(features)
            sample_indices.append(idx)
        else:
            samples_needed = window_size - len(buffer.buffer)
            predictions.append({
                "status": "buffering",
                "sample_index": idx,
                "samples_needed": samples_needed,
                "buffer_size": len(buffer.buffer)
            })
    
    # Batch predict all ready samples
    if features_to_predict:
        # Normalize if maxEMG available
        if req.session_id in session_max_emg:
            max_emg = session_max_emg[req.session_id]
            for features in features_to_predict:
                normalize_features(features, max_emg)
        
        # Build feature matrix
        feature_matrix = []
        for features in features_to_predict:
            ordered_features = [features[f] for f in cfg["feature_names"]]
            feature_matrix.append(ordered_features)
        
        X = pd.DataFrame(feature_matrix, columns=cfg["feature_names"])
        
        # STAGE 1: Batch gate model prediction (if available)
        gate_results = None
        if gate_pipe is not None:
            gate_probs_batch = gate_pipe.predict_proba(X)
            gate_threshold = cfg.get("gate_threshold", 0.5)
            gate_results = []
            
            for gate_probs in gate_probs_batch:
                rest_prob = float(gate_probs[0])
                gesture_prob = float(gate_probs[1])
                is_rest = rest_prob > gate_threshold
                gate_results.append({
                    "is_rest": is_rest,
                    "rest_prob": rest_prob,
                    "gesture_prob": gesture_prob
                })
        
        # STAGE 2: Batch gesture classifier prediction
        probs_batch = pipe.predict_proba(X)
        abstain_threshold = cfg.get("abstain_threshold", 0.90)
        top_k = cfg.get("top_k", 2)
        
        # Process results
        for idx, (probs, sample_idx) in enumerate(zip(probs_batch, sample_indices)):
            # Check gate model first
            if gate_results is not None and gate_results[idx]["is_rest"]:
                result = {
                    "label": "Neutral",
                    "prob": gate_results[idx]["rest_prob"],
                    "topk": [{"label": "Neutral", "prob": gate_results[idx]["rest_prob"]}],
                    "detection_method": "gate",
                    "gate_rest_prob": gate_results[idx]["rest_prob"],
                    "gate_gesture_prob": gate_results[idx]["gesture_prob"]
                }
            else:
                # Gesture classification
                top_idx = probs.argmax()
                top_prob = float(probs[top_idx])
                predicted_label = cfg["class_names"][top_idx]
                label = predicted_label if top_prob >= abstain_threshold else "Unknown"
                
                topk_idx = probs.argsort()[-top_k:][::-1]
                topk = [{"label": cfg["class_names"][i], "prob": float(probs[i])} for i in topk_idx]
                
                result = {
                    "label": label,
                    "prob": top_prob,
                    "topk": topk,
                    "detection_method": "gesture"
                }
                
                # Add gate info if available
                if gate_results is not None:
                    result["gate_rest_prob"] = gate_results[idx]["rest_prob"]
                    result["gate_gesture_prob"] = gate_results[idx]["gesture_prob"]
            
            result["status"] = "predicted"
            result["sample_index"] = sample_idx
            predictions.append(result)
    
    return {
        "session_id": req.session_id,
        "total_samples": batch_size,
        "predictions": predictions,
        "buffer_ready": buffer.is_ready()
    }


@app.get("/stats")
def get_stats():
    """Get server performance statistics."""
    predict_times = list(request_stats["predict"])
    batch_times = list(request_stats["batch_predict"])
    
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
        "batch_predict": {
            "count": len(batch_times),
            "avg_ms": sum(batch_times) / len(batch_times) if batch_times else 0,
            "min_ms": min(batch_times) if batch_times else 0,
            "max_ms": max(batch_times) if batch_times else 0,
        }
    }


@app.get("/info")
def get_info():
    """Get model and configuration information."""
    return {
        "model_directory": MODEL_DIR,
        "gate_enabled": gate_pipe is not None,
        "gesture_classes": cfg.get("class_names", []),
        "gate_threshold": cfg.get("gate_threshold", 0.5),
        "abstain_threshold": cfg.get("abstain_threshold", 0.90),
        "window_size": cfg.get("window_size", 10),
        "window_stride": cfg.get("window_stride", 5)
    }


@app.post("/config/gate_threshold")
def set_gate_threshold(gate_threshold: float):
    """
    Set gate model threshold (0-1).
    - Lower = more sensitive to gestures (more responsive)
    - Higher = more sensitive to rest (fewer false activations)
    """
    if gate_pipe is None:
        raise HTTPException(status_code=400, detail="No gate model loaded")
    if not 0 <= gate_threshold <= 1:
        raise HTTPException(status_code=422, detail="Threshold must be between 0 and 1")
    
    cfg["gate_threshold"] = gate_threshold
    return {"status": "success", "gate_threshold": gate_threshold}


@app.post("/session/set_max_emg")
def set_max_emg(session_id: str, max_emg: float):
    """Set maxEMG value for session (from calibration)."""
    session_max_emg[session_id] = max_emg
    return {"status": "success", "session_id": session_id, "max_emg": max_emg}


@app.post("/session/clear")
def clear_session(session_id: str):
    """Clear buffer and calibration data for session."""
    if session_id in session_buffers:
        session_buffers[session_id].clear()
    if session_id in session_max_emg:
        del session_max_emg[session_id]
    return {"status": "cleared", "session_id": session_id}


@app.get("/ping")
def ping():
    """Health check endpoint."""
    return {"status": "ok"}
