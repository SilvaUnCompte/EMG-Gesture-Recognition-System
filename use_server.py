from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json

# ===== Load model artifacts =====
MODEL_DIR = "models/gesture_cls/1.0.0_20250820T080853Z"
pipe = joblib.load(f"{MODEL_DIR}/pipeline.joblib")
with open(f"{MODEL_DIR}/config.json") as f:
    cfg = json.load(f)

# ===== Define request schema =====
class PredictRequest(BaseModel):
    features: dict

# ===== Create FastAPI app =====
app = FastAPI(title="Gesture Classifier API")

@app.post("/predict")
def predict(req: PredictRequest):

    # Reorder features according to the model's expectations
    ordered_features = [req.features[f] for f in cfg["feature_names"]]
    probs = pipe.predict_proba([ordered_features])[0]
    
    # Top-1
    top_idx = probs.argmax()
    top_prob = float(probs[top_idx])
    label = cfg["class_names"][top_idx]
    
    # Abstention
    if top_prob < cfg.get("abstain_threshold", 0.65):
        label = "unknown"
    
    # Top-K
    top_k = cfg.get("top_k", 2)
    topk_idx = probs.argsort()[-top_k:][::-1]
    topk = [{"label": cfg["class_names"][i], "prob": float(probs[i])} for i in topk_idx]
    
    return {"label": label, "prob": top_prob, "topk": topk}



# Commande to run server: python -m uvicorn use_server:app --reload --host 0.0.0.0 --port 8000
# Doc auto generated http://127.0.0.1:8000/docs
# POST http://127.0.0.1:8000/predict
# BODY {"features": {"EMG1": 51, "EMG2": 8, "EMG3": 2, "EMG4": -3, "EMG5": -25, "EMG6": 12, "EMG7": -7, "EMG8": -26}}