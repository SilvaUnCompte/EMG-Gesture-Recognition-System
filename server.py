from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os

# ============================================== Informations =========================================================
# Commande to run server: python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
# Doc auto generated http://127.0.0.1:8000/docs
# POST http://127.0.0.1:8000/predict
# BODY {"features": {"EMG1": 51, "EMG2": 8, "EMG3": 2, "EMG4": -3, "EMG5": -25, "EMG6": 12, "EMG7": -7, "EMG8": -26}}
# =====================================================================================================================


# ===== Load model artifacts =====
base_dir = "models"
models = sorted(os.listdir(base_dir))
if not models:
    print(f"\033[91m> Error: No model found in '{base_dir}' directory\033[0m")
    exit(1)
latest_model = models[-1]
MODEL_DIR = f"{base_dir}/{latest_model}"
print(f"\033[95m> Using model from: {MODEL_DIR}\033[0m")

pipe = joblib.load(f"{MODEL_DIR}/pipeline.joblib")
with open(f"{MODEL_DIR}/config.json") as f:
    cfg = json.load(f)


# ======= Define request schema =======
class PredictRequest(BaseModel):
    features: dict


# ======= Create FastAPI app =======
app = FastAPI(title="Gesture Classifier API")


# ======= Define API endpoints =======
@app.post("/predict")
def predict(req: PredictRequest):
    check_param(req)

    # Reorder features according to the model's expectations
    ordered_features = [req.features[f] for f in cfg["feature_names"]]
    X = pd.DataFrame([ordered_features], columns=cfg["feature_names"])
    probs = pipe.predict_proba(X)[0]
    
    # Top-1
    top_idx = probs.argmax()
    top_prob = float(probs[top_idx])
    label = cfg["class_names"][top_idx]

    # Not confident
    if top_prob < cfg.get("abstain_threshold", 0.65):
        label = "unknown"
    
    # Top-K
    top_k = cfg.get("top_k", 2)
    topk_idx = probs.argsort()[-top_k:][::-1]
    topk = [{"label": cfg["class_names"][i], "prob": float(probs[i])} for i in topk_idx]
    
    return {"label": label, "prob": top_prob, "topk": topk}


@app.post("/ping")
def ping():
    return {"status": "ok"}

def check_param(req: PredictRequest):
    if not req.features:
        raise HTTPException(status_code=422, detail="Missing features in request.")
    for key in cfg["feature_names"]:
        if key not in req.features:
            raise HTTPException(status_code=422, detail=f"Missing feature: {key}")