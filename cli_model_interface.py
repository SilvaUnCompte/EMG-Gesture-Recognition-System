import argparse, json, joblib, numpy as np
import pandas as pd
import os

def load_artifacts(model_dir):
    pipe = joblib.load(f"{model_dir}/pipeline.joblib")

    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    return pipe, cfg


def predict_one(pipe, cfg, features):
    features = np.asarray(features, dtype=np.float32)
    assert features.shape == (8,), "Expected 8 features in fixed order"

    # Use feature names from config if available
    feature_names = cfg.get("feature_names", [f"f{i}" for i in range(8)])
    features_df = pd.DataFrame([features], columns=feature_names)
    probs = pipe.predict_proba(features_df)[0]
    top_idx = probs.argmax()
    top_prob = float(probs[top_idx])
    label = cfg["class_names"][top_idx]

    # abstention
    if top_prob < cfg["abstain_threshold"]:
        label = "unknown"

    # top-k
    top_k = cfg["top_k"]
    topk_idx = probs.argsort()[-top_k:][::-1]
    topk = [{"label": cfg["class_names"][i], "prob": float(probs[i])} for i in topk_idx]
    return {"label": label, "prob": top_prob, "topk": topk}

def default_model_dir():
    base_dir = "models"
    models = sorted(os.listdir(base_dir))
    if not models:
        print(f"\033[91m> Error: No model found in '{base_dir}' directory\033[0m")
        exit(1)
    latest_model = models[-1]
    selected_model_dir = f"{base_dir}/{latest_model}"
    print(f"\033[95m> Using model from: {selected_model_dir}\033[0m")

    return selected_model_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Path to the model directory")
    parser.add_argument("--features", nargs=8, type=float, required=True,
                        help="8 numeric features in correct order")
    
    args = parser.parse_args()

    if not args.model_dir:
        args.model_dir = default_model_dir()
    if not os.path.exists(args.model_dir):
        print(f"\033[91m> Error: Model directory '{args.model_dir}' does not exist\033[0m")
        exit(1)

    pipe, cfg = load_artifacts(args.model_dir)
    out = predict_one(pipe, cfg, args.features)
    print(json.dumps(out, indent=2))
