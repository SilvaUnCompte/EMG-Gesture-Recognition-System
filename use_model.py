# predict_cli.py
import argparse, json, joblib, numpy as np
from sklearn.discriminant_analysis import StandardScaler



def load_artifacts(model_dir):
    pipe = joblib.load(f"{model_dir}/pipeline.joblib")

    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    return pipe, cfg



def predict_one(pipe, cfg, features):
    features = np.asarray(features, dtype=np.float32)
    assert features.shape == (8,), "Expected 8 features in fixed order"
    probs = pipe.predict_proba(features.reshape(1, -1))[0]
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--features", nargs=8, type=float, required=True,
                        help="8 numeric features in correct order")
    args = parser.parse_args()


    pipe, cfg = load_artifacts(args.model_dir)
    out = predict_one(pipe, cfg, args.features)
    print(json.dumps(out, indent=2))
