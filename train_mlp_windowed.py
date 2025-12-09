"""
MLP Training with Temporal Windowing and Integrated Gate Model

This script trains TWO models in one go:
1. MLP Classifier - Identifies specific gestures (PalmarGrasp, WristFlexion, WristExtension, LateralGrasp)
2. RF Gate Model - Binary classifier for Rest vs Gesture detection

Both models are exported to the same directory for easy deployment.
The gate model enables proper rest state detection without contaminating gesture training data.

Usage:
    python train_mlp_windowed.py [path_to_data.csv]
    
Output:
    models/gesture_windowed_mlp_TIMESTAMP/
    ├── gesture_pipeline.joblib  # MLP gesture classifier
    ├── gate_pipeline.joblib     # RF gate model (Rest vs Gesture)
    ├── gesture_config.json & gesture_metrics.json
    └── gate_config.json & gate_metrics.json
"""

from shared_functions import import_data, encode_labels, splitting_data, drop_outside_scope_data
from windowing_features import create_windows_from_dataframe, get_feature_names, WINDOW_SIZE, WINDOW_STRIDE, EMG_CHANNELS
from sklearn.calibration import label_binarize
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from datetime import datetime
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os


CLASS_NAMES = []  # Define dynamically
GESTURE_COL = "gesture"  # After windowing, gesture label column
THRESHOLD_COL = "threshold"
PRE_SEP = "\n\n\033[92m =========="
POST_SEP = "===========\033[0m"


def debug_class_distribution(df, stage_name):
    """Print detailed class distribution"""
    print(f"\n{'='*60}")
    print(f"CLASS DISTRIBUTION: {stage_name}")
    print('='*60)
    
    # Use 'label' column if it exists (after encoding), otherwise use GESTURE_COL
    label_col = 'label' if 'label' in df.columns else GESTURE_COL
    
    if label_col in df.columns:
        counts = df[label_col].value_counts()
        total = len(df)
        
        for cls_idx, count in counts.items():
            # If using encoded labels, map to class names
            if label_col == 'label':
                cls_idx = int(cls_idx)
                cls_name = CLASS_NAMES[cls_idx] if (CLASS_NAMES and cls_idx < len(CLASS_NAMES)) else f"Class_{cls_idx}"
            else:
                # Raw gesture names (before encoding)
                cls_name = str(cls_idx)
            
            pct = (count / total) * 100
            print(f"{cls_name:20s}: {count:6d} ({pct:5.1f}%)")
        
        # Check if severely imbalanced
        max_count = counts.max()
        min_count = counts.min()
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 3:
            print(f"\n⚠️  SEVERE IMBALANCE DETECTED!")
            print(f"   Max/Min ratio: {imbalance_ratio:.1f}:1")
            print(f"   Model will likely favor majority classes!")


def main():
    global CLASS_NAMES

    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    else:
        print("\033[91mNo path provided, using default: data.csv\033[0m")
        data_path = "data.csv"

    # Import and preprocess data
    print(PRE_SEP, "Importing and preprocessing data", POST_SEP)
    df = import_data(data_path)
    df = drop_outside_scope_data(df, "CurrentGestures", "Threshold")
    
    # Calculate gesture-specific maxEMG before filtering
    # This will be used for normalization in inference
    gesture_max_emg = {}
    
    print("\nGesture-specific maxEMG (from training data):")
    for gesture in df['CurrentGestures'].unique():
        if gesture == 'Neutral':
            continue  # Skip Neutral
        gesture_df = df[df['CurrentGestures'] == gesture]
        max_val = gesture_df[EMG_CHANNELS].abs().max().max()
        gesture_max_emg[gesture] = float(max_val)
        print(f"  {gesture:20s}: {max_val:.2f}")
    
    # Keep Neutral data for now - it's needed for gate model training
    # We'll filter it out after windowing for the gesture classifier
    print(f"\nTotal samples (including Neutral): {len(df)}")
    print(f"Gesture distribution:\n{df['CurrentGestures'].value_counts()}")

    # Create windows and extract features
    print(PRE_SEP, f"Creating windows (size={WINDOW_SIZE}, stride={WINDOW_STRIDE})", POST_SEP)
    windowed_df = create_windows_from_dataframe(df, gesture_col="CurrentGestures")
    print(f"Original samples: {len(df)}")
    print(f"Windowed samples: {len(windowed_df)}")
    print(f"Features extracted per window: {len(get_feature_names())}")
    
    # Drop rows with invalid thresholds if threshold column exists
    if THRESHOLD_COL in windowed_df.columns:
        before_count = windowed_df.shape[0]
        windowed_df = windowed_df[windowed_df[THRESHOLD_COL] == "above"]
        removed = before_count - windowed_df.shape[0]
        if removed > 0:
            print(f"Filtered out threshold != 'above': {removed} removed")

    # Save a copy WITH Neutral for gate model training (BEFORE filtering out Neutral)
    # Gate model needs both Rest and Gesture samples
    windowed_df_with_neutral = windowed_df.copy()
    
    # Now filter out Neutral for gesture classifier training
    if GESTURE_COL in windowed_df.columns:
        before_count = windowed_df.shape[0]
        windowed_df = windowed_df[windowed_df[GESTURE_COL] != "Neutral"]
        removed = before_count - windowed_df.shape[0]
        if removed > 0:
            print(f"\nFiltered out 'Neutral' from gesture training data: {removed} removed")
            print(f"Remaining gesture samples: {len(windowed_df)}")

    # Encode labels
    windowed_df, CLASS_NAMES = encode_labels(windowed_df, GESTURE_COL)

    # Debug: Class distribution after windowing (before balancing)
    debug_class_distribution(windowed_df, "After windowing (before balancing)")
    
    # Balance dataset by undersampling majority classes
    class_counts = windowed_df['label'].value_counts()
    min_class_count = class_counts.min()
    print(f"\nBalancing: undersampling to {min_class_count} samples per class")
    balanced_dfs = []
    for cls_idx in class_counts.index:
        cls_df = windowed_df[windowed_df['label'] == cls_idx]
        if len(cls_df) > min_class_count:
            cls_df = cls_df.sample(n=min_class_count, random_state=42)
        balanced_dfs.append(cls_df)
    
    windowed_df = pd.concat(balanced_dfs, ignore_index=True)
    windowed_df = windowed_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Debug: Class distribution after balancing
    debug_class_distribution(windowed_df, "After balancing")

    # Get feature names
    FEATURE_NAMES = get_feature_names()

    # Split data
    X_train, y_train, X_test, y_test = splitting_data(windowed_df, FEATURE_NAMES, 0.25)
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Classes: {CLASS_NAMES}")

    # Create and train model
    print(PRE_SEP, "Creating and training the MLP model", POST_SEP)
    model = create_mlp_model(  
        hidden_layer_sizes=(128, 64, 32),               # Larger network for more features
        activation='relu',
        alpha=0.001,
        learning_rate_init=0.001
    )
    print("Model parameters:", model.get_params(), "\nTraining...")

    # Train gate model (Rest vs Gesture detection)
    print(PRE_SEP, "Training RF Gate Model (Rest vs Gesture)", POST_SEP)
    gate_model, gate_X_train, gate_y_train, gate_X_test, gate_y_test = train_gate_model(windowed_df_with_neutral, FEATURE_NAMES)

    # Export both models to same directory
    export_models(model, X_train, y_train, X_test, y_test, 
                  gate_model, gate_X_train, gate_y_train, gate_X_test, gate_y_test,
                  FEATURE_NAMES, gesture_max_emg)


# =========== Train Gate Model ===========

def train_gate_model(windowed_df_with_neutral, feature_names):
    """Train RF gate model for binary classification: Rest (0) vs Gesture (1)"""
    
    # Create binary labels: 0 = Neutral (rest), 1 = Any gesture (active)
    df_gate = windowed_df_with_neutral.copy()
    df_gate['gate_label'] = (df_gate['gesture'] != "Neutral").astype(int)
    
    rest_count = (df_gate['gate_label'] == 0).sum()
    gesture_count = (df_gate['gate_label'] == 1).sum()
    print(f"Gate training data - Rest: {rest_count}, Gesture: {gesture_count}")
    
    # Balance dataset by undersampling majority class
    rest_df = df_gate[df_gate['gate_label'] == 0]
    gesture_df = df_gate[df_gate['gate_label'] == 1]
    
    min_samples = min(len(rest_df), len(gesture_df))
    print(f"Balancing gate data to {min_samples} samples per class")
    
    rest_balanced = rest_df.sample(n=min_samples, random_state=42)
    gesture_balanced = gesture_df.sample(n=min_samples, random_state=42)
    
    df_balanced = pd.concat([rest_balanced, gesture_balanced], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    X = df_balanced[feature_names]
    y = df_balanced['gate_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Gate model - Training: {len(X_train)}, Test: {len(X_test)}")
    
    # Create gate model
    gate_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    return gate_model, X_train, y_train, X_test, y_test


# =========== Create MLP model ===========

def create_mlp_model(hidden_layer_sizes, activation, alpha, learning_rate_init):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver='adam',
        alpha=alpha,                                    # L2 regularization
        learning_rate_init=learning_rate_init,
        max_iter=2000,
        validation_fraction=0.2,                        # Use 20% of training data for validation
        early_stopping=True,                            # Stop when validation score stops improving
        n_iter_no_change=50,                            # Stop after 50 iterations without improvement
        random_state=42,
        batch_size='auto'
    )

# =========== Metrics about model performances ===========

def evaluate_model(pipe, X_test, y_test):
    print(PRE_SEP, "Evaluating model performance", POST_SEP)

    # Evaluate on test set
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Test set accuracy (single evaluation)
    test_accuracy = pipe.score(X_test, y_test)
    print(f"\033[95mTest set accuracy: {test_accuracy:.4f}\033[0m")
    
    accuracy = test_accuracy

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ROC Curve
    y_test_bin = label_binarize(y_test, classes=pipe.named_steps['mlp'].classes_)
    probs_full = pipe.predict_proba(X_test)

    plt.figure(figsize=(10, 8))
    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs_full[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return accuracy, report, cm


def evaluate_gate_model(pipe, X_test, y_test):
    """Evaluate the gate model (Rest vs Gesture)"""
    print(PRE_SEP, "Evaluating Gate Model", POST_SEP)
    
    y_pred = pipe.predict(X_test)
    
    class_names = ['Rest', 'Gesture']
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\033[95mGate model accuracy: {accuracy:.4f}\033[0m")
    
    # Analyze performance
    rest_recall = report['Rest']['recall']
    gesture_recall = report['Gesture']['recall']
    print(f"\n\033[93mGate Performance:\033[0m")
    print(f"Rest Recall: {rest_recall:.1%} (correctly identifies rest state)")
    print(f"Gesture Recall: {gesture_recall:.1%} (correctly detects gesture attempts)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Gate Model: Rest vs Gesture")
    plt.tight_layout()
    plt.show()
    
    return accuracy, report, cm


# =========== Model export ===========

def export_models(mlp, X_train, y_train, X_test, y_test, 
                  gate_model, gate_X_train, gate_y_train, gate_X_test, gate_y_test,
                  feature_names, gesture_max_emg):
    """Train and export both gesture classifier and gate model to same directory"""
    
    # Train gesture classifier pipeline
    gesture_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', mlp)
    ])
    gesture_pipe.fit(X_train, y_train)

    # Train gate model pipeline
    gate_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', gate_model)
    ])
    gate_pipe.fit(gate_X_train, gate_y_train)

    # Create a single directory for both models
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    model_dir = f"models/gesture_windowed_mlp_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    # Evaluate and save gesture classifier
    print("\n" + "="*60)
    print("GESTURE CLASSIFIER EVALUATION")
    print("="*60)
    gesture_accuracy, gesture_report, gesture_cm = evaluate_model(gesture_pipe, X_test, y_test)
    
    gesture_metrics = {
        "accuracy": gesture_accuracy,
        "report": gesture_report,
        "confusion_matrix": gesture_cm.tolist()
    }
    with open(f"{model_dir}/gesture_metrics.json", "w") as f:
        json.dump(gesture_metrics, f, indent=2)
    
    joblib.dump(gesture_pipe, f"{model_dir}/gesture_pipeline.joblib")
    
    gesture_config = {
        "model_type": "gesture_classifier",
        "feature_names": feature_names,
        "class_names": CLASS_NAMES,
        "gesture_max_emg": gesture_max_emg,
        "top_k": 2,
        "abstain_threshold": 0.90,
        "gate_threshold": 0.75,  
        "random_state": 42,
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "uses_windowing": True,
        "includes_neutral": False,
        "library": {"sklearn": ">=1.2", "numpy": ">=1.20"},
    }
    with open(f"{model_dir}/gesture_config.json", "w") as f:
        json.dump(gesture_config, f, indent=2)

    # Evaluate and save gate model
    print("\n" + "="*60)
    print("GATE MODEL EVALUATION")
    print("="*60)
    gate_accuracy, gate_report, gate_cm = evaluate_gate_model(gate_pipe, gate_X_test, gate_y_test)
    
    gate_metrics = {
        "accuracy": gate_accuracy,
        "report": gate_report,
        "confusion_matrix": gate_cm.tolist()
    }
    with open(f"{model_dir}/gate_metrics.json", "w") as f:
        json.dump(gate_metrics, f, indent=2)
    
    joblib.dump(gate_pipe, f"{model_dir}/gate_pipeline.joblib")
    
    gate_config = {
        "model_type": "gate",
        "description": "Binary classifier: Rest (0) vs Gesture (1)",
        "feature_names": feature_names,
        "class_names": ["Rest", "Gesture"],
        "random_state": 42,
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "uses_windowing": True,
        "library": {"sklearn": ">=1.2", "numpy": ">=1.20"},
    }
    with open(f"{model_dir}/gate_config.json", "w") as f:
        json.dump(gate_config, f, indent=2)

    print("\n" + "="*60)
    print("\033[92m✓ Both models exported to:", model_dir, "\033[0m")
    print("="*60)
    print(f"\033[96mGesture Classifier:\033[0m gesture_pipeline.joblib")
    print(f"\033[96mGate Model:\033[0m gate_pipeline.joblib")
    print("="*60)


if __name__ == "__main__":
    main()
