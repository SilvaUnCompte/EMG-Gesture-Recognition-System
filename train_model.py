import json
import os
from sklearn.calibration import LabelEncoder, label_binarize
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
import seaborn as sns
import joblib
import sys

CLASS_NAMES = ['gesture1', 'gesture2', 'gesture3', 'gesture4', 'gesture5'] # Define dynamically
FEATURE_NAMES = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8']
pre_sep = "\n\n\033[92m =========="
post_sep = "===========\033[0m"

def main():
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    else:
        print("\033[91mNo path provided, using default: data.csv\033[0m")
        data_path = "data.csv"

    # Import and preprocess data
    print(pre_sep, "Importing and preprocessing data", post_sep)
    df = import_data(data_path)
    df = encode_labels(df)

    # Split data
    X_train, y_train, X_test, y_test = splitting_data(df)

    # Create and train model
    print(pre_sep, "Creating and training the MLP model", post_sep)
    model = create_mlp_model(  
        hidden_layer_sizes=(64,),                   # TODO: change with adapted Hyperparameters
        activation='tanh',                          # TODO: change with adapted Hyperparameters
        alpha=0.01395146418868073,                  # TODO: change with adapted Hyperparameters
        learning_rate_init=0.0074171634474875445    # TODO: change with adapted Hyperparameters
    )
    print("Model parameters:", model.get_params(), "\nTraining...")

    # Export model
    export_model(model, X_train, y_train, X_test, y_test)



# =========== Import data ===========

def import_data(path):
    if not os.path.exists(path):
        print(f"\033[91mFile not found: {path}\033[0m")
        sys.exit(1)

    df = pd.read_csv(path, sep=";")
    print(df.shape)  # dimensions
    print(df.info())  # types and missing values
    df = df.drop_duplicates()  # remove duplicate rows
    df = df.dropna() # remove rows with missing values
    return df

# =========== Encode labels ===========

def encode_labels(df):
    """
    Encodes the gesture labels in the DataFrame using LabelEncoder.

    Parameters:
    df (DataFrame): The input DataFrame containing gesture labels.

    Returns:
    DataFrame: A new DataFrame with an additional 'label' column containing encoded labels.
    """
    global CLASS_NAMES

    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['gesture'])
    CLASS_NAMES = encoder.classes_.tolist()
    print("\nEncoded labels:")
    print(df[['label', 'gesture']].drop_duplicates())

    return df

# =========== Split data ===========

def splitting_data(df):
    """
    Splits the DataFrame into training and test sets.

    Parameters:
    df (DataFrame): The input DataFrame with features and encoded labels.

    Returns:
    Tuple: X_train, y_train, X_test, y_test
    """

    X = df[FEATURE_NAMES]
    y = df['label']

    # Split dataset into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, y_train, X_test, y_test

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
    print(pre_sep, "Evaluating model performance", post_sep)

    # Evaluate on test set
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    print(report)

    # Cross-validation
    scores = cross_val_score(pipe, X_test, y_test, cv=5)
    print("Cross-validation scores:", scores)
    accuracy = scores.mean()
    print("\033[95mMean accuracy:", accuracy, "\033[0m")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # ROC Curve
    y_test_bin = label_binarize(y_test, classes=pipe.named_steps['mlp'].classes_)
    probs_full = pipe.predict_proba(X_test)

    plt.figure()
    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs_full[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend()
    plt.show()

    return accuracy, report, cm

# =========== Model export ===========

def export_model(mlp, X_train, y_train, X_test, y_test):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', mlp)
    ])
    pipe.fit(X_train, y_train)


    # Create a directory for the model
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_dir = f"models/gesture_recognition_1.2.7_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)


    # Create metrics and show evaluation plots
    accuracy, report, cm = evaluate_model(pipe, X_test, y_test)
    metrics = {"accuracy": accuracy,"report": report,"confusion_matrix": cm.tolist()}
    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


    # Save pipeline
    joblib.dump(pipe, f"{model_dir}/pipeline.joblib")


    # Save config (schema, classes, thresholds, etc.)
    config = {
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "top_k": 2,
        "abstain_threshold": 0.6,  # below this max prob => "unknown"
        "random_state": 42,
        "library": {"sklearn": ">=1.2"},
    }
    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)


    print("\033[92mModel exported to:", model_dir,"\033[0m")


if __name__ == "__main__":
    main()