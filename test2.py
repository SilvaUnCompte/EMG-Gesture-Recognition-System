import pandas as pd # pyright: ignore[reportMissingModuleSource]
from sklearn.preprocessing import LabelEncoder, StandardScaler # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns


# ============================= Read Data =============================

df = pd.read_csv("data3.csv", sep=";")
print(df.shape)  # dimensions
print(df.info())  # types and missing values
print(df.head())  # 5 first lines

emg_cols = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8']


# ============================= Clean Data =============================

df = df.drop_duplicates()  # remove duplicate rows
df = df.dropna() # remove rows with missing values

# Or fill missing numeric values with column mean
for col in emg_cols:
    df[col] = df[col].fillna(df[col].mean())

# # Detect outliers using Inter Quartile Range (IQR)
# for col in emg_cols:
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1

#     df = df[~((df[col] < (Q1 - 4 * IQR)) | (df[col] > (Q3 + 4 * IQR)))]

# Standardization
scaler = StandardScaler()
df[emg_cols] = scaler.fit_transform(df[emg_cols])

print("Standardized EMG columns:")
print(df[emg_cols].head())

# Label Encoding
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['gesture'])
print("\nEncoded labels:")
print(df[['label', 'gesture']].drop_duplicates())

# ============================= Train/Test Split =============================

# Use the cleaned and prepared dataframe from previous cells
# Separate features (X) and labels (y)
X = df[emg_cols]
y = df['label']

# Split dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ============================= Model Evaluation =============================

def evaluate_model(model, X_val, y_val):
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    print("Validation accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))
	
    # Cross-validation
    scores = cross_val_score(model, X_val, y_val, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean accuracy:", scores.mean())


    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


    # Predict probabilities for the first 5 samples in the validation set
    probs = model.predict_proba(X_val[:5])
    print(probs)


    # ROC Curve
    y_val_bin = label_binarize(y_val, classes=model.classes_)
    probs_full = model.predict_proba(X_val)

    plt.figure()
    for i in range(y_val_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], probs_full[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend()
    plt.show()


# ============================= Model Training =============================


# Calculate class weights to handle imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print("Class weights:", class_weight_dict)

# Improved neural network with more layers and regularization
mlp_improved = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    alpha=0.001,                          # L2 regularization
    learning_rate_init=0.001,             # Lower learning rate
    max_iter=5000,
    early_stopping=True,                  # Stop when validation score stops improving
    validation_fraction=0.2,              # Use 20% of training data for validation
    n_iter_no_change=50,                  # Stop after 50 iterations without improvement
    random_state=42,
    batch_size='auto'
)

print("Training improved neural network...")
mlp_improved.fit(X_train, y_train)

print("Training completed!")
print(f"Number of iterations: {mlp_improved.n_iter_}")
print(f"Final loss: {mlp_improved.loss_:.4f}")



# Evaluate the improved model
print("\n=== IMPROVED NEURAL NETWORK RESULTS ===")
evaluate_model(mlp_improved, X_val, y_val)
