from sklearn.model_selection import train_test_split
from sklearn.calibration import LabelEncoder
import pandas as pd
import sys
import os

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

# ========= Remove useless data =========

def drop_outside_scope_data(df, gesture_col, threshold_col):
    """
    Drops rows from the DataFrame that are outside the scope defined by the gesture and threshold columns.

    Parameters:
    df (DataFrame): The input DataFrame containing gesture and threshold information.
    gesture_col (str): The name of the column containing gesture labels.
    threshold_col (str): The name of the column containing threshold values.

    Returns:
    DataFrame: A new DataFrame with rows outside the defined scope removed.
    """
    # Define the scope (Can be customized)
    valid_gestures = df[gesture_col] != "NULL"
    valid_thresholds = df[threshold_col] == "above"
    
    # Combine the valid gestures and thresholds
    valid_rows = df[valid_gestures & valid_thresholds]

    # Keep only the rows that are within the defined scope
    return valid_rows

# =========== Encode labels ===========

def encode_labels(df, GESTURE_COL):
    """
    Encodes the gesture labels in the DataFrame using LabelEncoder.

    Parameters:
    df (DataFrame): The input DataFrame containing gesture labels.

    Returns:
    DataFrame: A new DataFrame with an additional 'label' column containing encoded labels.
    """

    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df[GESTURE_COL])
    class_names = encoder.classes_.tolist()
    print("\nEncoded labels:")
    print(df[['label', GESTURE_COL]].drop_duplicates())

    return df, class_names

# =========== Split data ===========

def splitting_data(df, feature_names, test_size=0.25):
    """
    Splits the DataFrame into training and test sets.

    Parameters:
    df (DataFrame): The input DataFrame with features and encoded labels.
    feature_names (list): The list of feature column names to use for training.
    test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    Tuple: X_train, y_train, X_test, y_test
    """

    X = df[feature_names]
    y = df['label']

    # Split dataset into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, y_train, X_test, y_test
