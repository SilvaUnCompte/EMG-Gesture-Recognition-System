import pandas as pd # pyright: ignore[reportMissingModuleSource]
from sklearn.preprocessing import LabelEncoder, StandardScaler # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv", sep=";")
print(df.shape)  # dimensions
print(df.info())  # types and missing values
print(df.head())  # 5 first lines

emg_cols = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8']
df = df[emg_cols + ['gesture']]
df = df.drop_duplicates()  # remove duplicate rows
df = df.dropna() # remove rows with missing values



# =========== Standardize data ===========

def standardize_data(df, emg_cols):
    """
    Standardizes the EMG columns in the DataFrame.
    
    Parameters:
    df (DataFrame): The input DataFrame containing EMG data.
    emg_cols (list): List of EMG column names to standardize.
    
    Returns:
    DataFrame: A new DataFrame with standardized EMG columns.
    """
    scaler = StandardScaler()
    df[emg_cols] = scaler.fit_transform(df[emg_cols])
    return df

