"""
EMG Gesture Recognition System
============================

A comprehensive machine learning pipeline for EMG gesture recognition.
Includes data preprocessing, feature engineering, multiple ML models, and ensemble methods.

Author: Generated from Jupyter notebook
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Tuple, Dict, List, Any
from scipy import stats

# Scikit-learn imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (confusion_matrix, roc_curve, auc, accuracy_score, 
                           classification_report)
from sklearn.preprocessing import label_binarize


class EMGDataProcessor:
    """Handles EMG data loading, cleaning, and preprocessing."""
    
    def __init__(self, data_file: str = "data3.csv"):
        self.data_file = data_file
        self.emg_cols = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8']
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self) -> pd.DataFrame:
        """Load and display basic information about the dataset."""
        print("=== LOADING DATA ===")
        self.df = pd.read_csv(self.data_file, sep=";")
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Data types:\n{self.df.dtypes}")
        print(f"\nFirst 5 rows:\n{self.df.head()}")
        
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """Clean the dataset by removing duplicates and handling missing values."""
        print("\n=== CLEANING DATA ===")
        initial_shape = self.df.shape
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        print(f"Removed {initial_shape[0] - self.df.shape[0]} duplicate rows")
        
        # Handle missing values
        self.df = self.df.dropna()
        print(f"Removed rows with missing values. Final shape: {self.df.shape}")
        
        # Fill missing EMG values with column mean (backup)
        for col in self.emg_cols:
            if self.df[col].isnull().any():
                mean_val = self.df[col].mean()
                self.df[col] = self.df[col].fillna(mean_val)
                print(f"Filled missing values in {col} with mean: {mean_val:.2f}")
        
        return self.df
    
    def remove_outliers(self, method: str = "conservative", iqr_multiplier: float = 2.5) -> pd.DataFrame:
        """
        Remove outliers using different methods.
        
        Args:
            method: 'conservative' (2.5*IQR), 'standard' (1.5*IQR), 'aggressive' (1.0*IQR), 
                   'percentile' (1st-99th percentile), 'z_score' (3 standard deviations), 'none' (no removal)
            iqr_multiplier: Custom multiplier for IQR method
        """
        print(f"\n=== REMOVING OUTLIERS ({method.upper()}) ===")
        initial_shape = self.df.shape
        
        # Analyze class distribution before outlier removal
        print("Class distribution BEFORE outlier removal:")
        for gesture in self.df['gesture'].unique():
            count = (self.df['gesture'] == gesture).sum()
            print(f"  {gesture}: {count} samples")
        
        if method == "none":
            print("No outlier removal applied.")
            return self.df
        
        elif method == "conservative":
            # More conservative: 2.5 * IQR (keeps more data)
            multiplier = 2.5
            print(f"Using conservative IQR method (±{multiplier} * IQR)")
            
            for col in self.emg_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound))
                outlier_count = outliers.sum()
                self.df = self.df[~outliers]
                
                print(f"  {col}: Removed {outlier_count} outliers (bounds: {lower_bound:.1f} to {upper_bound:.1f})")
        
        elif method == "standard":
            # Standard method: 1.5 * IQR
            multiplier = 1.5
            print(f"Using standard IQR method (±{multiplier} * IQR)")
            
            for col in self.emg_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound))
                outlier_count = outliers.sum()
                self.df = self.df[~outliers]
                
                print(f"  {col}: Removed {outlier_count} outliers")
        
        elif method == "percentile":
            # Remove extreme 1% on each side
            print("Using percentile method (removing bottom 1% and top 1%)")
            
            for col in self.emg_cols:
                p1 = self.df[col].quantile(0.01)
                p99 = self.df[col].quantile(0.99)
                
                outliers = ((self.df[col] < p1) | (self.df[col] > p99))
                outlier_count = outliers.sum()
                self.df = self.df[~outliers]
                
                print(f"  {col}: Removed {outlier_count} outliers (bounds: {p1:.1f} to {p99:.1f})")
        
        elif method == "z_score":
            # Z-score method: remove values > 3 standard deviations
            print("Using Z-score method (removing |z| > 3)")
            
            for col in self.emg_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                z_scores = np.abs((self.df[col] - mean_val) / std_val)
                
                outliers = z_scores > 3
                outlier_count = outliers.sum()
                self.df = self.df[~outliers]
                
                print(f"  {col}: Removed {outlier_count} outliers (mean±3σ: {mean_val-3*std_val:.1f} to {mean_val+3*std_val:.1f})")
        
        elif method == "custom":
            # Custom IQR multiplier
            print(f"Using custom IQR method (±{iqr_multiplier} * IQR)")
            
            for col in self.emg_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound))
                outlier_count = outliers.sum()
                self.df = self.df[~outliers]
                
                print(f"  {col}: Removed {outlier_count} outliers")
        
        else:
            print("No outlier removal applied")
            return self.df
        
        print(f"\nTotal rows removed: {initial_shape[0] - self.df.shape[0]}")
        print(f"Final shape after outlier removal: {self.df.shape}")
        
        # Analyze class distribution after outlier removal
        print("\nClass distribution AFTER outlier removal:")
        for gesture in self.df['gesture'].unique():
            count = (self.df['gesture'] == gesture).sum()
            percentage = (count / len(self.df)) * 100
            print(f"  {gesture}: {count} samples ({percentage:.1f}%)")
        
        return self.df
    
    def analyze_outlier_impact(self) -> None:
        """Analyze the impact of different outlier removal methods."""
        print("\n=== ANALYZING OUTLIER REMOVAL IMPACT ===")
        
        # Save original data
        original_df = self.df.copy()
        methods = ['none', 'conservative', 'standard', 'percentile', 'z_score']
        results = {}
        
        for method in methods:
            # Reset to original data
            self.df = original_df.copy()
            
            if method != 'none':
                self.remove_outliers(method=method)
            
            # Store results
            results[method] = {
                'total_samples': len(self.df),
                'class_distribution': self.df['gesture'].value_counts().to_dict()
            }
        
        # Display comparison table
        print(f"\n{'Method':<15} {'Total':<8} {'Down':<6} {'Close':<6} {'Open':<6} {'Pin':<6}")
        print("-" * 55)
        
        for method, data in results.items():
            total = data['total_samples']
            down = data['class_distribution'].get('down', 0)
            close = data['class_distribution'].get('close', 0)
            open_g = data['class_distribution'].get('open', 0)
            pin = data['class_distribution'].get('pin', 0)
            
            print(f"{method:<15} {total:<8} {down:<6} {close:<6} {open_g:<6} {pin:<6}")
        
        # Restore original data
        self.df = original_df.copy()
        print(f"\nRecommendation: Use 'conservative' method to preserve more 'down' samples")
        print(f"Or use 'percentile' method for a balanced approach")
    
    def remove_outliers_by_class(self, method: str = "conservative") -> pd.DataFrame:
        """Remove outliers separately for each gesture class to preserve minority classes."""
        print(f"\n=== REMOVING OUTLIERS BY CLASS ({method.upper()}) ===")
        initial_shape = self.df.shape
        
        print("Class distribution BEFORE outlier removal:")
        for gesture in self.df['gesture'].unique():
            count = (self.df['gesture'] == gesture).sum()
            print(f"  {gesture}: {count} samples")
        
        cleaned_dfs = []
        
        for gesture in self.df['gesture'].unique():
            gesture_data = self.df[self.df['gesture'] == gesture].copy()
            initial_count = len(gesture_data)
            
            print(f"\nProcessing {gesture} class ({initial_count} samples):")
            
            if method == "conservative":
                multiplier = 2.5
            elif method == "standard":
                multiplier = 1.5
            else:
                multiplier = 2.0
            
            for col in self.emg_cols:
                Q1 = gesture_data[col].quantile(0.25)
                Q3 = gesture_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                outliers = ((gesture_data[col] < lower_bound) | (gesture_data[col] > upper_bound))
                outlier_count = outliers.sum()
                gesture_data = gesture_data[~outliers]
                
                if outlier_count > 0:
                    print(f"  {col}: Removed {outlier_count} outliers")
            
            final_count = len(gesture_data)
            print(f"  {gesture}: {initial_count} → {final_count} samples ({final_count/initial_count*100:.1f}% retained)")
            cleaned_dfs.append(gesture_data)
        
        # Combine all cleaned class data
        self.df = pd.concat(cleaned_dfs, ignore_index=True)
        
        print(f"\nTotal rows removed: {initial_shape[0] - self.df.shape[0]}")
        print(f"Final shape after class-wise outlier removal: {self.df.shape}")
        
        print("\nClass distribution AFTER outlier removal:")
        for gesture in self.df['gesture'].unique():
            count = (self.df['gesture'] == gesture).sum()
            percentage = (count / len(self.df)) * 100
            print(f"  {gesture}: {count} samples ({percentage:.1f}%)")
        
        return self.df
    
    def standardize_features(self) -> pd.DataFrame:
        """Standardize EMG features to have mean=0 and std=1."""
        print("\n=== STANDARDIZING FEATURES ===")
        
        self.df[self.emg_cols] = self.scaler.fit_transform(self.df[self.emg_cols])
        
        print("Standardized EMG columns:")
        print(f"Mean values: {self.df[self.emg_cols].mean().round(3).tolist()}")
        print(f"Std values: {self.df[self.emg_cols].std().round(3).tolist()}")
        
        return self.df
    
    def encode_labels(self) -> pd.DataFrame:
        """Encode gesture labels to numeric values."""
        print("\n=== ENCODING LABELS ===")
        
        self.df['label'] = self.label_encoder.fit_transform(self.df['gesture'])
        
        # Show label mapping
        label_mapping = dict(zip(self.label_encoder.classes_, 
                               self.label_encoder.transform(self.label_encoder.classes_)))
        print("Label mapping:")
        for gesture, label in label_mapping.items():
            count = (self.df['gesture'] == gesture).sum()
            print(f"  {gesture} → {label} ({count} samples)")
        
        return self.df
    
    def analyze_data_distribution(self) -> None:
        """Analyze and visualize data distribution."""
        print("\n=== DATA DISTRIBUTION ANALYSIS ===")
        
        # Class distribution
        class_counts = self.df['gesture'].value_counts()
        print("Class distribution:")
        for gesture, count in class_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {gesture:10s}: {count:4d} samples ({percentage:5.1f}%)")
        
        # Visualize distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Class distribution
        class_counts.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Class Distribution')
        axes[0, 0].set_xlabel('Gesture')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # EMG signal characteristics by gesture
        for gesture in self.df['gesture'].unique():
            gesture_data = self.df[self.df['gesture'] == gesture]
            mean_emg = gesture_data[self.emg_cols].mean().mean()
            std_emg = gesture_data[self.emg_cols].std().mean()
            axes[0, 1].scatter(mean_emg, std_emg, label=gesture, s=100)
        
        axes[0, 1].set_xlabel('Mean EMG Signal')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].set_title('EMG Signal Characteristics by Gesture')
        axes[0, 1].legend()
        
        # Correlation matrix
        correlation_matrix = self.df[self.emg_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, ax=axes[1, 0])
        axes[1, 0].set_title('EMG Channels Correlation Matrix')
        
        # Signal quality analysis
        ranges = []
        stds = []
        for col in self.emg_cols:
            signal_range = self.df[col].max() - self.df[col].min()
            signal_std = self.df[col].std()
            ranges.append(signal_range)
            stds.append(signal_std)
        
        axes[1, 1].bar(self.emg_cols, ranges, alpha=0.7, label='Range')
        axes[1, 1].set_title('Signal Range by EMG Channel')
        axes[1, 1].set_xlabel('EMG Channel')
        axes[1, 1].set_ylabel('Signal Range')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print signal quality metrics
        print("\nSignal quality analysis:")
        for i, col in enumerate(self.emg_cols):
            snr_estimate = abs(self.df[col].mean()) / stds[i] if stds[i] > 0 else 0
            print(f"  {col}: Range={ranges[i]:6.1f}, STD={stds[i]:6.2f}, SNR≈{snr_estimate:.2f}")


class EMGFeatureExtractor:
    """Extracts advanced features from EMG signals."""
    
    def __init__(self, emg_columns: List[str]):
        self.emg_cols = emg_columns
        self.scaler_enhanced = StandardScaler()
    
    def extract_emg_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract advanced time-domain features from EMG signals."""
        features = []
        
        for col in self.emg_cols:
            signal = data[col].values
            
            # Time domain features
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            rms = np.sqrt(np.mean(signal**2))  # Root Mean Square
            variance = np.var(signal)
            skewness = stats.skew(signal)
            kurtosis = stats.kurtosis(signal)
            
            # Statistical features
            percentile_25 = np.percentile(signal, 25)
            percentile_75 = np.percentile(signal, 75)
            median = np.median(signal)
            range_val = np.max(signal) - np.min(signal)
            
            # Add features for this channel
            features.extend([mean_val, std_val, rms, variance, skewness, kurtosis,
                           percentile_25, percentile_75, median, range_val])
        
        return np.array(features)
    
    def create_enhanced_dataset(self, df: pd.DataFrame, min_samples_per_window: int = 5, 
                               window_overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create enhanced feature matrix by grouping data by session and gesture.
        Uses sliding windows to create more samples.
        
        Args:
            df: Input dataframe
            min_samples_per_window: Minimum samples needed per window
            window_overlap: Overlap between windows (0.5 = 50% overlap)
        """
        print("\n=== EXTRACTING ENHANCED FEATURES ===")
        
        X_enhanced_list = []
        y_enhanced_list = []
        samples_per_class = {}
        
        for session_id in df['SessionID'].unique():
            session_data = df[df['SessionID'] == session_id]
            
            for gesture in session_data['gesture'].unique():
                gesture_data = session_data[session_data['gesture'] == gesture]
                gesture_samples = 0
                
                data_length = len(gesture_data)
                
                if data_length >= min_samples_per_window:
                    # Method 1: Use full session data
                    features = self.extract_emg_features(gesture_data)
                    X_enhanced_list.append(features)
                    y_enhanced_list.append(gesture_data['label'].iloc[0])
                    gesture_samples += 1
                    
                    # Method 2: Create sliding windows for more samples (if enough data)
                    if data_length >= min_samples_per_window * 2:
                        window_size = max(min_samples_per_window, data_length // 3)
                        step_size = max(1, int(window_size * (1 - window_overlap)))
                        
                        for start_idx in range(0, data_length - window_size + 1, step_size):
                            end_idx = start_idx + window_size
                            window_data = gesture_data.iloc[start_idx:end_idx]
                            
                            if len(window_data) >= min_samples_per_window:
                                features = self.extract_emg_features(window_data)
                                X_enhanced_list.append(features)
                                y_enhanced_list.append(gesture_data['label'].iloc[0])
                                gesture_samples += 1
                
                # Track samples per class
                if gesture not in samples_per_class:
                    samples_per_class[gesture] = 0
                samples_per_class[gesture] += gesture_samples
        
        X_enhanced = np.array(X_enhanced_list)
        y_enhanced = np.array(y_enhanced_list)
        
        print(f"Enhanced feature matrix shape: {X_enhanced.shape}")
        print(f"Number of features per sample: {X_enhanced.shape[1]}")
        print(f"Features per EMG channel: {X_enhanced.shape[1] // len(self.emg_cols)}")
        
        print(f"\nSamples created per class:")
        for gesture, count in samples_per_class.items():
            print(f"  {gesture}: {count} enhanced samples")
        
        if len(X_enhanced) == 0:
            print("Warning: No enhanced features could be created!")
            return np.array([]), np.array([])
        
        # Standardize enhanced features
        X_enhanced_scaled = self.scaler_enhanced.fit_transform(X_enhanced)
        
        return X_enhanced_scaled, y_enhanced


class EMGModelEvaluator:
    """Handles model evaluation and visualization."""
    
    @staticmethod
    def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray, 
                      model_name: str = "Model", show_plots: bool = True) -> Dict[str, float]:
        """Comprehensive model evaluation with metrics and visualizations."""
        print(f"\n=== {model_name.upper()} EVALUATION ===")
        
        # Basic predictions
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Validation accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        
        # Cross-validation (only if we have enough samples)
        n_samples = len(X_val)
        if n_samples >= 10:  # Need at least 10 samples for 5-fold CV
            cv_folds = min(5, n_samples // 2)  # Adjust CV folds based on sample size
            cv_scores = cross_val_score(model, X_val, y_val, cv=cv_folds)
            print(f"\nCross-validation scores ({cv_folds}-fold): {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        else:
            print(f"\nWarning: Too few samples ({n_samples}) for cross-validation. Skipping CV.")
            cv_scores = np.array([accuracy])  # Use single accuracy as fallback
        
        if show_plots:
            # Confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix - {model_name}")
            
            # ROC Curve for multiclass
            if hasattr(model, 'predict_proba'):
                y_val_bin = label_binarize(y_val, classes=model.classes_)
                probs_full = model.predict_proba(X_val)
                
                plt.subplot(1, 2, 2)
                for i in range(y_val_bin.shape[1]):
                    fpr, tpr, _ = roc_curve(y_val_bin[:, i], probs_full[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
                
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curves - {model_name}")
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    @staticmethod
    def plot_learning_curve(model, X: np.ndarray, y: np.ndarray, 
                          title: str = "Learning Curve") -> None:
        """Plot learning curve to analyze model performance vs dataset size."""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=42
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training accuracy')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation accuracy')
        plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
        plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


class EMGClassifier:
    """Main EMG gesture classification system."""
    
    def __init__(self, data_file: str = "data3.csv"):
        self.data_processor = EMGDataProcessor(data_file)
        self.feature_extractor = EMGFeatureExtractor(self.data_processor.emg_cols)
        self.evaluator = EMGModelEvaluator()
        
        # Data storage
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Enhanced features
        self.X_enh_train = None
        self.X_enh_val = None
        self.X_enh_test = None
        self.y_enh_train = None
        self.y_enh_val = None
        self.y_enh_test = None
        
        # Models
        self.models = {}
        self.best_model = None
        
    def prepare_data(self, analyze: bool = True, outlier_method: str = "conservative", 
                    analyze_outliers: bool = True, skip_enhanced_features: bool = False) -> None:
        """
        Complete data preparation pipeline.
        
        Args:
            analyze: Whether to show data analysis plots
            outlier_method: Method for outlier removal ('conservative', 'standard', 'percentile', 'z_score', 'by_class')
            analyze_outliers: Whether to analyze the impact of different outlier methods
        """
        print("Starting EMG data preparation pipeline...")
        
        # Load and clean data
        self.df = self.data_processor.load_data()
        print(f"📊 After loading: {self.df.shape}")
        
        self.df = self.data_processor.clean_data()
        print(f"📊 After cleaning: {self.df.shape}")
        
        # Analyze outlier impact if requested
        if analyze_outliers:
            self.data_processor.analyze_outlier_impact()
        
        # Remove outliers using specified method
        if outlier_method == "by_class":
            self.df = self.data_processor.remove_outliers_by_class(method="conservative")
        else:
            self.df = self.data_processor.remove_outliers(method=outlier_method)
        
        print(f"📊 After outlier removal: {self.df.shape}")
        
        # Check if we have enough data
        if len(self.df) < 50:
            print(f"⚠️  WARNING: Very few samples remaining ({len(self.df)}). Consider using a less aggressive outlier method.")
        
        self.df = self.data_processor.standardize_features()
        self.df = self.data_processor.encode_labels()
        
        # Print final class distribution
        print(f"📊 Final class distribution:")
        for gesture in self.df['gesture'].unique():
            count = (self.df['gesture'] == gesture).sum()
            percentage = (count / len(self.df)) * 100
            print(f"  {gesture}: {count} samples ({percentage:.1f}%)")
        
        if analyze:
            self.data_processor.analyze_data_distribution()
        
        # Split basic features
        X = self.df[self.data_processor.emg_cols]
        y = self.df['label']
        
        # Check if we have enough samples for stratified split
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_size = np.min(class_counts)
        
        if min_class_size < 2:
            print(f"⚠️  WARNING: Some classes have less than 2 samples. Using random split instead of stratified.")
            self.X_train, X_temp, self.y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
        else:
            self.X_train, X_temp, self.y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
        
        print(f"\nBasic features split:")
        print(f"  Training set: {self.X_train.shape}")
        print(f"  Validation set: {self.X_val.shape}")
        print(f"  Test set: {self.X_test.shape}")
        
        # Create enhanced features if requested
        if not skip_enhanced_features:
            X_enhanced, y_enhanced = self.feature_extractor.create_enhanced_dataset(self.df)
            
            if len(X_enhanced) > 0:
                # Check if we have enough samples per class for stratified split
                unique_classes, class_counts = np.unique(y_enhanced, return_counts=True)
                min_class_size = np.min(class_counts)
                
                print(f"\nEnhanced features class distribution:")
                for cls, count in zip(unique_classes, class_counts):
                    gesture_name = self.data_processor.label_encoder.inverse_transform([cls])[0]
                    print(f"  {gesture_name}: {count} samples")
                
                if min_class_size >= 2:
                    # Sufficient samples for stratified split
                    try:
                        self.X_enh_train, X_enh_temp, self.y_enh_train, y_enh_temp = train_test_split(
                            X_enhanced, y_enhanced, test_size=0.3, random_state=42, stratify=y_enhanced
                        )
                        self.X_enh_val, self.X_enh_test, self.y_enh_val, self.y_enh_test = train_test_split(
                            X_enh_temp, y_enh_temp, test_size=0.5, random_state=42, stratify=y_enh_temp
                        )
                        
                        print(f"\nEnhanced features split (stratified):")
                        print(f"  Training set: {self.X_enh_train.shape}")
                        print(f"  Validation set: {self.X_enh_val.shape}")
                        print(f"  Test set: {self.X_enh_test.shape}")
                    except ValueError as e:
                        print(f"Warning: Stratified split failed, using random split: {e}")
                        # Fallback to random split
                        self.X_enh_train, X_enh_temp, self.y_enh_train, y_enh_temp = train_test_split(
                            X_enhanced, y_enhanced, test_size=0.3, random_state=42
                        )
                        self.X_enh_val, self.X_enh_test, self.y_enh_val, self.y_enh_test = train_test_split(
                            X_enh_temp, y_enh_temp, test_size=0.5, random_state=42
                        )
                        
                        print(f"\nEnhanced features split (random):")
                        print(f"  Training set: {self.X_enh_train.shape}")
                        print(f"  Validation set: {self.X_enh_val.shape}")
                        print(f"  Test set: {self.X_enh_test.shape}")
                else:
                    print(f"Warning: Minimum class size ({min_class_size}) too small for enhanced features.")
                    print("Enhanced features will be skipped. Using only basic features.")
                    self.X_enh_train = None
                    self.X_enh_val = None
                    self.X_enh_test = None
                    self.y_enh_train = None
                    self.y_enh_val = None
                    self.y_enh_test = None
            else:
                print("Warning: No enhanced features created. Using only basic features.")
                self.X_enh_train = None
                self.X_enh_val = None
                self.X_enh_test = None
                self.y_enh_train = None
                self.y_enh_val = None
                self.y_enh_test = None
        else:
            print("Enhanced features skipped by user request. Using only basic features.")
            self.X_enh_train = None
            self.X_enh_val = None
            self.X_enh_test = None
            self.y_enh_train = None
            self.y_enh_val = None
            self.y_enh_test = None
            self.y_enh_val = None
            self.y_enh_test = None
    
    def train_baseline_model(self) -> None:
        """Train and optimize baseline Decision Tree model."""
        print("\n" + "="*60)
        print("TRAINING BASELINE DECISION TREE MODEL")
        print("="*60)
        
        # Grid search for best parameters
        param_grid = {
            "max_depth": [2, 4, 6, 8, 10, None],
            "criterion": ["gini", "entropy"]
        }
        
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42), 
            param_grid, cv=5, scoring="accuracy"
        )
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best validation accuracy: {grid_search.best_score_:.4f}")
        
        # Train final model with best parameters
        self.models['Decision Tree'] = DecisionTreeClassifier(
            random_state=42,
            criterion=grid_search.best_params_["criterion"],
            max_depth=grid_search.best_params_["max_depth"]
        )
        self.models['Decision Tree'].fit(self.X_train, self.y_train)
        
        # Evaluate
        self.evaluator.evaluate_model(
            self.models['Decision Tree'], self.X_val, self.y_val, "Decision Tree"
        )
    
    def train_neural_networks(self) -> None:
        """Train multiple neural network architectures."""
        print("\n" + "="*60)
        print("TRAINING NEURAL NETWORK MODELS")
        print("="*60)
        
        # Basic Neural Network
        print("\n--- Basic Neural Network ---")
        self.models['Basic NN'] = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation='relu',
            solver='adam',
            max_iter=3000,
            random_state=42
        )
        self.models['Basic NN'].fit(self.X_train, self.y_train)
        self.evaluator.evaluate_model(
            self.models['Basic NN'], self.X_val, self.y_val, "Basic Neural Network"
        )
        
        # Improved Neural Network
        print("\n--- Improved Neural Network ---")
        self.models['Improved NN'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=5000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            random_state=42
        )
        self.models['Improved NN'].fit(self.X_train, self.y_train)
        
        print(f"Training iterations: {self.models['Improved NN'].n_iter_}")
        print(f"Final loss: {self.models['Improved NN'].loss_:.4f}")
        
        self.evaluator.evaluate_model(
            self.models['Improved NN'], self.X_val, self.y_val, "Improved Neural Network"
        )
        
        # Enhanced Features Neural Network
        if self.X_enh_train is not None:
            print("\n--- Enhanced Features Neural Network ---")
            self.models['Enhanced NN'] = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=5000,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=100,
                random_state=42
            )
            self.models['Enhanced NN'].fit(self.X_enh_train, self.y_enh_train)
            self.evaluator.evaluate_model(
                self.models['Enhanced NN'], self.X_enh_val, self.y_enh_val, 
                "Enhanced Features Neural Network"
            )
    
    def train_advanced_models(self) -> None:
        """Train Random Forest and SVM models."""
        print("\n" + "="*60)
        print("TRAINING ADVANCED MODELS")
        print("="*60)
        
        # Use enhanced features if available, otherwise basic features
        if self.X_enh_train is not None:
            X_train, X_val = self.X_enh_train, self.X_enh_val
            y_train, y_val = self.y_enh_train, self.y_enh_val
            feature_type = "Enhanced"
        else:
            X_train, X_val = self.X_train, self.X_val
            y_train, y_val = self.y_train, self.y_val
            feature_type = "Basic"
        
        print(f"Using {feature_type} features")
        
        # Random Forest
        print("\n--- Random Forest ---")
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        start_time = time.time()
        self.models['Random Forest'].fit(X_train, y_train)
        rf_time = time.time() - start_time
        
        rf_results = self.evaluator.evaluate_model(
            self.models['Random Forest'], X_val, y_val, "Random Forest"
        )
        print(f"Training time: {rf_time:.2f}s")
        
        # SVM
        print("\n--- Support Vector Machine ---")
        self.models['SVM'] = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        
        start_time = time.time()
        self.models['SVM'].fit(X_train, y_train)
        svm_time = time.time() - start_time
        
        svm_results = self.evaluator.evaluate_model(
            self.models['SVM'], X_val, y_val, "Support Vector Machine"
        )
        print(f"Training time: {svm_time:.2f}s")
        
        # Feature importance from Random Forest
        if hasattr(self.models['Random Forest'], 'feature_importances_'):
            print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
            if self.X_enh_train is not None:
                feature_names = [f"Feature_{i}" for i in range(self.X_enh_train.shape[1])]
            else:
                feature_names = self.data_processor.emg_cols
            
            importance_pairs = list(zip(feature_names, self.models['Random Forest'].feature_importances_))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(importance_pairs[:10]):
                print(f"{i+1:2d}. {feature:15s}: {importance:.4f}")
    
    def create_ensemble_model(self) -> None:
        """Create and train ensemble model combining multiple algorithms."""
        print("\n" + "="*60)
        print("CREATING ENSEMBLE MODEL")
        print("="*60)
        
        # Use enhanced features if available
        if self.X_enh_train is not None:
            X_train, X_val = self.X_enh_train, self.X_enh_val
            y_train, y_val = self.y_enh_train, self.y_enh_val
        else:
            X_train, X_val = self.X_train, self.X_val
            y_train, y_val = self.y_train, self.y_val
        
        # Create individual models for ensemble
        rf_ensemble = RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight='balanced', random_state=42
        )
        svm_ensemble = SVC(
            kernel='rbf', C=10, class_weight='balanced', probability=True, random_state=42
        )
        mlp_ensemble = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32, 16),
            activation='relu', solver='adam', alpha=0.001,
            early_stopping=True, random_state=42
        )
        
        # Create ensemble
        self.models['Ensemble'] = VotingClassifier(
            estimators=[
                ('rf', rf_ensemble),
                ('svm', svm_ensemble),
                ('mlp', mlp_ensemble)
            ],
            voting='soft'
        )
        
        print("Training ensemble model...")
        start_time = time.time()
        self.models['Ensemble'].fit(X_train, y_train)
        ensemble_time = time.time() - start_time
        
        ensemble_results = self.evaluator.evaluate_model(
            self.models['Ensemble'], X_val, y_val, "Ensemble Model"
        )
        print(f"Training time: {ensemble_time:.2f}s")
    
    def compare_all_models(self) -> None:
        """Compare performance of all trained models."""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            if 'Enhanced' in name and self.X_enh_val is not None:
                X_val, y_val = self.X_enh_val, self.y_enh_val
            else:
                X_val, y_val = self.X_val, self.y_val
            
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            results[name] = accuracy
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Model':<25} {'Accuracy':<10}")
        print("-" * 40)
        for name, accuracy in sorted_results:
            print(f"{name:<25} {accuracy:<10.4f}")
        
        # Identify best model
        self.best_model = sorted_results[0][0]
        print(f"\nBest model: {self.best_model} ({sorted_results[0][1]:.4f})")
    
    def print_recommendations(self) -> None:
        """Print recommendations for further improvement."""
        print("\n" + "="*60)
        print("RECOMMENDATIONS FOR IMPROVEMENT")
        print("="*60)
        
        recommendations = [
            "📊 COLLECT MORE DATA for underrepresented classes",
            "🔄 USE DATA AUGMENTATION (add noise, time shifting)",
            "⚡ TRY TEMPORAL FEATURES (RNN/LSTM for sequence data)",
            "🎯 OPTIMIZE HYPERPARAMETERS with more extensive grid search",
            "🧪 EXPERIMENT with different preprocessing (filters, normalization)",
            "📈 USE CROSS-VALIDATION for more robust evaluation",
            "🔍 ANALYZE MISCLASSIFIED SAMPLES for insights",
            "🎛️ TRY DIFFERENT FEATURE SCALING methods",
            "🔧 IMPLEMENT FEATURE SELECTION techniques",
            "📚 CONSIDER DOMAIN-SPECIFIC EMG features"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")
    
    def run_complete_pipeline(self, analyze_data: bool = True, outlier_method: str = "conservative", 
                            use_enhanced_features: bool = False) -> None:
        """
        Run the complete EMG classification pipeline.
        
        Args:
            analyze_data: Whether to show data analysis plots
            outlier_method: Method for outlier removal ('conservative', 'standard', 'by_class', etc.)
            use_enhanced_features: Whether to use enhanced features (may cause issues with small datasets)
        """
        print("Starting EMG Gesture Recognition Pipeline")
        print("=" * 80)
        
        # Data preparation with conservative outlier removal
        self.prepare_data(analyze=analyze_data, outlier_method=outlier_method, 
                         skip_enhanced_features=not use_enhanced_features)
        
        # Check if we have enough data for enhanced features
        if use_enhanced_features and len(self.df) < 1000:
            print(f"⚠️  Warning: Dataset too small ({len(self.df)} samples) for enhanced features.")
            print("Using basic features only for better stability.")
            use_enhanced_features = False
        
        # Train all models
        self.train_baseline_model()
        self.train_neural_networks()
        self.train_advanced_models()
        self.create_ensemble_model()
        
        # Compare and summarize
        self.compare_all_models()
        self.print_recommendations()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)


def quick_outlier_analysis(data_file: str = "data3.csv"):
    """Quick function to analyze outlier impact only."""
    print("Quick Outlier Analysis")
    print("=" * 30)
    
    processor = EMGDataProcessor(data_file)
    df = processor.load_data()
    df = processor.clean_data()
    processor.df = df
    processor.analyze_outlier_impact()


def test_conservative_method(data_file: str = "data3.csv"):
    """Test specifically the conservative method for 'down' class preservation."""
    print("Testing Conservative Outlier Removal")
    print("=" * 40)
    
    try:
        classifier = EMGClassifier(data_file)
        print("✓ EMGClassifier initialized")
        
        # Load data first to show initial distribution
        classifier.df = classifier.data_processor.load_data()
        print("✓ Data loaded")
        
        classifier.df = classifier.data_processor.clean_data()
        print("✓ Data cleaned")
        
        # Show distribution before outlier removal
        print("\nClass distribution BEFORE outlier removal:")
        for gesture in classifier.df['gesture'].unique():
            count = (classifier.df['gesture'] == gesture).sum()
            print(f"  {gesture}: {count} samples")
        
        # Apply conservative outlier removal
        classifier.df = classifier.data_processor.remove_outliers(method="conservative")
        print("✓ Conservative outlier removal applied")
        
        # Show final class distribution
        print("\nFinal class distribution with conservative method:")
        for gesture in classifier.df['gesture'].unique():
            count = (classifier.df['gesture'] == gesture).sum()
            percentage = (count / len(classifier.df)) * 100
            print(f"  {gesture}: {count} samples ({percentage:.1f}%)")
            
        print(f"\nTotal samples retained: {len(classifier.df)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the EMG classification system."""
    print("EMG Gesture Recognition System")
    print("=" * 50)
    
    # Initialize classifier
    classifier = EMGClassifier("data3.csv")
    
    # Ask user about outlier method
    print("\nChoose outlier removal method:")
    print("1. Conservative (2.5*IQR) - Keeps more data")
    print("2. Standard (1.5*IQR) - Traditional method")
    print("3. By class - Process each gesture separately")
    print("4. Percentile - Remove extreme 1% on each side")
    print("5. None - Keep all data")
    print("6. Analyze all methods first")
    
    choice = input("Enter choice (1-6) or press Enter for conservative: ").strip()
    
    if choice == "2":
        outlier_method = "standard"
    elif choice == "3":
        outlier_method = "by_class"
    elif choice == "4":
        outlier_method = "percentile"
    elif choice == "5":
        outlier_method = "none"
    elif choice == "6":
        # Just analyze outlier impact and exit
        classifier.df = classifier.data_processor.load_data()
        classifier.df = classifier.data_processor.clean_data()
        classifier.data_processor.analyze_outlier_impact()
        return
    else:
        outlier_method = "conservative"
    
    print(f"\nUsing {outlier_method} outlier removal method...")
    
    # Ask about enhanced features
    enhanced_choice = input("\nUse enhanced features? (y/n, default=n): ").strip().lower()
    use_enhanced = enhanced_choice == 'y'
    
    # Run complete pipeline
    classifier.run_complete_pipeline(analyze_data=True, outlier_method=outlier_method, 
                                   use_enhanced_features=use_enhanced)
    
    # Optional: Show learning curves for best model
    if classifier.best_model and input("\nShow learning curve for best model? (y/n): ").lower() == 'y':
        best_model = classifier.models[classifier.best_model]
        if 'Enhanced' in classifier.best_model:
            X, y = classifier.X_enh_train, classifier.y_enh_train
        else:
            X, y = classifier.X_train, classifier.y_train
        
        classifier.evaluator.plot_learning_curve(
            best_model, X, y, f"Learning Curve - {classifier.best_model}"
        )


if __name__ == "__main__":
    # Option to test outlier analysis only
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test_outliers":
        quick_outlier_analysis()
    elif len(sys.argv) > 1 and sys.argv[1] == "test_conservative":
        test_conservative_method()
    else:
        main()
