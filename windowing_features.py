"""
Windowing and feature extraction utilities for EMG signal processing.
These functions are used in both training and inference to ensure consistency.
"""

import numpy as np
import pandas as pd


# ============================================ Configuration ============================================
WINDOW_SIZE = 10  # Number of samples per window (e.g., 10 samples at 50Hz = 200ms)
WINDOW_STRIDE = 5  # Overlap: stride of 5 means 50% overlap
EMG_CHANNELS = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8']


# ======================================= Feature Extraction ========================================

def extract_features_from_window(window_data):
    """
    Extract Hudgins time domain features from a single window of EMG data.
    
    Hudgins feature set (1993):
    - MAV: Mean Absolute Value
    - WL: Waveform Length
    - ZC: Zero Crossings
    - SSC: Slope Sign Changes
    
    Parameters:
    -----------
    window_data : numpy array or DataFrame
        Shape: (window_size, n_channels) - e.g., (10, 8)
    
    Returns:
    --------
    dict : Dictionary of extracted features (4 features × 8 channels = 32 total)
    """
    if isinstance(window_data, pd.DataFrame):
        window_data = window_data.values
    
    features = {}
    n_channels = window_data.shape[1]
    
    for ch in range(n_channels):
        channel_data = window_data[:, ch]
        ch_name = f'EMG{ch+1}'
        
        # Mean Absolute Value (MAV)
        features[f'{ch_name}_MAV'] = np.mean(np.abs(channel_data))
        
        # Waveform Length (WL) - sum of absolute differences
        features[f'{ch_name}_WL'] = np.sum(np.abs(np.diff(channel_data)))
        
        # Zero Crossings (ZC)
        features[f'{ch_name}_ZC'] = np.sum(np.diff(np.sign(channel_data)) != 0)
        
        # Slope Sign Changes (SSC)
        if len(channel_data) >= 3:
            ssc = 0
            for i in range(1, len(channel_data) - 1):
                if (channel_data[i] - channel_data[i-1]) * (channel_data[i] - channel_data[i+1]) > 0:
                    ssc += 1
            features[f'{ch_name}_SSC'] = ssc
        else:
            features[f'{ch_name}_SSC'] = 0
    
    return features


def get_feature_names():
    """
    Get the ordered list of Hudgins feature names that will be extracted.
    Important for maintaining consistent feature order in training and inference.
    
    Hudgins features: MAV, WL, ZC, SSC (4 features × 8 channels = 32 total)
    """
    # Generate feature names in consistent order
    feature_names = []
    feature_types = ['MAV', 'WL', 'ZC', 'SSC']  # Hudgins time domain features
    
    for ch in range(1, 9):  # EMG1 to EMG8
        for ft in feature_types:
            feature_names.append(f'EMG{ch}_{ft}')  
    
    return feature_names
# ======================================= Windowing for Training ========================================

def create_windows_from_dataframe(df, gesture_col='CurrentGestures'):
    """
    Create sliding windows from the training data CSV.
    
    Parameters:
    -----------
    df : DataFrame
        Training data with EMG columns and gesture labels
    gesture_col : str
        Name of the column containing gesture labels
    
    Returns:
    --------
    DataFrame : Windowed data with extracted features
    """
    windowed_data = []
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_values('Timestamp')
    emg_data = df[EMG_CHANNELS].values
    gestures = df[gesture_col].values
    thresholds = df['Threshold'].values if 'Threshold' in df.columns else None
    
    # Create sliding windows
    for i in range(0, len(emg_data) - WINDOW_SIZE + 1, WINDOW_STRIDE):
        window = emg_data[i:i + WINDOW_SIZE]
        
        # Get gestures and thresholds in this window
        window_gestures = gestures[i:i + WINDOW_SIZE]
        gesture_label = pd.Series(window_gestures).mode()[0]  # Majority vote
        
        # CRITICAL: Ensure threshold consistency across entire window
        # Skip windows where EMG intensity changes from above to below threshold (or vice versa)
        # This ensures all samples in the window are at similar activation levels
        if thresholds is not None:
            window_thresholds = thresholds[i:i + WINDOW_SIZE]
            unique_thresholds = pd.Series(window_thresholds).unique()
            
            if len(unique_thresholds) > 1:
                # This window spans a threshold boundary - skip it
                continue
            
            window_threshold = unique_thresholds[0]  # All same, so pick first
        else:
            window_threshold = None
        
        # Extract features
        features = extract_features_from_window(window)
        features['gesture'] = gesture_label
        if window_threshold is not None:
            features['threshold'] = window_threshold
        
        windowed_data.append(features)
    
    return pd.DataFrame(windowed_data)


# ======================================= Buffering for Inference ========================================

class EMGBuffer:
    """
    Maintains a rolling buffer of EMG samples for real-time feature extraction.
    Use one instance per client/session in the server.
    """
    
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.buffer = []  # List of samples, each sample is a dict of {EMG1: val, EMG2: val, ...}
    
    def add_sample(self, sample):
        """
        Add a new EMG sample to the buffer.
        
        Parameters:
        -----------
        sample : dict
            Dictionary with keys EMG1-EMG8 and their values
        """
        self.buffer.append(sample)
        
        # Keep only the last window_size samples
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
    
    def is_ready(self):
        """Check if buffer has enough samples to extract features."""
        return len(self.buffer) >= self.window_size
    
    def extract_features(self):
        """
        Extract features from the current buffer.
        
        Returns:
        --------
        dict : Extracted features, or None if buffer not ready
        """
        if not self.is_ready():
            return None
        
        # Convert buffer to numpy array
        window_data = np.array([[sample[ch] for ch in EMG_CHANNELS] for sample in self.buffer])
        
        return extract_features_from_window(window_data)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []


# ======================================== Testing/Debugging ========================================

if __name__ == "__main__":
    # Test feature extraction
    print("Testing feature extraction...")
    
    # Create dummy window data (10 samples, 8 channels)
    test_window = np.random.randn(WINDOW_SIZE, 8) * 10
    
    features = extract_features_from_window(test_window)
    print(f"\nExtracted {len(features)} features:")
    for i, (name, value) in enumerate(list(features.items())[:6]):
        print(f"  {name}: {value:.3f}")
    print("  ...")
    
    print(f"\nTotal feature names: {len(get_feature_names())}")
    print(f"Feature names: {get_feature_names()[:6]}...")
    
    # Test buffer
    print("\n\nTesting EMGBuffer...")
    buffer = EMGBuffer(window_size=5)
    
    for i in range(7):
        sample = {f'EMG{ch}': np.random.randn() * 10 for ch in range(1, 9)}
        buffer.add_sample(sample)
        print(f"Sample {i+1}: Buffer size = {len(buffer.buffer)}, Ready = {buffer.is_ready()}")
    
    features = buffer.extract_features()
    print(f"\nExtracted features from buffer: {list(features.keys())[:6]}...")
