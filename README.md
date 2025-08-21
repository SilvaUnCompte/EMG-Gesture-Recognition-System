# EMG Gesture Recognition System

Machine learning system for recognizing hand gestures using Myo Armband (EMG) sensor data.
This system analyzes EMG signals from 8 sensors to classify different hand gestures.
It uses neural networks (Multi-Layer Perceptron) and exposes recognition through a web API.

## Supported Gestures
Currently, the project was tested on a limited number of gestures:
- **rest** - Relaxed arm position
- **fist** - Closed fist gesture
- **pinch** - Pinching gesture
- **wrist-back** - Wrist flexion backward
- **wrist-forward** - Wrist flexion forward

## Project Structure

```
├── models/                    # Trained model artifacts (Not in git)
│   ├── gesture_recognition_old
│   └── gesture_recognition_1.2.7_[timestamp]/
│          ├── pipeline.joblib
│          ├── config.json
│          └── metrics.json
└── notebooks/                 # Jupyter notebooks for experimentation
    ├── train_ai.ipynb
    ├── train_model_v1         # Training script with hyperparameter optimization
    └── ...
├── cli_model_interface.py     # CLI prediction tool
├── data.csv                   # Training dataset with EMG readings
├── server.py                  # Python server implementation
├── train_model.py             # Main training script
```

## Technologies Used

- **Python 3.11+**
- **Machine Learning**: scikit-learn, pandas, numpy
- **Web API**: FastAPI, uvicorn
- **Visualization**: matplotlib
- **Model Persistence**: joblib

## Data Format

The EMG data is stored in CSV format with semicolon separators:

```csv
gesture;EMG1;EMG2;EMG3;EMG4;EMG5;EMG6;EMG7;EMG8;SessionID;Timestamp
fist;-7;5;-6;-8;-4;-1;10;16;2;7ff2a046-ff05-452b-88f2-7538daf97f48;2025-08-20 09:50:33.3417
wrist-back;-2;48;-12;-22;-1;-1;-10;0;7ff2a046-ff05-452b-88f2-7538daf97f48;2025-08-20 09:55:20.5966
pinch;-2;4;-22;-6;0;0;-2;-3;7ff2a046-ff05-452b-88f2-7538daf97f48;2025-08-20 09:51:40.0482
```

### Features
- **gesture**: Target gesture label
- **EMG1-EMG8**: Raw EMG sensor readings (8 channels)
- **Additional metadata (Optional)**: SessionID, Timestamp

## Quick Start

### 1. Training a Model

```bash
python train_model.py data.csv
```
_Specify data path, or use the default root file 'data.csv'_'

This will:
- Load and preprocess the data
- Train an MLP neural network
- Evaluate performance with cross-validation
- Export the trained model with timestamp

### 2. Starting the API Server

```bash
python -m uvicorn use_server:app --reload --host 0.0.0.0 --port 8000
```
_By default, the server uses the last model created in the 'models' directory._

### 3. Making Predictions

#### Via REST API:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "features": {
    "EMG1": 51, "EMG2": 8, "EMG3": 2, "EMG4": -3,
    "EMG5": -25, "EMG6": 12, "EMG7": -7, "EMG8": -26
  }
}'
```

#### Via Command Line: (doesn't need server)
```bash
python cli_model_interface.py --model_dir models/gesture_recognition_1.2.7_xxxxxxxxx --features 51 8 2 -3 -25 12 -7 -26
```
*If no 'model_dir' is provided, uses the latest in 'models' directory.*

## API Documentation

### Endpoint: `POST /predict`

**Request Body:**
```json
{
  "features": {
    "EMG1": float,
    "EMG2": float,
    "EMG3": float,
    "EMG4": float,
    "EMG5": float,
    "EMG6": float,
    "EMG7": float,
    "EMG8": float
  }
}
```

**Response:**
```json
{
  "label": "wrist-back",
  "prob": 0.95,
  "topk": [
    {"label": "wrist-back", "prob": 0.95},
    {"label": "pinch", "prob": 0.03}
  ]
}
```

**Features:**
- **Confidence threshold**: Returns "unknown" if confidence < 0.6
- **Top-K predictions**: Returns top 2 most likely gestures
- **Input validation**: Ensures all 8 EMG features are provided

## Basic Training Pipeline (train_model.py)
The [train_model.py](train_model.py) script implements a training pipeline with fixed hyperparameters. For hyperparameter optimization, use [train_model_v1.ipynb](notebooks/train_model_v1.ipynb).

### Training Workflow

1. **Data Loading**: Import CSV with semicolon separator
2. **Quality Control**: Remove duplicates and missing values
3. **Preprocessing**: Standardize EMG features, encode labels
4. **Model Training**: Train MLP with early stopping
5. **Evaluation**: Comprehensive performance analysis
6. **Export**: Save complete pipeline with metadata

### Model Export & Versioning

#### Exported Artifacts:
1. **`pipeline.joblib`**: Complete sklearn pipeline (scaler + model)
2. **`config.json`**: Model configuration and metadata
3. **`metrics.json`**: Training performance metrics

#### Configuration Schema:
```json
{
  "feature_names": ["EMG1", "EMG2", ..., "EMG8"],
  "class_names": ["fist", "pinch", "rest", ...],
  "top_k": 2,
  "abstain_threshold": 0.6,
  "random_state": 42,
  "library": {"sklearn": ">=1.2"}
}
```

### Hyperparameter Optimization
Don't forget to adapt hyperparameters and experiment with different architectures for best results. You can use the `Hyperparameter optimization` section in [train_model_v1.ipynb](notebooks/train_model_v1.ipynb).

Example:
```python
model = create_mlp_model(
    hidden_layer_sizes=(128, 64, 32),  # Multi-layer architecture
    activation='tanh',                 # Alternative activation
    alpha=0.001,                      # Different regularization
    learning_rate_init=0.01           # Higher learning rate
)
```

### Visual Outputs

During training, the script generates:
- **Confusion Matrix Heatmap**: Classification accuracy visualization
- **Multi-class ROC Curves**: Performance curves for each gesture class
- **Console Output**: Detailed classification report and cross-validation scores

## Installation

1. **Clone the repository**
2. **Install dependencies:**
```bash
pip install pandas scikit-learn fastapi uvicorn matplotlib seaborn joblib pydantic numpy
```

3. **Ensure you have a data file**

## Configuration

Models are exported with configuration files containing:
- **Feature names**: Expected input order
- **Class names**: Gesture labels
- **Thresholds**: Confidence and top-k settings
- **Metadata**: Library versions, random state

## Model Versioning

Models are automatically versioned with timestamps:
- Format: `models/gesture_cls/VERSION_YYYYMMDDTHHMMSSZ`
- Example: `models/gesture_cls/1.2.7_20250821T092331Z`

## Notes
- **Data quality**: Ensure EMG sensors are properly calibrated
- **Real-time performance**: API response time typically < 100ms
- **Model updates**: Retrain periodically with new data for best performance
- **Hardware compatibility**: Works with any 8-channel EMG acquisition system (e.g. Myo Armband)

_Feel free to use my work._