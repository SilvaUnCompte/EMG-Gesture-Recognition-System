# EMG Gesture Recognition System

Machine learning system for recognizing hand gestures using Myo Armband (EMG) sensor data.
This system analyzes EMG signals from 8 sensors to classify different hand gestures.
It uses machine learning models (Multi-Layer Perceptron and Random Forest) and exposes recognition through a web API.

## Supported Gestures
Currently, the project was tested on a limited number of gestures:
- **rest** - Relaxed arm position
- **fist** - Closed fist gesture
- **pinch** - Pinching gesture
- **wrist-back** - Wrist flexion backward
- **wrist-forward** - Wrist flexion forward

## Data Format

The EMG data is stored in CSV format with semicolon separators:

```csv
CurrentGestures;EMG1;EMG2;EMG3;EMG4;EMG5;EMG6;EMG7;EMG8;Framecount;SessionID;Threshold;Timestamp
fist;-2;19;16;2;13;4;7;2;483;993b0c69-a43a-4942-b533-e6f782a5a939;above;2025-09-03 09:30:50.4475
wrist;-2;-5;-24;6;2;-5;-3;0;483;993b0c69-a43a-4942-b533-e6f782a5a939;above;2025-09-03 09:30:50.4485
pinch;-7;-8;3;2;-12;4;-6;-3;484;993b0c69-a43a-4942-b533-e6f782a5a939;below;2025-09-03 09:30:50.4624
```

### Features
- **CurrentGestures**: Target gesture label
- **EMG1-EMG8**: Raw EMG sensor readings (8 channels)
- **Threshold**: Used to only keep line with "above"
- **Additional metadata (Optional)**: SessionID, Timestamp

## Installation

1. **Clone the repository**
2. **Install dependencies:**
```bash
pip install pandas scikit-learn fastapi uvicorn matplotlib seaborn joblib pydantic numpy optuna
```
3. **Ensure you have a data file**

## Quick Start

### 1. Training a Model

```bash
python train_mlp_model.py data.csv
```

_Specify data path, or use the default root file 'data.csv'_.

This will:
- Load and preprocess the data
- Train the selected model (MLP or Random Forest)
- Evaluate performance with cross-validation
- Export the trained model with timestamp

_More information in the [Basic Training](#basic-training-pipeline) section._

### 2. Starting the API Server

```bash
python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
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
The [train_mlp_model.py](train_mlp_model.py) script implements a training pipeline (MLP) with fixed hyperparameters. For hyperparameter optimization, use [train_model_MLP.ipynb](notebooks/train_model_MLP.ipynb).

### Training Workflow

1. **Data Loading**: Import CSV with semicolon separator
2. **Quality Control**: Remove duplicates and missing values
3. **Preprocessing**: Standardize EMG features, encode labels
4. **Model Training**: Train selected model with cross-validation
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
Don't forget to adapt hyperparameters and experiment with different architectures for best results. You can use the `Hyperparameter optimization` section in [train_model_MLP.ipynb](notebooks/train_model_MLP.ipynb).

#### MLP Example:
```python
model = create_mlp_model(
    hidden_layer_sizes=(128, 64, 32),  # Multi-layer architecture
    activation='tanh',                 # Alternative activation
    alpha=0.001,                      # Different regularization
    learning_rate_init=0.01           # Higher learning rate
)
```

### Visual Outputs

During training, the scripts generate:
- **Confusion Matrix Heatmap**: Classification accuracy visualization
- **Multi-class ROC Curves**: Performance curves for each gesture class
- **Console Output**: Detailed classification report and cross-validation scores

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

## Project Structure

```
├── models/                    # Trained model artifacts (Not in git)
│   ├── gesture_recognition_old
│   └── gesture_recognition_1.2.7_[timestamp]/
│          ├── pipeline.joblib
│          ├── config.json
│          └── metrics.json
└── notebooks/                 # Jupyter notebooks for experimentation
    ├── train_model_RF
    ├── train_model_MLP        # Training script with hyperparameter optimization
    └── ...
├── cli_model_interface.py     # CLI prediction tool
├── data.csv                   # Training dataset with EMG readings
├── server.py                  # Python server implementation
├── train_mlp_model.py         # Training mlp model script
├── shared_functions.py        # Functions used in all training scripts
```

## Technologies Used

- **Python 3.11+**
- **Machine Learning**: scikit-learn, pandas, numpy
- **Web API**: FastAPI, uvicorn
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib
- **Hyperparameter Optimization**: optuna

## Notes
- **Data quality**: Ensure EMG sensors are properly calibrated
- **Real-time performance**: API response time typically < 100ms
- **Model updates**: Retrain periodically with new data for best performance
- **Hardware compatibility**: Works with any 8-channel EMG acquisition system (e.g. Myo Armband)


_Feel free to use my work._

