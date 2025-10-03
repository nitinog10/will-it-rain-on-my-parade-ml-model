# Extreme Weather Prediction System

A comprehensive machine learning system for predicting extreme weather conditions using NASA Earth observation data.

## Features

- **Multi-label Classification**: Predicts probability of 5 extreme weather conditions:
  - Very Hot
  - Very Cold
  - Very Windy
  - Very Wet
  - Very Uncomfortable (heat index)

- **NASA Data Integration**: Automatically fetches historical weather data from NASA POWER API
- **Advanced Feature Engineering**: Rolling averages, trends, historical comparisons
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Calibrated Predictions**: Ensures probabilities reflect true frequencies
- **REST API**: FastAPI backend for easy integration
- **Modern UI**: Interactive frontend with visualizations

## Installation

1. Install Python 3.8+
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Get NASA API key from https://api.nasa.gov/ (free)
4. Update `config.yaml` with your API key

## Usage

### 1. Collect Data
```bash
python src/data_collection.py --lat 40.7128 --lon -74.0060 --name "New_York"
```

### 2. Train Models
```bash
python src/train_models.py --data data/processed/New_York_processed.csv
```

### 3. Run API Server
```bash
python src/api.py
```

### 4. Open Frontend
Open `frontend/index.html` in your browser or serve it:
```bash
cd frontend
python -m http.server 8080
```

## Project Structure

```
Ml model Nasa/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── src/
│   ├── data_collection.py   # NASA data fetching
│   ├── feature_engineering.py
│   ├── train_models.py      # Model training pipeline
│   ├── evaluate.py          # Evaluation metrics
│   └── api.py               # FastAPI backend
├── models/                  # Trained models
├── data/
│   ├── raw/                 # Raw NASA data
│   └── processed/           # Processed features
└── frontend/
    ├── index.html           # Main UI
    ├── styles.css           # Styling
    └── app.js               # Frontend logic

## Model Performance

Models are evaluated using:
- ROC-AUC Score
- Precision-Recall AUC
- Brier Score (calibration)
- Log Loss

## API Endpoints

- `POST /predict`: Get predictions for a location and date
- `GET /health`: Health check
- `GET /model/info`: Model metadata

## Contributing

This is an educational project demonstrating ML best practices for weather prediction.

## License

MIT License

