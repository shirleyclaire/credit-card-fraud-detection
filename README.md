# Credit Card Fraud Detection System

## Project Description
An advanced credit card fraud detection system that leverages machine learning to identify fraudulent transactions in real-time. The system uses Isolation Forest as the primary anomaly detection algorithm, enhanced with feature engineering and ensemble methods to handle severe class imbalance in credit card transaction data.

## Key Features
- **Real-time Fraud Detection**: Low-latency prediction pipeline for instant transaction scoring
- **Advanced Feature Engineering**: 
  - Time-based patterns
  - Transaction velocity monitoring
  - PCA-based anomaly features
  - Card-level behavioral analysis
- **Comprehensive Monitoring**:
  - Population drift detection
  - Performance degradation alerts
  - Real-time metric tracking
- **Investigation Support**:
  - Detailed case reports
  - Risk factor analysis
  - Similar case retrieval
  - Investigation queue management

## Technologies Used
- Python 3.8+
- scikit-learn
- pandas
- numpy
- FastAPI
- plotly/dash
- Redis (for caching)
- prometheus-client (for metrics)

## Installation & Setup

### Prerequisites
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

### Configuration
Create a `.env` file with necessary configurations:
```env
MODEL_PATH=models/fraud_detector.joblib
PREPROCESSOR_PATH=models/preprocessor.joblib
LOG_LEVEL=INFO
```

## Quick Start Guide

1. Install dependencies:
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

2. Train and evaluate the model:
```bash
python src/main.py
```
This will:
- Load and preprocess the data
- Train the Isolation Forest model
- Perform model evaluation
- Generate visualization insights
- Save the trained model

3. Start the API server:
```bash
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000
```
The API will be available at http://localhost:8000

4. Access the API documentation:
- OpenAPI docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Fraud Detection
```bash
POST /predict
```
Example payload:
```json
{
    "card_id": "card_123",
    "amount": 150.00,
    "merchant": "merchant_456",
    "timestamp": 1634567890,
    "additional_features": {}
}
```

### Investigation
```bash
GET /investigation/case/{case_id}
GET /investigation/queue
POST /investigation/feedback/{case_id}
```

### Monitoring
```bash
GET /health
```

## Usage Instructions

### Running the API
```bash
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000
```

### Model Training
```bash
python src/main.py
```

## Model Performance

### Metrics
- AUPRC (Area Under Precision-Recall Curve)
- Precision at fixed recall levels
- Real-time latency statistics

### Feature Importance
Key predictive features include:
- Transaction amount deviation
- Time-based patterns
- Merchant frequency
- PCA component interactions

## Monitoring & Maintenance

### Real-time Monitoring
- Transaction volume
- Model performance metrics
- Drift detection
- System health

### Model Updates
- Automated retraining triggers
- Performance degradation detection
- Feedback loop integration

## Future Improvements
1. Enhanced Feature Engineering
   - Geographic pattern analysis
   - Network-based features
   - Deep learning integration

2. System Enhancements
   - Multi-model ensemble voting
   - A/B testing framework
   - Advanced drift detection

3. Operational Improvements
   - Kubernetes deployment
   - Enhanced monitoring
   - Automated model retraining

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

