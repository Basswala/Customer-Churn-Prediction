# 🎯 Customer Churn Prediction - Production ML System

A production-ready machine learning system for predicting customer churn in the telecommunications industry. This project demonstrates end-to-end MLOps practices including data validation, model comparison, experiment tracking, REST API deployment, and cloud deployment.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Deployment](#api-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Project Structure](#project-structure)

## ✨ Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, and LightGBM
- **MLflow Tracking**: Complete experiment tracking and model versioning
- **Data Validation**: Great Expectations for data quality checks
- **REST API**: FastAPI for production-grade predictions
- **Web UI**: Interactive Gradio interface
- **Docker Support**: Containerized deployment
- **CI/CD Pipeline**: GitHub Actions for automated deployment
- **Cloud Ready**: AWS ECS, GCP Cloud Run compatible

## 🏗️ Architecture

```
┌─────────────────┐
│   Raw Data      │
│  (CSV File)     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Data Validation │  ← Great Expectations
│  & Cleaning     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│    Feature      │  ← Binary encoding, one-hot encoding
│  Engineering    │     Engineered features
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Model Training │  ← LogReg, RF, XGBoost, LightGBM
│  & Comparison   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ MLflow Logging  │  ← Experiments, metrics, models
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Best Model     │
│   Selection     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  FastAPI +      │  ← REST API + Web UI
│    Gradio       │
└─────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- Git

### Local Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train Models

Run the complete training pipeline:

```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
```

This will:
- Load and validate data
- Engineer features
- Train 4 different models
- Compare performance
- Log everything to MLflow
- Save the best model

### 2. View MLflow Experiments

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Open http://localhost:5000 to view:
- Model comparisons
- Metrics (precision, recall, F1, ROC-AUC)
- Parameters
- Artifacts

### 3. Run API Server

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

Access:
- **API Docs**: http://localhost:8000/docs
- **Web UI**: http://localhost:8000/ui
- **Health Check**: http://localhost:8000/

### 4. Make Predictions

#### Using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 1020.0
  }'
```

#### Using Python:

```python
import requests

customer_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    # ... rest of the fields
}

response = requests.post("http://localhost:8000/predict", json=customer_data)
print(response.json())
```

## 🎓 Model Training

### Algorithms Compared

1. **Logistic Regression**: Fast, interpretable baseline
2. **Random Forest**: Ensemble method with good generalization
3. **XGBoost**: Gradient boosting with excellent performance
4. **LightGBM**: Fast, memory-efficient gradient boosting

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Avoid false alarms
- **Recall**: Catch all churners
- **F1 Score**: Balance between precision and recall
- **ROC-AUC**: Overall discriminative power

### Feature Engineering

- Binary encoding for Yes/No fields
- One-hot encoding for categorical features
- Tenure bins (customer lifecycle)
- Charges categories (spending tiers)
- Average monthly charge per tenure

## 🌐 API Deployment

### Local Docker Deployment

```bash
# Build Docker image
docker build -t churn-prediction .

# Run container
docker run -p 8000:8000 churn-prediction
```

### Production Deployment

#### AWS ECS (Fargate)

1. Push image to ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag churn-prediction:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction:latest
```

2. Update ECS service:
```bash
aws ecs update-service --cluster <cluster-name> --service <service-name> --force-new-deployment
```

#### GCP Cloud Run

```bash
gcloud run deploy churn-prediction-api \
  --image gcr.io/<project-id>/churn-prediction \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 📁 Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/                    # Raw CSV data
│   └── processed/              # Cleaned data
├── src/
│   ├── data/                   # Data loading & preprocessing
│   │   ├── load_data.py        # CSV loading & validation
│   │   └── preprocess.py       # Data cleaning
│   ├── features/               # Feature engineering
│   │   └── build_features.py   # Transform features
│   ├── models/                 # Model training
│   │   ├── train_model.py      # Train & compare models
│   │   └── mlflow_utils.py     # MLflow integration
│   ├── utils/                  # Utilities
│   │   └── validate_data.py    # Great Expectations
│   ├── serving/                # Inference
│   │   └── inference.py        # Prediction logic
│   └── app/                    # API
│       └── main.py             # FastAPI + Gradio
├── scripts/
│   └── run_pipeline.py         # Complete training pipeline
├── mlruns/                     # MLflow experiments
├── artifacts/                  # Model artifacts
├── .github/workflows/          # CI/CD
│   └── deploy.yml              # GitHub Actions
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file for sensitive configuration:

```bash
# AWS
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1

# Docker Hub
DOCKERHUB_USERNAME=your_username
DOCKERHUB_TOKEN=your_token

# MLflow
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=Telco_Churn_Prediction
```

## 📊 Performance

Typical model performance on the Telco dataset:

| Model              | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.80     | 0.65      | 0.55   | 0.60     | 0.84    |
| Random Forest      | 0.79     | 0.64      | 0.50   | 0.56     | 0.82    |
| **XGBoost**        | **0.82** | **0.68**  | **0.58**| **0.63**| **0.86**|
| LightGBM           | 0.81     | 0.67      | 0.56   | 0.61     | 0.85    |

*Note: Performance may vary based on data splits and hyperparameters*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset: IBM Telco Customer Churn
- MLflow for experiment tracking
- FastAPI for the web framework
- Gradio for the interactive UI
