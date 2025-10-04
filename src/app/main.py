"""
FastAPI Application with Gradio UI

This module provides a production-ready REST API for churn prediction
along with an interactive web UI built with Gradio.

Endpoints:
    - GET  /           : Health check
    - POST /predict    : Predict churn for a customer
    - GET  /ui         : Gradio web interface

Usage:
    uvicorn src.app.main:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path
from typing import Optional
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import gradio as gr

from src.serving.inference import ChurnPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE VALIDATION
# ============================================================================

class CustomerData(BaseModel):
    """
    Pydantic model for customer data validation.
    Ensures all inputs are valid before making predictions.
    """
    gender: str = Field(..., description="Customer gender: Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen: 0 or 1")
    Partner: str = Field(..., description="Has partner: Yes or No")
    Dependents: str = Field(..., description="Has dependents: Yes or No")
    tenure: int = Field(..., ge=0, le=100, description="Months with company (0-100)")
    PhoneService: str = Field(..., description="Has phone service: Yes or No")
    MultipleLines: str = Field(..., description="Multiple lines: Yes, No, or No phone service")
    InternetService: str = Field(..., description="Internet service: DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Online security: Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Online backup: Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Device protection: Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Tech support: Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Streaming TV: Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Streaming movies: Yes, No, or No internet service")
    Contract: str = Field(..., description="Contract: Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Paperless billing: Yes or No")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly charges in dollars")
    TotalCharges: float = Field(..., ge=0, description="Total charges in dollars")

    @validator('gender')
    def validate_gender(cls, v):
        """Validate gender field."""
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be Male or Female')
        return v

    @validator('Partner', 'Dependents', 'PhoneService', 'PaperlessBilling')
    def validate_yes_no(cls, v):
        """Validate Yes/No fields."""
        if v not in ['Yes', 'No']:
            raise ValueError('Must be Yes or No')
        return v

    @validator('Contract')
    def validate_contract(cls, v):
        """Validate contract field."""
        valid_contracts = ['Month-to-month', 'One year', 'Two year']
        if v not in valid_contracts:
            raise ValueError(f'Contract must be one of {valid_contracts}')
        return v

    class Config:
        # Example for API documentation
        schema_extra = {
            "example": {
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
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: str = Field(..., description="Churn prediction")
    customer_data: dict = Field(..., description="Input customer data")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production ML API for predicting customer churn",
    version="1.0.0"
)

# Initialize predictor (will be loaded on startup)
predictor: Optional[ChurnPredictor] = None


@app.on_event("startup")
async def load_model():
    """
    Load ML model on application startup.
    This ensures the model is ready before serving requests.
    """
    global predictor
    try:
        logger.info("Loading model...")
        predictor = ChurnPredictor(
            model_path="artifacts/model",
            feature_cols_path="artifacts/feature_columns.json"
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Continue running but predictions will fail gracefully


@app.get("/")
def health_check():
    """
    Health check endpoint.
    Returns API status and model availability.
    """
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Predict churn for a single customer.

    Args:
        customer (CustomerData): Customer attributes

    Returns:
        PredictionResponse: Churn prediction and input data

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Convert Pydantic model to dict
        customer_dict = customer.dict()

        # Make prediction
        prediction = predictor.predict(customer_dict)

        logger.info(f"Prediction: {prediction}")

        return PredictionResponse(
            prediction=prediction,
            customer_data=customer_dict
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================================
# GRADIO WEB UI
# ============================================================================

def create_gradio_interface():
    """
    Create Gradio web interface for interactive predictions.

    Returns:
        gr.Blocks: Gradio interface
    """
    def predict_churn_gradio(gender, senior_citizen, partner, dependents, tenure,
                             phone_service, multiple_lines, internet_service,
                             online_security, online_backup, device_protection,
                             tech_support, streaming_tv, streaming_movies,
                             contract, paperless_billing, payment_method,
                             monthly_charges, total_charges):
        """
        Gradio prediction function.
        Takes individual inputs and returns prediction.
        """
        try:
            # Create customer dict
            customer = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': int(tenure),
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': float(monthly_charges),
                'TotalCharges': float(total_charges)
            }

            # Make prediction
            if predictor is not None:
                prediction = predictor.predict(customer)
                return f"🔮 **Prediction:** {prediction}"
            else:
                return "❌ Model not loaded"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    # Create Gradio interface
    with gr.Blocks(title="Customer Churn Prediction") as interface:
        gr.Markdown("# 🎯 Customer Churn Prediction")
        gr.Markdown("Enter customer information to predict churn likelihood")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 👤 Customer Demographics")
                gender = gr.Radio(["Male", "Female"], label="Gender", value="Female")
                senior_citizen = gr.Radio([0, 1], label="Senior Citizen", value=0)
                partner = gr.Radio(["Yes", "No"], label="Has Partner", value="Yes")
                dependents = gr.Radio(["Yes", "No"], label="Has Dependents", value="No")
                tenure = gr.Slider(0, 100, value=12, label="Tenure (months)")

            with gr.Column():
                gr.Markdown("### 📱 Services")
                phone_service = gr.Radio(["Yes", "No"], label="Phone Service", value="Yes")
                multiple_lines = gr.Radio(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")
                internet_service = gr.Radio(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
                online_security = gr.Radio(["Yes", "No", "No internet service"], label="Online Security", value="No")
                online_backup = gr.Radio(["Yes", "No", "No internet service"], label="Online Backup", value="No")

        with gr.Row():
            with gr.Column():
                device_protection = gr.Radio(["Yes", "No", "No internet service"], label="Device Protection", value="No")
                tech_support = gr.Radio(["Yes", "No", "No internet service"], label="Tech Support", value="No")
                streaming_tv = gr.Radio(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes")
                streaming_movies = gr.Radio(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes")

            with gr.Column():
                gr.Markdown("### 💳 Billing")
                contract = gr.Radio(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
                paperless_billing = gr.Radio(["Yes", "No"], label="Paperless Billing", value="Yes")
                payment_method = gr.Dropdown(
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    label="Payment Method",
                    value="Electronic check"
                )
                monthly_charges = gr.Number(label="Monthly Charges ($)", value=85.0)
                total_charges = gr.Number(label="Total Charges ($)", value=1020.0)

        predict_btn = gr.Button("🔮 Predict Churn", variant="primary")
        output = gr.Markdown()

        predict_btn.click(
            fn=predict_churn_gradio,
            inputs=[gender, senior_citizen, partner, dependents, tenure,
                    phone_service, multiple_lines, internet_service,
                    online_security, online_backup, device_protection,
                    tech_support, streaming_tv, streaming_movies,
                    contract, paperless_billing, payment_method,
                    monthly_charges, total_charges],
            outputs=output
        )

    return interface


# Mount Gradio interface to FastAPI
gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
