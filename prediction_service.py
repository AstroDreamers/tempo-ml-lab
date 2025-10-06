from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List

# Optional: load .env in development if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
from pydantic import BaseModel, Field
from typing import List
import pickle
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI(title="PM2.5 Prediction Service")

# CORS configuration
# Read allowed origins from environment variable PREDICTION_CORS_ORIGINS (comma-separated)
# Defaults to allow only the production frontend and localhost:8080 for local dev
DEFAULT_ORIGINS = [
    "https://tempo-backend-rzn2.onrender.com",
    "http://localhost:8080",
]
raw_origins = os.environ.get('PREDICTION_CORS_ORIGINS')
if raw_origins:
    # allow commas or semicolons as separators
    origins = [o.strip() for o in raw_origins.replace(';', ',').split(',') if o.strip()]
else:
    origins = DEFAULT_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
print("Loading model...")
model = pickle.load(open('pm25_model.pkl', 'rb'))
feature_cols = pickle.load(open('feature_cols.pkl', 'rb'))
print(f"Model loaded with {len(feature_cols)} features")

# Request/Response models
class HistoricalDataPoint(BaseModel):
    datetime: str = Field(..., description="ISO datetime string")
    pm25: float = Field(..., description="PM2.5 value in µg/m³")
    
    class Config:
        extra = "ignore"

class PredictionRequest(BaseModel):
    historical_data: List[HistoricalDataPoint]

class PredictionPoint(BaseModel):
    datetime: str
    predicted_pm25: float
    forecast_hour: int

class PredictionResponse(BaseModel):
    success: bool
    predictions: List[PredictionPoint]
    forecast_start: str
    forecast_hours: int

@app.get("/")
def root():
    return {
        "message": "PM2.5 Prediction Service", 
        "status": "running",
        "forecast_hours": "6 hours ahead"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "model_loaded": True,
        "features": len(feature_cols)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert to list of dicts
        data = [item.model_dump() for item in request.historical_data]
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Validate minimum data
        if len(df) < 24:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 24 hours of data, got {len(df)} hours"
            )
        
        # Create features
        df_featured = create_features_for_prediction(df)
        
        if len(df_featured) == 0:
            raise HTTPException(
                status_code=400,
                detail="Not enough data after feature engineering. Need 48+ hours of historical data."
            )
        
        # Get latest data point
        X_latest = df_featured.iloc[[-1]][feature_cols]
        
        # Make prediction
        forecast = model.predict(X_latest)[0]
        
        # Get number of forecast hours from model output
        num_forecast_hours = len(forecast)
        
        # Create response
        last_time = df['datetime'].iloc[-1]
        predictions = []
        
        for h in range(1, num_forecast_hours + 1):
            future_time = last_time + pd.Timedelta(hours=h)
            predictions.append(PredictionPoint(
                datetime=future_time.isoformat(),
                predicted_pm25=float(forecast[h-1]),
                forecast_hour=h
            ))
        
        return PredictionResponse(
            success=True,
            predictions=predictions,
            forecast_start=last_time.isoformat(),
            forecast_hours=num_forecast_hours
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def create_features_for_prediction(df):
    """Create same features as training pipeline"""
    df = df.copy()
    
    # Temporal features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'pm25_lag_{lag}h'] = df['pm25'].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12, 24]:
        df[f'pm25_rolling_mean_{window}h'] = df['pm25'].shift(1).rolling(window).mean()
        df[f'pm25_rolling_std_{window}h'] = df['pm25'].shift(1).rolling(window).std()
    
    # Drop rows with NaN
    df = df.dropna()
    
    return df

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    reload_flag = os.environ.get('RELOAD', 'false').lower() in ('1', 'true', 'yes')
    uvicorn.run(app, host='0.0.0.0', port=port, reload=reload_flag)