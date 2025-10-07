# PM2.5 Prediction Service

A FastAPI-based machine learning service that predicts PM2.5 (particulate matter) air quality levels up to 6 hours ahead using historical data.

## Overview

This service uses a trained machine learning model to forecast PM2.5 concentrations based on historical hourly measurements. It provides a REST API for integration with air quality monitoring systems.

## Features

- 6-hour ahead PM2.5 predictions
- RESTful API with FastAPI
- CORS support for frontend integration
- Dockerized deployment
- Health check endpoint
- Interactive API documentation (Swagger UI)

## Requirements

- Python 3.12.10
- Docker (for containerized deployment)

## Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/AstroDreamers/tempo-ml-lab.git
cd tempo-ml-lab
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the service:
```bash
python prediction_service.py
```

The service will be available at `http://localhost:5000`

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t tempo-ml-lab .
```

2. Run the container:
```bash
docker run -p 5000:5000 tempo-ml-lab
```

Or with environment variables:
```bash
docker run -p 5000:5000 \
  -e PREDICTION_CORS_ORIGINS="https://your-frontend.com" \
  tempo-ml-lab
```

## API Endpoints

### Root
- **GET** `/`
- Returns service information

### Health Check
- **GET** `/health`
- Returns service health status and model info

### Predict PM2.5
- **POST** `/predict`
- Predicts PM2.5 levels for the next 6 hours

**Request Body:**
```json
{
  "historical_data": [
    {
      "datetime": "2025-10-05T00:00:00",
      "pm25": 35.2
    },
    {
      "datetime": "2025-10-05T01:00:00",
      "pm25": 32.8
    }
    // ... minimum 48 hours of data required
  ]
}
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "datetime": "2025-10-07T01:00:00",
      "predicted_pm25": 42.5,
      "forecast_hour": 1
    }
    // ... 6 predictions total
  ],
  "forecast_start": "2025-10-07T00:00:00",
  "forecast_hours": 6
}
```

## API Documentation

Once the service is running, visit:
- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `5000` |
| `PREDICTION_CORS_ORIGINS` | Comma-separated list of allowed CORS origins | `https://tempo-backend-rzn2.onrender.com,http://localhost:8080` |
| `RELOAD` | Enable auto-reload in development | `false` |

### Example CORS Configuration

```bash
# Single origin
PREDICTION_CORS_ORIGINS=https://your-frontend.com

# Multiple origins
PREDICTION_CORS_ORIGINS=https://your-frontend.com,https://api.example.com

# Allow all (not recommended for production)
PREDICTION_CORS_ORIGINS=*
```

## Model Files

The service requires these model files in the root directory:
- `pm25_model.pkl` - Trained prediction model
- `feature_cols.pkl` - Feature column names

## Feature Engineering

The model uses the following features:
- **Temporal features**: hour, day of week, month, weekend indicator
- **Lag features**: PM2.5 values from 1, 2, 3, 6, 12, and 24 hours ago
- **Rolling statistics**: Mean and standard deviation over 3, 6, 12, and 24-hour windows

Minimum historical data requirement: **24 hours** (48+ hours recommended for better accuracy)

## Deployment on Render

1. Push your code to GitHub
2. Create a new Web Service on Render
3. Select **Docker** as the environment
4. Set environment variables in Render dashboard
5. Deploy!

### Render Configuration

- **Environment**: Docker
- **Dockerfile Path**: `Dockerfile`
- **Docker Context**: `./`

## Project Structure

```
tempo-ml-lab/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── prediction_service.py    # Main FastAPI application
├── pm25_model.pkl           # Trained ML model
├── feature_cols.pkl         # Feature columns
└── README.md
```

## Dependencies

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pandas==2.1.4
numpy
scikit-learn
pickle (built-in)
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200` - Success
- `400` - Bad request (insufficient data, invalid format)
- `500` - Internal server error

## Development

### Running with Auto-reload

```bash
export RELOAD=true
python prediction_service.py
```

### Testing the API

Using curl:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_data.json
```

Using Python:
```python
import requests

data = {
    "historical_data": [
        {"datetime": "2025-10-05T00:00:00", "pm25": 35.2},
        # ... more data points
    ]
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

Copyright (c) 2025 AstroDreamers

## Contact

For questions or support, please contact +61450306460

## Acknowledgments

- Built with FastAPI
- Deployed on Render
- Part of the Tempo Air Quality Monitoring System
