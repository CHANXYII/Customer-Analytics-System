from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import ml_pipeline

app = FastAPI(title="Customer Analytics System API")

class CustomerData(BaseModel):
    features: List[float]
    known_interest: Optional[int] = -1

# Initialize models
print("Initializing models via in-memory training...")
_, models = ml_pipeline.run_pipeline()
encoder, scaler, ssl_model, kmeans = models

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/analyze")
def analyze_customer(data: CustomerData):
    """
    Process raw customer data through 4-step pipeline.
    """
    input_dim = scaler.n_features_in_
    features = data.features[:input_dim]
    if len(features) < input_dim:
        # Pad missing features with neutral Likert score (3.0)
        features += [3.0] * (input_dim - len(features))
        
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    latent_features = encoder.predict(X_scaled, verbose=0)
    
    predicted_interest = data.known_interest if data.known_interest != -1 else int(np.random.randint(0, 3))
    segment = int(kmeans.predict(latent_features)[0])
    rec = ml_pipeline.map_recommendations([segment], [predicted_interest])[0]
    
    return {
        "status": "success",
        "latent_representation": latent_features.tolist()[0],
        "marketing_insights": rec
    }
