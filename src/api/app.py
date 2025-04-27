from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from src.models.hybrid_recommender import HybridRecommender
import mlflow
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'nba_api_requests_total',
    'Total number of requests to the NBA API',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'nba_api_request_duration_seconds',
    'Request latency in seconds',
    ['endpoint']
)

RECOMMENDATION_COUNT = Counter(
    'nba_api_recommendations_total',
    'Total number of recommendations made',
    ['user_id']
)

app = FastAPI(title="Next Best Action API",
             description="API for recommending next best actions (games) to users",
             version="1.0.0")

class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: Optional[int] = 10

class RecommendationResponse(BaseModel):
    recommendations: List[dict]
    user_id: int

# Initialize the recommender
recommender = HybridRecommender()

@app.on_event("startup")
async def startup_event():
    try:
        # Load models and mappings
        model_path = os.getenv("MODEL_PATH", "../models/als_model.pkl")
        interaction_matrix_path = os.getenv("INTERACTION_MATRIX_PATH", "../models/interaction_matrix.pkl")
        tfidf_matrix_path = os.getenv("TFIDF_MATRIX_PATH", "../models/tfidf_matrix.pkl")
        
        recommender.load_models(model_path, interaction_matrix_path, tfidf_matrix_path)
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    start_time = time.time()
    try:
        # Get recommendations
        recommendations = recommender.get_hybrid_recommendations(
            request.user_id,
            request.num_recommendations
        )
        
        # Format response
        formatted_recommendations = [
            {"game_id": game_id, "score": float(score)}
            for game_id, score in recommendations
        ]
        
        # Update metrics
        REQUEST_COUNT.labels(endpoint='/recommendations', status='success').inc()
        RECOMMENDATION_COUNT.labels(user_id=str(request.user_id)).inc(len(recommendations))
        
        return RecommendationResponse(
            recommendations=formatted_recommendations,
            user_id=request.user_id
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='/recommendations', status='error').inc()
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(endpoint='/recommendations').observe(time.time() - start_time)

@app.get("/health")
async def health_check():
    start_time = time.time()
    try:
        REQUEST_COUNT.labels(endpoint='/health', status='success').inc()
        return {"status": "healthy"}
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='/health', status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(endpoint='/health').observe(time.time() - start_time)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 