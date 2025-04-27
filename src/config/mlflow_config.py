"""
Configuration settings for MLflow
"""

MLFLOW_CONFIG = {
    "tracking_uri": "http://localhost:5001",
    "experiment_collaborative_name": "collaborative_filtering",
    "experiment_content_name": "content_based_recommender",
    "experiment_hybrid_name": "hybrid_recommender",
    "registered_collaborative_model_name": "collaborative_filtering_model",
    "registered_content_model_name": "content_based_model",
    "registered_hybrid_model_name": "hybrid_recommender_model"
} 