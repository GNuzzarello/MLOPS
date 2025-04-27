import pandas as pd
import numpy as np
import mlflow
import logging
import sys
import os
import gc
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import paths
from src.config.mlflow_config import MLFLOW_CONFIG
from src.models.collaborative_filtering import load_model as load_collaborative_model
from src.models.content_recommender import load_model as load_content_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRecommender(mlflow.pyfunc.PythonModel):
    def __init__(self, collaborative_weight: float = 0.6):
        self.collaborative_model = None
        self.content_model = None
        self.collaborative_weight = collaborative_weight
        self.content_weight = 1 - collaborative_weight
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_user = None
        self.idx_to_item = None
        self.tfidf_matrix = None
        self.interaction_matrix = None

    def load_models(self) -> None:
        """
        Load the collaborative and content-based models using their respective load_model functions
        """
        try:
            # Set MLFlow tracking URI
            mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
            
            # Load collaborative model
            self.collaborative_model = load_collaborative_model()
            logger.info("Collaborative model loaded successfully")
            
            # Load content model
            self.content_model = load_content_model()
            logger.info("Content model loaded successfully")
            
            # Load necessary data
            self._load_data()
            
            logger.info("Both models and data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def _load_data(self):
        """Load necessary data for recommendations"""
        # Load games data
        games = pd.read_parquet(paths.get_cleaned_data_file("games_cleaned.parquet"))
        
        # Create mappings
        unique_items = games['app_id'].unique()
        self.item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_item = {idx: iid for iid, idx in self.item_to_idx.items()}
        
        # Load matrices
        # Access the content model's attributes through the MLflow wrapper
        content_model_impl = self.content_model._model_impl
        self.tfidf_matrix = content_model_impl.tfidf_matrix
        
        # Load interaction matrix from collaborative model
        self.interaction_matrix = self.collaborative_model.data_preprocessor.interaction_matrix
        
        # Load user mappings
        train_df = pd.read_parquet(paths.get_cleaned_data_file("train.parquet"))
        unique_users = train_df['user_id'].unique()
        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}

    def get_collaborative_recommendations(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations from collaborative filtering model"""
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        recommended_items = self.collaborative_model.get_recommendations(user_id, n=n)
        
        # Filter out already seen items
        recommended_items_filtered = []
        for idx, score in recommended_items:
            if self.interaction_matrix[user_idx, self.item_to_idx[idx]] == 0:
                recommended_items_filtered.append((idx, score))
        
        return sorted(recommended_items_filtered, key=lambda x: x[1], reverse=True)

    def get_content_recommendations(self, user_id: int, n: int = 10, mode: str = 'train') -> List[Tuple[int, float]]:
        """Get recommendations from content-based model"""
        # Load appropriate dataset
        if mode == 'train':
            dataset = pd.read_parquet(paths.get_cleaned_data_file("train.parquet"))
        else:
            train_df = pd.read_parquet(paths.get_cleaned_data_file("train.parquet"))
            test_df = pd.read_parquet(paths.get_cleaned_data_file("test.parquet"))
            dataset = pd.concat([train_df, test_df])
        
        recommended_games_dict = {}
        
        # Get recommendations for each game the user has interacted with
        for game_id in dataset[dataset['user_id'] == user_id]['app_id']:
            game_idx = self.item_to_idx[game_id]
            game_vector = self.tfidf_matrix[game_idx]
            
            # Calculate cosine similarity
            similarity_scores = cosine_similarity(
                game_vector.reshape(1, -1), 
                self.tfidf_matrix
            )[0]
            
            # Get top N similar games
            similar_indices = similarity_scores.argsort()[::-1][1:n+1]
            recommended_games = [self.idx_to_item[idx] for idx in similar_indices]
            recommended_similarity = similarity_scores[similar_indices]
            
            # Update recommendations dictionary
            for rec_game_id, similarity in zip(recommended_games, recommended_similarity):
                if rec_game_id != game_id:
                    if rec_game_id not in recommended_games_dict:
                        recommended_games_dict[rec_game_id] = similarity
                    else:
                        recommended_games_dict[rec_game_id] = max(
                            recommended_games_dict[rec_game_id], 
                            similarity
                        )
        
        # Sort and return top N recommendations
        sorted_recommendations = sorted(
            recommended_games_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_recommendations[:n]

    def get_recommendations(self, user_id: int, n: int = 10, mode: str = 'train') -> List[Tuple[int, float]]:
        """Get hybrid recommendations by combining both models"""
        collaborative_recs = self.get_collaborative_recommendations(user_id, n=n)
        content_recs = self.get_content_recommendations(user_id, n=n, mode=mode)
        
        # Combine recommendations with weights
        weighted_recommendations = {}
        
        # Add collaborative recommendations
        for item, score in collaborative_recs:
            weighted_recommendations[item] = score * self.collaborative_weight
        
        # Add content recommendations
        for item, score in content_recs:
            if item in weighted_recommendations:
                weighted_recommendations[item] += score * self.content_weight
            else:
                weighted_recommendations[item] = score * self.content_weight
        
        # Sort by combined score
        sorted_recommendations = sorted(
            weighted_recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_recommendations[:n]

    def predict(self, context, model_input):
        """Predict method required by MLFlow"""
        user_id = model_input['user_id'].iloc[0]
        n = model_input.get('n', 10).iloc[0]
        mode = model_input.get('mode', 'train').iloc[0]
        
        recommendations = self.get_recommendations(user_id, n=n, mode=mode)
        return pd.DataFrame(recommendations, columns=['app_id', 'score'])

    def evaluate(self, test_df: pd.DataFrame, train_df: pd.DataFrame, n_users: int = 1000) -> Dict[str, float]:
        """Evaluate the hybrid model on test data"""
        def calculate_precision(user_id: int, recommended_items: List[Tuple[int, float]], test_df: pd.DataFrame) -> float:
            actual_items = test_df[(test_df['user_id'] == user_id) & (test_df['is_recommended'] == 1)]['app_id']
            recommended_item_ids = [item[0] for item in recommended_items]
            relevant_recommendations = sum(1 for item in recommended_item_ids if item in actual_items.values)
            return relevant_recommendations / len(recommended_item_ids) if len(recommended_item_ids) > 0 else 0

        def calculate_recall(user_id: int, recommended_items: List[Tuple[int, float]], test_df: pd.DataFrame) -> float:
            actual_items = test_df[(test_df['user_id'] == user_id) & (test_df['is_recommended'] == 1)]['app_id']
            recommended_item_ids = [item[0] for item in recommended_items]
            relevant_recommendations = sum(1 for item in recommended_item_ids if item in actual_items.values)
            return relevant_recommendations / len(actual_items) if len(actual_items) > 0 else 0

        def calculate_f1_score(precision: float, recall: float) -> float:
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        def calculate_hit_ratio(user_id: int, recommended_items: List[Tuple[int, float]], test_df: pd.DataFrame) -> int:
            actual_items = test_df[(test_df['user_id'] == user_id) & (test_df['is_recommended'] == 1)]['app_id']
            recommended_item_ids = [item[0] for item in recommended_items]
            return 1 if any(item in actual_items.values for item in recommended_item_ids) else 0

        # Get users present in both test and train sets
        test_users = set(test_df['user_id'].unique())
        train_users = set(train_df['user_id'].unique())
        known_users = list(test_users & train_users)

        # Filter users with at least 10 interactions in test set
        user_counts = test_df.groupby('user_id').size()
        known_users_filtered = [user_id for user_id in known_users if user_counts[user_id] >= 10]
        
        # Limit to n_users for evaluation
        known_users_filtered = known_users_filtered[:n_users]

        metrics = []
        for user_id in known_users_filtered:
            recommended_items = self.get_recommendations(user_id)
            precision = calculate_precision(user_id, recommended_items, test_df)
            recall = calculate_recall(user_id, recommended_items, test_df)
            f1 = calculate_f1_score(precision, recall)
            hit = calculate_hit_ratio(user_id, recommended_items, test_df)
            metrics.append((precision, recall, f1, hit))

        avg_precision = np.mean([m[0] for m in metrics])
        avg_recall = np.mean([m[1] for m in metrics])
        avg_f1 = np.mean([m[2] for m in metrics])
        avg_hit = np.mean([m[3] for m in metrics])

        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1,
            "hit_ratio": avg_hit
        }

def train_with_mlflow(test_df: pd.DataFrame, train_df: pd.DataFrame,
                     collaborative_weights: List[float] = [0.3, 0.5, 0.7]) -> None:
    """Train the hybrid model with different weights and track with MLFlow"""
    # Set MLFlow tracking URI and enable tracking
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_hybrid_name"])
    
    # Enable autologging
    mlflow.autolog()

    # Create input example for model signature
    input_example = pd.DataFrame({
        'user_id': [train_df['user_id'].iloc[0]],
        'n': [10],
        'mode': ['train']
    })

    # Create base model for signature
    base_model = HybridRecommender(collaborative_weight=0.5)
    base_model.load_models()

    # Create model signature
    signature = mlflow.models.infer_signature(
        model_input=input_example,
        model_output=base_model.predict(None, input_example)
    )

    # Clean up base model
    del base_model
    gc.collect()

    for collaborative_weight in collaborative_weights:
        logger.info(f"Training with collaborative_weight={collaborative_weight}")
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                "collaborative_weight": collaborative_weight,
                "content_weight": 1 - collaborative_weight
            })

            # Create and train the model
            model = HybridRecommender(collaborative_weight=collaborative_weight)
            model.load_models()

            # Evaluate and log metrics
            metrics = model.evaluate(test_df, train_df)
            mlflow.log_metrics(metrics)

            # Save model using MLFlow's format
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model,
                registered_model_name=MLFLOW_CONFIG["registered_hybrid_model_name"],
                signature=signature,
                input_example=input_example
            )
            
            # Clear memory
            del model
            gc.collect()

def find_best_model(experiment_name: str = MLFLOW_CONFIG["experiment_hybrid_name"], 
                   metric: str = "hit_ratio") -> str:
    """Find the best model based on a specified metric and register it"""
    # Set MLFlow tracking URI
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    
    # Get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")
    
    # Get all runs from the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        raise ValueError("No runs found in the experiment")
    
    # Find the run with the best metric
    best_run = runs.loc[runs[f"metrics.{metric}"].idxmax()]
    
    # Get the run ID and metrics
    best_run_id = best_run["run_id"]
    best_metrics = {
        "precision": best_run["metrics.precision"],
        "recall": best_run["metrics.recall"],
        "f1_score": best_run["metrics.f1_score"],
        "hit_ratio": best_run["metrics.hit_ratio"]
    }
    
    # Log the best model information
    logger.info(f"Best model found in run {best_run_id}")
    logger.info(f"Best {metric}: {best_metrics[metric]:.4f}")
    logger.info(f"Other metrics: {best_metrics}")
    logger.info(f"Collaborative weight: {best_run['params.collaborative_weight']}")

    # Register the model with a new version
    model_name = MLFLOW_CONFIG["registered_hybrid_model_name"]
    model_uri = f"runs:/{best_run_id}/model"
    
    # Get current model version
    client = mlflow.tracking.MlflowClient()
    try:
        current_model = client.get_latest_versions(model_name, stages=["Production"])
        current_version = current_model[0].version if current_model else None
    except:
        current_version = None

    # Register new version
    new_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    # If there's a current production model, compare metrics
    if current_version:
        current_metrics = client.get_run(current_model[0].run_id).data.metrics
        if best_metrics[metric] > current_metrics[metric]:
            # Transition new version to Production
            client.transition_model_version_stage(
                name=model_name,
                version=new_version.version,
                stage="Production"
            )
            # Archive old version
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Archived"
            )
            logger.info(f"New model version {new_version.version} promoted to Production")
        else:
            logger.info(f"New model version {new_version.version} not better than current Production version")
    else:
        # First model, automatically promote to Production
        client.transition_model_version_stage(
            name=model_name,
            version=new_version.version,
            stage="Production"
        )
        logger.info(f"First model version {new_version.version} set to Production")

    return new_version.version

def load_model(version: str = None) -> mlflow.pyfunc.PythonModel:
    """Load a specific version of the model or the latest production version"""
    model_name = MLFLOW_CONFIG["registered_hybrid_model_name"]
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    
    if version:
        model_uri = f"models:/{model_name}/{version}"
    else:
        # Get latest production version
        client = mlflow.tracking.MlflowClient()
        production_models = client.get_latest_versions(model_name, stages=["Production"])
        if not production_models:
            raise ValueError("No production model found")
        model_uri = f"models:/{model_name}/{production_models[0].version}"
    
    return mlflow.pyfunc.load_model(model_uri)

def main():
    # Load data
    train_df = pd.read_parquet(paths.get_cleaned_data_file("train.parquet"))
    test_df = pd.read_parquet(paths.get_cleaned_data_file("test.parquet"))

    # Train with different collaborative weights
    train_with_mlflow(test_df, train_df)
    
    # Find and register the best model
    find_best_model()

if __name__ == "__main__":
    main()
