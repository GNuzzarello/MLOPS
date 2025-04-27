import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import logging
import sys
import os
import gc
from typing import List, Tuple, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import paths
from src.config.mlflow_config import MLFLOW_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentRecommender(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.games_df = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_user = None
        self.idx_to_item = None
        self.ngram_range = (1, 1)  # Default value

    def prepare_data(self, games_df: pd.DataFrame, train_df: pd.DataFrame) -> None:
        """
        Prepare data for the content-based recommender
        """
        
        # Create mappings
        unique_users = train_df['user_id'].unique()
        unique_items = games_df['app_id'].unique()

        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: iid for iid, idx in self.item_to_idx.items()}

        # Prepare content for TF-IDF
        self.games_df = games_df.copy()

    def train(self, ngram_range: tuple = (1, 1)) -> None:
        # Create TF-IDF matrix with specified ngram_range
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.games_df['content'])

    def recommend_content_based(self, game_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Recommend similar games based on content
        """
        if game_id not in self.item_to_idx:
            raise ValueError(f"Game ID {game_id} not found in mappings.")

        game_idx = self.item_to_idx[game_id]
        
        # Extract the game vector
        game_vector = self.tfidf_matrix[game_idx]
        
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(game_vector.reshape(1, -1), self.tfidf_matrix)[0]
        
        # Sort games by similarity (descending)
        similar_indices = similarity_scores.argsort()[::-1][1:n+1]  # Skip the game itself
        
        # Get app_ids and similarity scores
        recommended_game_ids = [self.idx_to_item[idx] for idx in similar_indices]
        recommended_scores = similarity_scores[similar_indices].tolist()
        
        return list(zip(recommended_game_ids, recommended_scores))

    def get_recommendations_for_user(self, user_id: int, train_df: pd.DataFrame, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get content-based recommendations for a user
        """
        recommended_games_dict = {}

        # Iterate over all games the user has interacted with
        for game_id in train_df[train_df['user_id'] == user_id]['app_id']:
            # Get recommendations for the current game
            recommended_games = self.recommend_content_based(game_id, n=10)
            
            # Add recommendations to dictionary, excluding the game itself
            for rec_game_id, similarity in recommended_games:
                if rec_game_id != game_id:  # Exclude the game itself
                    if rec_game_id not in recommended_games_dict:
                        recommended_games_dict[rec_game_id] = similarity
                    else:
                        # Keep only the highest similarity for each game
                        recommended_games_dict[rec_game_id] = max(recommended_games_dict[rec_game_id], similarity)

        # Sort games by similarity and take top n
        sorted_recommendations = sorted(recommended_games_dict.items(), key=lambda x: x[1], reverse=True)
        top_n_games = sorted_recommendations[:n]
        
        return top_n_games

    def predict(self, model_input):
        """
        Predict method required by MLFlow
        """
        user_id = model_input['user_id'].iloc[0]
        n = model_input.get('n', 10).iloc[0]
        train_df = pd.read_parquet(paths.get_cleaned_data_file("train.parquet"))
        recommendations = self.get_recommendations_for_user(user_id, train_df, n=n)
        return pd.DataFrame(recommendations, columns=['app_id', 'score'])

    def evaluate(self, test_df: pd.DataFrame, train_df: pd.DataFrame, n_users: int = 1000) -> Dict[str, float]:
        """
        Evaluate the model on test data
        """
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
            recommended_items = self.get_recommendations_for_user(user_id, train_df)
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

def train_with_mlflow(games_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame,
                     ngram_ranges: List[tuple] = [(1, 1), (2, 3)]) -> None:
    """
    Train the content-based model with different hyperparameters and track with MLFlow
    """
    # Set MLFlow tracking URI and enable tracking
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_content_name"])
    
    # Enable autologging
    mlflow.autolog()

    # Create input example for model signature
    input_example = pd.DataFrame({
        'user_id': [train_df['user_id'].iloc[0]],
        'n': [10]
    })

    # Create base model and prepare all data once
    base_model = ContentRecommender()
    base_model.prepare_data(games_df, train_df)
    base_model.train(ngram_range=(1,1))  # Train the base model

    # Create model signature
    signature = mlflow.models.infer_signature(
        model_input=input_example,
        model_output=base_model.predict(None, input_example)
    )

    for ngram_range in ngram_ranges:
        logger.info(f"Training with ngram_range={ngram_range}")
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                "ngram_range": str(ngram_range)
            })

            # Create new model instance
            model = ContentRecommender()
            logger.info(f"ContentRecommender model created with ngram_range={ngram_range}")

            
            # Copy all prepared data from base model
            model.user_to_idx = base_model.user_to_idx
            model.item_to_idx = base_model.item_to_idx
            model.idx_to_user = base_model.idx_to_user
            model.idx_to_item = base_model.idx_to_item
            model.games_df = base_model.games_df
            model.ngram_range = ngram_range

            # Train the model
            logger.info(f"Starting training with ngram_range={ngram_range}")
            model.train(ngram_range=model.ngram_range)
            logger.info(f"Training completed with ngram_range={ngram_range}")

            # Evaluate and log metrics
            logger.info(f"Evaluating model with ngram_range={ngram_range}")
            metrics = model.evaluate(test_df, train_df)
            mlflow.log_metrics(metrics)

            # Save model using MLFlow's format
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model,
                registered_model_name=MLFLOW_CONFIG["registered_content_model_name"],
                signature=signature,
                input_example=input_example
            )

            # Log additional information
            mlflow.log_param("num_users", len(model.user_to_idx))
            mlflow.log_param("num_items", len(model.item_to_idx))
            
            # Clear memory
            del model
            gc.collect()

    # Clean up base model
    del base_model
    gc.collect()

def find_best_model(experiment_name: str = MLFLOW_CONFIG["experiment_content_name"], metric: str = "hit_ratio") -> str:
    """
    Find the best model based on a specified metric and register it
    Returns the version of the registered model
    """
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

    # Register the model with a new version
    model_name = MLFLOW_CONFIG["registered_content_model_name"]
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
    """
    Load a specific version of the model or the latest production version
    """
    model_name = MLFLOW_CONFIG["registered_content_model_name"]
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
    games_df = pd.read_parquet(paths.get_cleaned_data_file("games_cleaned.parquet"))
    train_df = pd.read_parquet(paths.get_cleaned_data_file("train.parquet"))
    test_df = pd.read_parquet(paths.get_cleaned_data_file("test.parquet"))

    # Train with different hyperparameters
    train_with_mlflow(games_df, train_df, test_df)
    
    # Find and register the best model
    find_best_model()

if __name__ == "__main__":
    main()
