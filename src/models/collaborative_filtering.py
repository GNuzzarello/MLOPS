import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import implicit
import pickle
import os
import logging
import mlflow
from typing import Tuple, Dict, List
from tqdm import tqdm
import sys
import os
import gc  # Add garbage collector

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import paths
from src.config.mlflow_config import MLFLOW_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_user = None
        self.idx_to_item = None
        self.interaction_matrix = None

    def prepare_data(self, train_df: pd.DataFrame, games_df: pd.DataFrame) -> None:
        """
        Prepare data for training by creating mappings and interaction matrix
        """
        # Create mappings
        unique_users = train_df['user_id'].unique()
        unique_items = games_df['app_id'].unique()

        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: iid for iid, idx in self.item_to_idx.items()}

        # Create sparse matrix
        user_indices = train_df['user_id'].map(self.user_to_idx)
        item_indices = train_df['app_id'].map(self.item_to_idx)
        interaction_strength = train_df['is_recommended'] * np.log1p(train_df['hours'])

        self.interaction_matrix = coo_matrix((interaction_strength, (user_indices, item_indices))).tocsr()

class CollaborativeFiltering(mlflow.pyfunc.PythonModel):
    def __init__(self, data_preprocessor: DataPreprocessor = None):
        self.model = None
        self.data_preprocessor = data_preprocessor

    def prepare_data(self, train_df: pd.DataFrame, games_df: pd.DataFrame) -> None:
        """
        Prepare data for training by creating mappings and interaction matrix
        """
        if self.data_preprocessor is None:
            self.data_preprocessor = DataPreprocessor()
        self.data_preprocessor.prepare_data(train_df, games_df)

    def train(self, factors: int = 20, regularization: float = 0.1, iterations: int = 10) -> None:
        """
        Train the ALS model
        """
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            num_threads=8
        )
        self.model.fit(self.data_preprocessor.interaction_matrix)

    def get_recommendations(self, user_id: int, n: int = 10, filter_already_recommended: bool = True) -> List[Tuple[int, float]]:
        """
        Get recommendations for a user
        """
        if user_id not in self.data_preprocessor.user_to_idx:
            raise ValueError(f"User ID {user_id} not found in mappings.")

        user_idx = self.data_preprocessor.user_to_idx[user_id]
        recommended_items = self.model.recommend(user_idx, self.data_preprocessor.interaction_matrix[user_idx], N=n)

        if not filter_already_recommended:
            return [(self.data_preprocessor.idx_to_item[idx], score) for idx, score in zip(recommended_items[0], recommended_items[1])]

        recommended_items_filtered = []
        for idx, score in zip(recommended_items[0], recommended_items[1]):
            if self.data_preprocessor.interaction_matrix[user_idx, idx] == 0:
                recommended_items_filtered.append((self.data_preprocessor.idx_to_item[idx], score))

        return sorted(recommended_items_filtered, key=lambda x: x[1], reverse=True)

    def predict(self, context, model_input):
        """
        Predict method required by MLFlow
        """
        user_id = model_input['user_id'].iloc[0]
        n = model_input.get('n', 10).iloc[0]
        recommendations = self.get_recommendations(user_id, n=n)
        return pd.DataFrame(recommendations, columns=['app_id', 'score'])

    def evaluate(self, test_df: pd.DataFrame, n_users: int = 1000) -> Dict[str, float]:
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
        train_users = set(self.data_preprocessor.user_to_idx.keys())
        known_users = list(test_users & train_users)

        # Filter users with at least 10 interactions in test set
        user_counts = test_df.groupby('user_id').size()
        known_users_filtered = [user_id for user_id in known_users if user_counts[user_id] >= 10]
        
        # Limit to n_users for evaluation
        known_users_filtered = known_users_filtered[:n_users]

        metrics = []
        for user_id in tqdm(known_users_filtered):
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

def train_with_mlflow(train_df: pd.DataFrame, games_df: pd.DataFrame, test_df: pd.DataFrame,
                     factors_list: List[int] = [20], iterations_list: List[int] = [1, 2, 3],
                     regularization: float = 0.1) -> None:
    """
    Train the model with different hyperparameters and track with MLFlow
    """
    # Set MLFlow tracking URI and enable tracking
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_collaborative_name"])
    
    # Enable autologging
    mlflow.autolog()

    # Prepare data once
    data_preprocessor = DataPreprocessor()
    data_preprocessor.prepare_data(train_df, games_df)

    # Create input example for model signature
    input_example = pd.DataFrame({
        'user_id': [train_df['user_id'].iloc[0]],
        'n': [10]
    })

    # Create temporary model for signature
    temp_model = CollaborativeFiltering(data_preprocessor=data_preprocessor)
    temp_model.train(factors=1, iterations=1, regularization=regularization)

    # Create model signature
    signature = mlflow.models.infer_signature(
        model_input=input_example,
        model_output=temp_model.predict(None, input_example)
    )

    # Clean up temporary model
    del temp_model
    gc.collect()

    for factors in factors_list:
        for iterations in iterations_list:
            logger.info(f"Training with factors={factors}, iterations={iterations}, regularization={regularization}")
            with mlflow.start_run():
                # Log hyperparameters
                mlflow.log_params({
                    "factors": factors,
                    "iterations": iterations,
                    "regularization": regularization
                })

                # Create new model instance with shared data preprocessor
                cf = CollaborativeFiltering(data_preprocessor=data_preprocessor)

                # Train model with current hyperparameters
                cf.train(factors=factors, iterations=iterations, regularization=regularization)

                # Evaluate and log metrics
                metrics = cf.evaluate(test_df)
                mlflow.log_metrics(metrics)

                # Save model using MLFlow's format with signature and input example
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=cf,
                    registered_model_name=MLFLOW_CONFIG["registered_collaborative_model_name"],
                    signature=signature,
                    input_example=input_example
                )

                # Log additional information
                mlflow.log_param("num_users", len(data_preprocessor.user_to_idx))
                mlflow.log_param("num_items", len(data_preprocessor.item_to_idx))
                
                # Clear memory after each run
                del cf
                gc.collect()  # Force garbage collection

    # Clear remaining memory
    del data_preprocessor
    gc.collect()

def find_best_model(experiment_name: str = MLFLOW_CONFIG["experiment_collaborative_name"], metric: str = "hit_ratio") -> str:
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
    logger.info(f"Model parameters: factors={best_run['params.factors']}, iterations={best_run['params.iterations']}")

    # Register the model with a new version
    model_name = MLFLOW_CONFIG["registered_collaborative_model_name"]
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
    model_name = MLFLOW_CONFIG["registered_collaborative_model_name"]
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
    games_df = pd.read_parquet(paths.get_cleaned_data_file("games_cleaned.parquet"))

    # Train with different hyperparameters
    #train_with_mlflow(train_df, games_df, test_df)
    
    # Find and register the best model
    #find_best_model()

    # Load the best model
    model_version = load_model()
    print(f"Best model version: {model_version}")

if __name__ == "__main__":
    main()
