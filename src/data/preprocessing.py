import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
from typing import Tuple
import logging
import json
from src.config import paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        pass

    def load_data(self, users_path: str, games_path: str, 
                 recommendations_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess the data
        """
        try:
            logger.info("Loading data...")
            users = pd.read_csv(users_path)
            games = pd.read_csv(games_path)
            recommendations = pd.read_csv(recommendations_path)
            
            logger.info("Data loaded successfully")
            return users,games,recommendations
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise


    def remove_duplicates_recommendations(self, recommendations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate entries from the DataFrame
        """
        duplicates = recommendations_df.duplicated(subset=['app_id', 'user_id']).sum()
        logger.info(f"Number of duplicate recommendations: {duplicates}")
        recommendations_df['date'] = pd.to_datetime(recommendations_df['date'])
        recommendations_df = recommendations_df.sort_values(by='date', ascending=False).drop_duplicates(subset=['app_id', 'user_id'], keep='first')

        return recommendations_df
    

    def create_recommendations_stats(self, recommendations_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistics for recommendations
        """
        try:
            logger.info("Creating recommendations stats...")
            recommendation_stats = recommendations_df.groupby("app_id").agg(
                    median_playtime=("hours", "median"),
                    proportion_recommended=("is_recommended", lambda x: x.mean())
                    ).reset_index()
            
            games_df = games_df.merge(
                recommendation_stats,
                    how="left",
                    left_on="app_id",
                    right_on="app_id"
            ).fillna(0)
            
            logger.info("Recommendations stats created successfully and added to games!")
            return games_df
        except Exception as e:
            logger.error(f"Error creating recommendations stats: {str(e)}")
            raise


    def add_games_metadata(self, games_df: pd.DataFrame, metadata_path: str) -> pd.DataFrame:
        """
        Add metadata to games DataFrame.
        Create a "content" column with the description and tags of the game.
        """

        try:
            with open(metadata_path, 'r') as file:
                metadata = [json.loads(line) for line in file]

            metadata_df = pd.DataFrame(metadata)

            games_df = games_df.merge(metadata_df[['app_id', 'description', 'tags']], on='app_id', how='left')
            games_df['content'] = games_df['title'] + ' ' + games_df['description'] + ' ' + games_df['tags'].apply(lambda tags: ' '.join(tags))

            logger.info("Metadata added successfully")
            return games_df
        except Exception as e:
            logger.error(f"Error adding metadata: {str(e)}")
            raise


    def remove_games_without_metadata_or_interactions(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove games without metadata or interactions: games with no description are not suitable for content-based recommendations
        and games with no interactions are not suitable for collaborative filtering
        """
        try:
            games_to_delete = games_df[
            (games_df['description'].isna() | 
            (games_df['description'] == '')) & 
            (games_df['tags'].isna() | 
            (games_df['tags'].apply(len) == 0)) & 
            ((games_df['median_playtime'] == 0) |
            (games_df['proportion_recommended'] == 0))
            ]

            missing_app_ids = games_to_delete['app_id'].tolist()
            logger.info(f"Games to remove: {len(missing_app_ids)}")

            # Remove from games_df the games with missing metadata or median_playtime == 0
            games_df = games_df[~games_df['app_id'].isin(missing_app_ids)]
            games_df = games_df.reset_index(drop=True)
        
        except Exception as e:
            logger.error(f"Error removing games without metadata or interactions: {str(e)}")
            raise
        return games_df
    

    def train_test_split(self, recommendations_df: pd.DataFrame, train_size: float = 0.9) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split recommendations DataFrame into train and test sets. 
        """

        try:
            recommendations_df = recommendations_df.sort_values(by='date')
            cutoff_index = int(len(recommendations_df) * train_size)
            cutoff_date = recommendations_df.iloc[cutoff_index]['date']

            train_df = recommendations_df[recommendations_df['date'] <= cutoff_date]
            test_df = recommendations_df[recommendations_df['date'] > cutoff_date]

            return train_df, test_df
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise


    def export_cleaned_data(self, games_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame,
                            games_path: str, train_path: str, test_path: str) -> None:
        """
        Export cleaned data to respective path. Parquet format is used for better performance
        """

        try:
            logger.info("Exporting cleaned data...")
            games_df.to_parquet(games_path, index=False)
            train_df.to_parquet(train_path, index=False)
            test_df.to_parquet(test_path, index=False)
            logger.info("Data exported successfully")

        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise


def main():
    # Paths to the raw data files
    users_path = paths.get_raw_data_file("users.csv")
    games_path = paths.get_raw_data_file("games.csv")
    recommendations_path = paths.get_raw_data_file("recommendations.csv")

    # Paths to the cleaned data files
    cleaned_games_path = paths.get_cleaned_data_file("games_cleaned.parquet")
    cleaned_train_path = paths.get_cleaned_data_file("train.parquet")
    cleaned_test_path = paths.get_cleaned_data_file("test.parquet")

    # Initialize the DataPreprocessor
    preprocessor = DataPreprocessor()

    # Load the data
    users_df, games_df, recommendations_df = preprocessor.load_data(users_path, games_path, recommendations_path)

    # Remove duplicates from recommendations
    recommendations_df = preprocessor.remove_duplicates_recommendations(recommendations_df)

    # Create recommendation stats and add them to games DataFrame
    games_df = preprocessor.create_recommendations_stats(recommendations_df, games_df)

    # Add metadata to games DataFrame
    metadata_path = paths.get_raw_data_file("games_metadata.json")
    games_df = preprocessor.add_games_metadata(games_df, metadata_path)

    # Remove games without metadata or interactions
    games_df = preprocessor.remove_games_without_metadata_or_interactions(games_df)
    

    # Split recommendations into train and test sets
    train_df, test_df = preprocessor.train_test_split(recommendations_df)

    # Export cleaned data
    preprocessor.export_cleaned_data(games_df, train_df, test_df, cleaned_games_path, cleaned_train_path, cleaned_test_path)
    logger.info("Data preprocessing completed successfully")


if __name__ == "__main__":
    main()