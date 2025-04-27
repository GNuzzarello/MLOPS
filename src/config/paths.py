from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Main directories
DATA_DIR = ROOT_DIR / "data"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
DATA_CLEANING_DIR = ROOT_DIR / DATA_DIR / "cleaned"
MODELS_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, NOTEBOOKS_DIR, DATA_CLEANING_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def get_data_dir() -> Path:
    """
    Returns the path to the data directory.
    """
    return DATA_DIR

def get_notebooks_dir() -> Path:
    """
    Returns the path to the notebooks directory.
    """
    return NOTEBOOKS_DIR

def get_data_cleaning_dir() -> Path:    
    """
    Returns the path to the data cleaning directory.
    """
    return DATA_CLEANING_DIR

# Filename management functions
def get_raw_data_file(filename: str) -> Path:
    """
    Returns the full path for a raw data file.
    Example: get_raw_data_file("games.csv") -> data/games.csv
    """
    return DATA_DIR / filename

def get_cleaned_data_file(filename: str) -> Path:
    """
    Returns the full path for a cleaned data file.
    Example: get_cleaned_data_file("games_cleaned.csv") -> data/cleaned/games_cleaned.csv
    """
    return DATA_CLEANING_DIR / filename

def get_notebook_file(notebook_name: str) -> Path:
    """
    Returns the full path for a notebook file.
    Example: get_notebook_file("01_data_analysis.ipynb") -> notebooks/01_data_analysis.ipynb
    """
    return NOTEBOOKS_DIR / notebook_name

def get_models_dir() -> Path:
    """
    Returns the path to the models directory.
    """
    return MODELS_DIR

def get_model_file(model_name: str) -> Path:
    """
    Returns the full path for a model file.
    Example: get_model_file("collaborative_filtering_model.pkl") -> models/collaborative_filtering_model.pkl
    """
    return MODELS_DIR / model_name