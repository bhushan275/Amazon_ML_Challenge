"""
Pipeline configuration settings and path declarations for Business Entity Resolution.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


# Base & Output Directory Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATASET_PATH: Path = BASE_DIR / "dataset"
TRAIN_PATH: Path = DATASET_PATH / "train"
TEST_PATH: Path = DATASET_PATH / "test"
OUTPUT_PATH: Path = BASE_DIR / "output"

# Raw Source Files Mapping (relative to train/test directories)
RAW_SOURCE_FILES: Dict[str, str] = {
    "source1": "train_source1.tsv",
    "source2": "train_source2.tsv",
    "source3": "train_source3.tsv",
    "ground_truth": "train_ground_truth.tsv",
}

TEST_RAW_SOURCE_FILES: Dict[str, str] = {
    "source1": "test_source1.tsv",
    "source2": "test_source2.tsv",
    "source3": "test_source3.tsv",
}


@dataclass
class Config:
    """Central configuration parameters for dataset paths, blocking, feature extraction, and models."""

    # File System Paths
    BASE_DIR: Path = BASE_DIR
    DATASET_PATH: Path = DATASET_PATH
    TRAIN_PATH: Path = TRAIN_PATH
    TEST_PATH: Path = TEST_PATH
    OUTPUT_PATH: Path = OUTPUT_PATH

    # Raw Source Files
    RAW_SOURCE_FILES: Dict[str, str] = None

    # Blocking & Retrieval Parameters
    TOP_K_CANDIDATES: int = 50
    MIN_NGRAM_LEN: int = 3
    MAX_NGRAM_LEN: int = 5

    # Feature Generation Parameters
    SIMILARITY_METRICS: tuple = ("fuzzy_ratio", "token_sort", "jaccard", "levenshtein")

    # Model Hyperparameters
    CLASSIFIER_TYPE: str = "lightgbm"
    CLASSIFIER_PARAMS: Dict[str, Any] = None

    # Clustering / Postprocessing
    MATCH_THRESHOLD: float = 0.5

    def __post_init__(self):
        if self.RAW_SOURCE_FILES is None:
            self.RAW_SOURCE_FILES = RAW_SOURCE_FILES.copy()

        if self.CLASSIFIER_PARAMS is None:
            self.CLASSIFIER_PARAMS = {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 6,
                "random_state": 42,
            }

        # Ensure output directory exists
        self.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


config = Config()

