"""
Classifier model interface for pairwise entity resolution matching.
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
import pandas as pd


class EntityMatcher:
    """Classifier model wrapper for predicting match probabilities between candidate entity pairs."""

    def __init__(self, model_type: str = "lightgbm", params: Optional[Dict[str, Any]] = None):
        """Initialize entity matching model.

        Args:
            model_type: Name of classification model ('lightgbm', 'xgboost', 'random_forest').
            params: Dictionary of model hyperparameters.
        """
        self.model_type = model_type
        self.params = params or {}
        self.model = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> "EntityMatcher":
        """Train pairwise entity matching classifier.

        Args:
            X: Feature matrix DataFrame for candidate pairs.
            y: Binary target label series (1 for true match, 0 for non-match).

        Returns:
            EntityMatcher: Trained model instance.
        """
        raise NotImplementedError("EntityMatcher.train stub - to be implemented.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match probability for candidate pairs.

        Args:
            X: Feature matrix DataFrame.

        Returns:
            np.ndarray: Array of match probabilities (range 0.0 to 1.0).
        """
        raise NotImplementedError("EntityMatcher.predict_proba stub - to be implemented.")

    def save_model(self, file_path: Union[str, Path]) -> None:
        """Serialize and save model state to disk.

        Args:
            file_path: Destination output path.
        """
        raise NotImplementedError("EntityMatcher.save_model stub - to be implemented.")

    def load_model(self, file_path: Union[str, Path]) -> "EntityMatcher":
        """Load trained model state from disk.

        Args:
            file_path: Model file path.

        Returns:
            EntityMatcher: Loaded model instance.
        """
        raise NotImplementedError("EntityMatcher.load_model stub - to be implemented.")
