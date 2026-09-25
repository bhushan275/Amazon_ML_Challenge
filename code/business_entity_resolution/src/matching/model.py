"""
Stage 3: Matching Model Classifier Module
Trains a gradient-boosted binary classifier on pair similarity features to output entity match probabilities.
Supports LightGBM if available, fallback to sklearn GradientBoostingClassifier / ExtraTrees.
MIT / Apache 2.0 license compliant, < 8B parameters.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class EntityMatchingModel:
    def __init__(self, use_lightgbm: bool = True):
        self.use_lightgbm = use_lightgbm and HAS_LIGHTGBM
        if self.use_lightgbm:
            self.model = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )

    def fit(self, X: list, y: list):
        """Fits binary classification model on feature vectors X and binary labels y (1=match, 0=non-match)."""
        X_arr = np.array(X)
        y_arr = np.array(y)
        self.model.fit(X_arr, y_arr)

    def predict_proba(self, X: list) -> np.ndarray:
        """Returns array of match probabilities for given feature vectors X."""
        if len(X) == 0:
            return np.array([])
        X_arr = np.array(X)
        probs = self.model.predict_proba(X_arr)
        return probs[:, 1]
