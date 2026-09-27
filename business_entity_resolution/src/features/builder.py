"""
Feature engineering routines for computing pairwise similarity metrics across entity pairs.
"""

from typing import Dict, List, Optional
import pandas as pd


class FeatureBuilder:
    """Extracts string similarity, token overlap, numerical, and structural features for candidate entity pairs."""

    def __init__(self, metric_names: Optional[List[str]] = None):
        """Initialize feature builder with desired similarity metrics.

        Args:
            metric_names: List of metrics to compute (e.g. ['fuzzy_ratio', 'jaro_winkler', 'token_sort_ratio']).
        """
        self.metric_names = metric_names or ["fuzzy_ratio", "token_sort_ratio", "jaccard_similarity"]

    def build_features(
        self,
        candidate_pairs: pd.DataFrame,
        s1_df: pd.DataFrame,
        target_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Construct feature vectors for candidate entity pairs.

        Args:
            candidate_pairs: DataFrame containing candidate pairs ('s1_id', 'target_id', 'target_source').
            s1_df: Source 1 entity DataFrame.
            target_dfs: Target entities dictionary ('source2', 'source3').

        Returns:
            pd.DataFrame: DataFrame with candidate pair metadata and computed feature columns.
        """
        raise NotImplementedError("FeatureBuilder.build_features stub - to be implemented.")


def extract_pair_features(
    candidate_pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    target_dfs: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Functional interface for pair feature extraction.

    Args:
        candidate_pairs: Pair candidate DataFrame.
        s1_df: Source 1 entity DataFrame.
        target_dfs: Target entities dictionary.

    Returns:
        pd.DataFrame: Extracted feature matrix for pair classification.
    """
    raise NotImplementedError("extract_pair_features stub - to be implemented.")
