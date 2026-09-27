"""
Data splitting utilities for creating entity-disjoint train and validation subsets.
"""

from typing import Dict, Tuple, Union, Optional, Set
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TRAIN_PATH
from src.utils.io_helpers import read_tsv, write_tsv


def make_split(
    ground_truth_path: Optional[Union[str, Path]] = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    save_to_disk: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible train/val split of ground truth entities with stratification.

    Reads train_ground_truth.tsv and stratifies on match/no-match status (has_match),
    returning (train_df, val_df) and writing split_train.tsv and split_val.tsv.

    Args:
        ground_truth_path: Path to train_ground_truth.tsv. Defaults to TRAIN_PATH / "train_ground_truth.tsv".
        val_fraction: Fraction of dataset to assign to validation split (default 0.2).
        seed: Random seed for explicit reproducibility (default 42).
        save_to_disk: If True, writes split_train.tsv and split_val.tsv to ground truth parent directory.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df)
    """
    if ground_truth_path is None:
        gt_file = TRAIN_PATH / "train_ground_truth.tsv"
    else:
        gt_file = Path(ground_truth_path)

    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found at: {gt_file}")

    gt_df = read_tsv(gt_file)

    id_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else gt_df.columns[0]
    matched_col = "matched_entity_ids" if "matched_entity_ids" in gt_df.columns else gt_df.columns[1]

    # Categorical label for stratification: 1 if entity has matches, 0 if no matches
    has_match = gt_df[matched_col].apply(
        lambda val: 1 if pd.notna(val) and str(val).strip() != "" else 0
    )

    if has_match.nunique() > 1:
        train_df, val_df = train_test_split(
            gt_df,
            test_size=val_fraction,
            random_state=seed,
            stratify=has_match
        )
    else:
        train_df, val_df = train_test_split(
            gt_df,
            test_size=val_fraction,
            random_state=seed
        )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if save_to_disk:
        out_dir = gt_file.parent
        write_tsv(train_df, out_dir / "split_train.tsv")
        write_tsv(val_df, out_dir / "split_val.tsv")

    return train_df, val_df


def train_val_split(
    s1_df: pd.DataFrame,
    target_dfs: Dict[str, pd.DataFrame],
    ground_truth_df: pd.DataFrame,
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform entity-disjoint train/validation split on Source 1 entities and ground truth.

    Ensures no overlap between Source 1 entities in training and validation sets to prevent data leakage.

    Args:
        s1_df: Source 1 entity DataFrame.
        target_dfs: Target entities dictionary ('source2', 'source3').
        ground_truth_df: Full ground truth mapping DataFrame.
        val_ratio: Proportion of S1 entities to assign to validation set.
        random_seed: Seed for random generator reproducible partitioning.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (s1_train, s1_val, gt_train, gt_val)
    """
    id_col = "source1_entity_id" if "source1_entity_id" in ground_truth_df.columns else ground_truth_df.columns[0]
    matched_col = "matched_entity_ids" if "matched_entity_ids" in ground_truth_df.columns else ground_truth_df.columns[1]

    has_match = ground_truth_df[matched_col].apply(
        lambda val: 1 if pd.notna(val) and str(val).strip() != "" else 0
    )

    if has_match.nunique() > 1:
        gt_train, gt_val = train_test_split(
            ground_truth_df,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=has_match
        )
    else:
        gt_train, gt_val = train_test_split(
            ground_truth_df,
            test_size=val_ratio,
            random_state=random_seed
        )

    train_ids: Set[str] = set(gt_train[id_col].astype(str))
    val_ids: Set[str] = set(gt_val[id_col].astype(str))

    s1_id_col = "entity_id" if "entity_id" in s1_df.columns else s1_df.columns[0]

    s1_train = s1_df[s1_df[s1_id_col].astype(str).isin(train_ids)].reset_index(drop=True)
    s1_val = s1_df[s1_df[s1_id_col].astype(str).isin(val_ids)].reset_index(drop=True)

    return s1_train, s1_val, gt_train.reset_index(drop=True), gt_val.reset_index(drop=True)
