"""
Dataset loader functions for reading multi-source entity resolution datasets.
"""

from typing import Dict, Tuple, Union, Optional
from pathlib import Path
import pandas as pd

from src.config import TRAIN_PATH, TEST_PATH, RAW_SOURCE_FILES, TEST_RAW_SOURCE_FILES
from src.utils.io_helpers import read_tsv

REQUIRED_COLUMNS = {"business_name", "business_address", "country"}


def load_source(
    source_name: str,
    split: str = "train",
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Load raw dataset for source1, source2, or source3 using io_helpers.read_tsv and validate schema.

    Args:
        source_name: Name of source dataset ('source1', 'source2', or 'source3').
        split: Dataset split ('train' or 'test'). Defaults to 'train'.
        data_dir: Optional path override for dataset directory.

    Returns:
        pd.DataFrame: DataFrame containing entity records.

    Raises:
        ValueError: If source_name is unknown or required columns are missing.
        FileNotFoundError: If the source file does not exist.
    """
    normalized_source = source_name.lower().strip()
    if normalized_source not in ("source1", "source2", "source3"):
        raise ValueError(
            f"Invalid source_name '{source_name}'. Expected one of: 'source1', 'source2', 'source3'."
        )

    if data_dir is not None:
        file_path = Path(data_dir) / f"{split}_{normalized_source}.tsv"
    else:
        split_dir = TRAIN_PATH if split == "train" else TEST_PATH
        filename_dict = RAW_SOURCE_FILES if split == "train" else TEST_RAW_SOURCE_FILES
        filename = filename_dict.get(normalized_source, f"{split}_{normalized_source}.tsv")
        file_path = split_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Source data file not found at: {file_path}")

    df = read_tsv(file_path)

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Source dataset '{source_name}' at {file_path} is missing required columns: {sorted(missing_cols)}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def load_raw_source_data(
    data_dir: Optional[Union[str, Path]] = None,
    split_name: str = "train"
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load raw entity datasets for source 1, source 2, and source 3.

    Args:
        data_dir: Optional path to data directory.
        split_name: Name of dataset partition ('train' or 'test').

    Returns:
        Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
            - s1_df: DataFrame containing query Source 1 entities.
            - target_dfs: Dictionary mapping source names ('source2', 'source3') to target DataFrames.
    """
    s1_df = load_source("source1", split=split_name, data_dir=data_dir)
    s2_df = load_source("source2", split=split_name, data_dir=data_dir)
    s3_df = load_source("source3", split=split_name, data_dir=data_dir)
    target_dfs = {"source2": s2_df, "source3": s3_df}
    return s1_df, target_dfs


def load_ground_truth(
    file_path: Optional[Union[str, Path]] = None,
    split_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Load ground truth mappings for training/validation sets.

    Args:
        file_path: Explicit path to train_ground_truth.tsv.
        split_dir: Optional path to dataset directory containing ground truth file.

    Returns:
        pd.DataFrame: DataFrame containing ground truth entity linkages.
    """
    if file_path is None:
        base_dir = Path(split_dir) if split_dir is not None else TRAIN_PATH
        file_path = base_dir / RAW_SOURCE_FILES.get("ground_truth", "train_ground_truth.tsv")
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Ground truth file not found at: {file_path}")

    return read_tsv(file_path)

