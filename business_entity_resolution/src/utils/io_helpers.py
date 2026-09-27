"""
Input/Output helper functions for reading and writing TSV, JSON, and submission files.
"""

from typing import Dict, List, Any, Union
from pathlib import Path
import json
import pandas as pd


def read_tsv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read a tab-delimited TSV file using Pandas with UTF-8 encoding.

    Args:
        path: Path to the TSV file.
        **kwargs: Additional keyword arguments to pass to pd.read_csv.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(path, sep="\t", encoding="utf-8", **kwargs)


def write_tsv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """Write a Pandas DataFrame to a tab-delimited TSV file with UTF-8 encoding and no index.

    Args:
        df: DataFrame to write.
        path: Destination file path.
        **kwargs: Additional keyword arguments to pass to df.to_csv.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, sep="\t", index=False, encoding="utf-8", **kwargs)


# Alias functions for compatibility
load_tsv = read_tsv
save_tsv = write_tsv


def save_submission_tsv(
    results: Dict[str, List[str]],
    file_path: Union[str, Path],
    id_col: str = "source1_entity_id",
    list_col: str = "matched_entity_ids"
) -> None:
    """Save prediction results mapping to submission format TSV.

    Args:
        results: Dictionary mapping Source 1 entity IDs to lists of matched target entity IDs.
        file_path: Destination TSV path.
        id_col: Name of the entity ID column in submission header.
        list_col: Name of comma-separated matched entity IDs list column.
    """
    rows = []
    for s1_id, matched_ids in results.items():
        matched_str = ",".join(matched_ids) if matched_ids else ""
        rows.append({id_col: s1_id, list_col: matched_str})
    df = pd.DataFrame(rows)
    write_tsv(df, file_path)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file into a Python dictionary.

    Args:
        file_path: Path to JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON dictionary content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save a dictionary structure to a JSON file with UTF-8 encoding.

    Args:
        data: Dictionary structure to serialize.
        file_path: Destination JSON output path.
    """
    target_path = Path(file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

