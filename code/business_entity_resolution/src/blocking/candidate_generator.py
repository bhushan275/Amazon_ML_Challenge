"""
Stage 2: Blocking & Candidate Generation Module (Compatibility Layer)
====================================================================
Exports the enhanced MultiPassBlocker, compute_blocking_metrics, and related
algorithms from blocking.py for backward compatibility with existing pipelines.
"""

from .blocking import (
    generate_soundex_key,
    generate_metaphone_key,
    CountryAwareBlocker,
    SortedNeighborhoodBlocker,
    MinHashLSHBlocker,
    TfidfCosineBlocker,
    MultiPassBlocker,
    compute_blocking_metrics
)

from .candidate_gen import (
    load_source_tsv,
    load_ground_truth,
    write_candidate_pairs_tsv,
    generate_mock_normalized_data,
    evaluate_blocking_on_split
)

__all__ = [
    "generate_soundex_key",
    "generate_metaphone_key",
    "CountryAwareBlocker",
    "SortedNeighborhoodBlocker",
    "MinHashLSHBlocker",
    "TfidfCosineBlocker",
    "MultiPassBlocker",
    "compute_blocking_metrics",
    "load_source_tsv",
    "load_ground_truth",
    "write_candidate_pairs_tsv",
    "generate_mock_normalized_data",
    "evaluate_blocking_on_split"
]
