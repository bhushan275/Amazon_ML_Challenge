"""
Unit tests for data normalization and preprocessing pipeline (src/data/normalize.py).
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalize import (
    normalize_name,
    normalize_address,
    normalize_country,
    normalize_record,
    normalize_entity_dataframe,
    SUFFIXES
)

EXPECTED_CONTRACT_COLUMNS = [
    "entity_id",
    "name_norm",
    "address_norm",
    "street",
    "city",
    "state",
    "postal_code",
    "country"
]


def test_names_case_punctuation_normalize_identically():
    """(a) Test names differing only by case/punctuation normalize identically."""
    name1 = "Acme Retail Inc."
    name2 = "ACME RETAIL, INC"
    name3 = "acme retail inc"
    name4 = "A.C.M.E. R.E.T.A.I.L., Inc."

    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    norm3 = normalize_name(name3)
    norm4 = normalize_name(name4)

    assert norm1 == norm2 == norm3 == norm4 == "acme retail"


def test_legal_suffixes_stripped_consistently():
    """(b) Test legal suffixes are stripped consistently."""
    assert normalize_name("Prime Money Transfers LLC") == "prime money transfers"
    assert normalize_name("Global Services Pvt Ltd") == "global"
    assert normalize_name("Boulangerie SARL") == "boulangerie"
    assert normalize_name("B Plus Retail Corporation") == "b plus retail"
    assert normalize_name("Tech Enterprise Co. Ltd.") == "tech"


def test_address_abbreviations_expand_correctly():
    """(c) Test address abbreviations expand correctly."""
    addr = "1712 Montebello Ave., Phoenix, AZ 85015"
    norm_info = normalize_address(addr)

    assert "avenue" in norm_info["street"]
    assert norm_info["postal_code"] == "85015"

    addr2 = "100 Main St, Ste 4B, Bldg 2"
    norm_info2 = normalize_address(addr2)
    assert "street" in norm_info2["street"]
    assert "suite" in norm_info2["street"]
    assert "building" in norm_info2["street"]


def test_france_like_record_normalizes_without_crashing():
    """(d) Test France-like record normalizes without crashing or defaulting to garbage."""
    raw_record = {
        "entity_id": "FR-999",
        "business_name": "Boulangerie De Paris S.A.R.L.",
        "business_address": "15 Rue de la Paix, Paris 75002",
        "country": "Republique Francaise"
    }
    rec = normalize_record(raw_record)

    assert rec["entity_id"] == "FR-999"
    assert rec["name_norm"] == "boulangerie de paris"
    assert rec["country"] == "FR"
    assert rec["postal_code"] == "75002"
    assert "paris" in rec["city"].lower()
    assert rec["address_norm"] != ""


def test_output_schema_matches_contracts_md():
    """(e) Test output schema matches CONTRACTS.md column names exactly."""
    raw_df = pd.DataFrame([
        {
            "entity_id": "E1",
            "business_name": "Acme Inc",
            "business_address": "123 Main St, NY 10001",
            "country": "US"
        }
    ])
    norm_df = normalize_entity_dataframe(raw_df)
    cols = list(norm_df.columns)

    for expected_col in EXPECTED_CONTRACT_COLUMNS:
        assert expected_col in cols, f"Missing required contract column: {expected_col}"

    assert list(norm_df[EXPECTED_CONTRACT_COLUMNS].columns) == EXPECTED_CONTRACT_COLUMNS
