"""
Generate sample normalized stub datasets for testing candidate blocking, feature extraction, and model training.

Runs normalization on a representative sample containing:
1. Obvious exact matches
2. Fuzzy / ambiguous matches
3. No-match / singleton records
4. Duplicate names across sources with different locations
5. Synthetic France-like records
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalize import normalize_entity_dataframe
from src.utils.io_helpers import write_tsv


def main():
    stubs_dir = PROJECT_ROOT / "stubs"
    stubs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Source 1 Raw Sample
    s1_raw = pd.DataFrame([
        # Exact match 1
        {"entity_id": "S1-101", "business_name": "B+ Retail Inc", "business_address": "1712 Montebello Avenue, Phoenix, AZ 85015", "country": "US"},
        # Exact match 2
        {"entity_id": "S1-102", "business_name": "Orelee's Barbershop LLC", "business_address": "1795 Westchester Drive, High Point, NC 27262", "country": "US"},
        # Fuzzy match
        {"entity_id": "S1-103", "business_name": "Prime Money Transfers", "business_address": "17560 Ellis Road, Tahlequah, OK 74464", "country": "US"},
        # Singleton (No-match)
        {"entity_id": "S1-104", "business_name": "Unique Local Bakery Pvt Ltd", "business_address": "797 Lake Town Block A, Kolkata 700089", "country": "India"},
        # Duplicate name record 1 (Phoenix branch)
        {"entity_id": "S1-105", "business_name": "Metro Cafe Co", "business_address": "500 Main Street, Phoenix, AZ 85001", "country": "US"},
        # Synthetic France-like record
        {"entity_id": "S1-106", "business_name": "Boulangerie De Paris SARL", "business_address": "15 Rue de la Paix, Paris 75002", "country": "France"},
    ])

    # 2. Source 2 Raw Sample
    s2_raw = pd.DataFrame([
        # Exact match 1
        {"entity_id": "S2-201", "business_name": "B Plus Retail Incorporated", "business_address": "1712 Montebello Ave, Phoenix, AZ 85015", "country": "United States"},
        # Exact match 2
        {"entity_id": "S2-202", "business_name": "Orelees Barbershop", "business_address": "1795 Westchester Dr, High Point, NC 27262", "country": "USA"},
        # Fuzzy match
        {"entity_id": "S2-203", "business_name": "Prime Money Express", "business_address": "17560 Ellis Rd, Tahlequah, OK 74464", "country": "US"},
        # Duplicate name record 2 (Dallas branch)
        {"entity_id": "S2-205", "business_name": "Metro Cafe", "business_address": "1200 Commerce St, Dallas, TX 75201", "country": "US"},
        # Synthetic France-like record match
        {"entity_id": "S2-206", "business_name": "Boulangerie De Paris", "business_address": "15 Rue de la Paix, Paris 75002", "country": "FR"},
    ])

    # 3. Source 3 Raw Sample
    s3_raw = pd.DataFrame([
        # Exact match 1
        {"entity_id": "S3-301", "business_name": "B+ Retail", "business_address": "1712 Montebello Ave, Phoenix, Arizona 85015", "country": "US"},
        # Exact match 2
        {"entity_id": "S3-302", "business_name": "Orelee Barber Shop", "business_address": "1795 Westchester Drive, High Point, NC", "country": "U.S.A."},
        # Fuzzy match
        {"entity_id": "S3-303", "business_name": "Prime Money Exchange", "business_address": "17560 Ellis Road, Tahlequah, OK", "country": "US"},
        # Synthetic France-like record match
        {"entity_id": "S3-306", "business_name": "La Boulangerie De Paris S.A.R.L.", "business_address": "15 Rue de la Paix, Paris 75002", "country": "Republique Francaise"},
    ])

    # Normalize datasets
    s1_norm = normalize_entity_dataframe(s1_raw)
    s2_norm = normalize_entity_dataframe(s2_raw)
    s3_norm = normalize_entity_dataframe(s3_raw)

    # Required contract columns
    contract_cols = ["entity_id", "name_norm", "address_norm", "street", "city", "state", "postal_code", "country"]

    s1_out = s1_norm[contract_cols]
    s2_out = s2_norm[contract_cols]
    s3_out = s3_norm[contract_cols]

    # Save to stubs directory
    write_tsv(s1_out, stubs_dir / "normalized_source1.tsv")
    write_tsv(s2_out, stubs_dir / "normalized_source2.tsv")
    write_tsv(s3_out, stubs_dir / "normalized_source3.tsv")

    print("[SUCCESS] Successfully generated normalized stub datasets:")
    print(f" - {stubs_dir / 'normalized_source1.tsv'}")
    print(f" - {stubs_dir / 'normalized_source2.tsv'}")
    print(f" - {stubs_dir / 'normalized_source3.tsv'}\n")

    print("=== Sample Normalized Source 1 Output ===")
    print(s1_out.to_string(index=False))


if __name__ == "__main__":
    main()
