"""
Execution script for running full normalization pipeline over raw dataset files:
loaders -> normalize -> write to output/normalized_source{1,2,3}.tsv.
"""

import sys
import time
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_PATH
from src.data.loaders import load_source
from src.data.normalize import normalize_entity_dataframe
from src.utils.io_helpers import write_tsv

CONTRACT_COLUMNS = [
    "entity_id",
    "name_norm",
    "address_norm",
    "street",
    "city",
    "state",
    "postal_code",
    "country"
]

DATASET_RAW_DIR = Path(r"C:\Users\ASUS\Desktop\coding\Projects_rs\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource\student_resource\dataset\train")


def process_source(source_name: str) -> dict:
    print(f"\n================ Processing {source_name.upper()} ================")
    start_time = time.time()

    # 1. Load raw data using loaders module
    raw_df = load_source(source_name, split="train", data_dir=DATASET_RAW_DIR)
    row_count_in = len(raw_df)
    print(f"[{source_name}] Loaded {row_count_in:,} raw rows in {time.time() - start_time:.2f}s")

    # 2. Run normalization
    norm_start = time.time()
    norm_df = normalize_entity_dataframe(raw_df)
    norm_time = time.time() - norm_start
    print(f"[{source_name}] Normalized records in {norm_time:.2f}s")

    # 3. Select contract schema columns
    final_df = norm_df[CONTRACT_COLUMNS]
    row_count_out = len(final_df)

    # 4. Check & flag failed rows
    failed_names = int((final_df["name_norm"].str.strip() == "").sum())
    failed_addresses = int((final_df["address_norm"].str.strip() == "").sum())
    failed_ids = int((final_df["entity_id"].str.strip() == "").sum())

    total_flagged = failed_names + failed_ids

    # 5. Write to output TSV
    out_file = OUTPUT_PATH / f"normalized_{source_name}.tsv"
    write_start = time.time()
    write_tsv(final_df, out_file)
    write_time = time.time() - write_start
    print(f"[{source_name}] Saved to {out_file} in {write_time:.2f}s")

    elapsed = time.time() - start_time
    print(f"[{source_name}] Rows IN: {row_count_in:,} | Rows OUT: {row_count_out:,}")
    print(f"[{source_name}] Flagged Rows (Empty Name/ID): {total_flagged:,} (Empty Names: {failed_names:,}, Empty Addresses: {failed_addresses:,})")
    print(f"[{source_name}] Total Elapsed: {elapsed:.2f}s")

    return {
        "source": source_name,
        "rows_in": row_count_in,
        "rows_out": row_count_out,
        "flagged_failed": total_flagged,
        "empty_names": failed_names,
        "empty_addresses": failed_addresses,
        "output_path": str(out_file),
    }


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    results = []

    for source in ["source1", "source2", "source3"]:
        res = process_source(source)
        results.append(res)

    print("\n" + "=" * 65)
    print("      FULL DATASET NORMALIZATION PIPELINE SUMMARY")
    print("=" * 65)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
