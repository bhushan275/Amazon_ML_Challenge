#!/usr/bin/env python3
"""
Submission Validator for Amazon ML Challenge 2026 - Business Entity Resolution
Standard library only (no external dependencies required).

Checks matching_results.tsv and candidate_pairs.tsv against format and logical constraints.
Usage:
    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""

import sys
import os
import argparse
from pathlib import Path


def load_valid_ids(test_dir):
    """Load valid S2 and S3 entity IDs from test set files."""
    test_path = Path(test_dir)
    s2_file = test_path / "test_source2.tsv"
    s3_file = test_path / "test_source3.tsv"
    s1_file = test_path / "test_source1.tsv"

    valid_s1_ids = set()
    valid_target_ids = set()

    if not s1_file.exists():
        print(f"[ERROR] Test Source 1 file not found: {s1_file}")
        sys.exit(1)

    with open(s1_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("entity_id") if "entity_id" in header else 0
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                valid_s1_ids.add(parts[id_idx])

    for filepath in [s2_file, s3_file]:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                header = f.readline().strip().split("\t")
                id_idx = header.index("entity_id") if "entity_id" in header else 0
                for line in f:
                    parts = line.strip().split("\t")
                    if parts and parts[0]:
                        valid_target_ids.add(parts[id_idx])

    return valid_s1_ids, valid_target_ids


def parse_tsv(filepath, id_col_name, list_col_name):
    """Parse output TSV file into mapping from S1 ID to list of matched/candidate IDs."""
    if not os.path.exists(filepath):
        return None, [f"File does not exist: {filepath}"]

    errors = []
    data = {}
    seen_s1 = set()

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return None, [f"File is empty: {filepath}"]

    header = lines[0].rstrip("\r\n").split("\t")
    if len(header) != 2:
        errors.append(f"Header must contain exactly 2 tab-separated columns, found {len(header)}: {lines[0].strip()}")
        return None, errors

    if header[0] != id_col_name or header[1] != list_col_name:
        errors.append(f"Header columns must be '{id_col_name}\\t{list_col_name}', got '{header[0]}\\t{header[1]}'")

    for i, line in enumerate(lines[1:], start=2):
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 1:
            errors.append(f"Line {i}: Empty or malformed row")
            continue

        s1_id = parts[0].strip()
        matched_str = parts[1].strip() if len(parts) > 1 else ""

        if not s1_id:
            errors.append(f"Line {i}: Missing source1_entity_id")
            continue

        if s1_id in seen_s1:
            errors.append(f"Line {i}: Duplicate source1_entity_id '{s1_id}'")
        seen_s1.add(s1_id)

        if matched_str:
            matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
        else:
            matched_ids = []

        # Check intra-row duplicates
        if len(matched_ids) != len(set(matched_ids)):
            errors.append(f"Line {i}: Duplicate IDs found in list for '{s1_id}'")

        data[s1_id] = matched_ids

    return data, errors


def validate(matching_file, candidate_file, test_dir):
    all_errors = []
    print(f"=== Validating Submission Files ===")
    print(f"  Matching file : {matching_file}")
    print(f"  Candidate file: {candidate_file}")
    print(f"  Test dir      : {test_dir}\n")

    valid_s1_ids, valid_target_ids = load_valid_ids(test_dir)
    print(f"[INFO] Loaded {len(valid_s1_ids)} S1 test entities and {len(valid_target_ids)} target (S2/S3) test entities.")

    # 1. Parse matching_results.tsv
    matching_data, m_errors = parse_tsv(matching_file, "source1_entity_id", "matched_entity_ids")
    if m_errors:
        all_errors.extend([f"[matching_results.tsv] {e}" for e in m_errors])

    # 2. Parse candidate_pairs.tsv
    candidate_data, c_errors = parse_tsv(candidate_file, "source1_entity_id", "candidate_entity_ids")
    if c_errors:
        all_errors.extend([f"[candidate_pairs.tsv] {e}" for e in c_errors])

    if matching_data is not None:
        # Check all test S1 IDs present
        missing_s1 = valid_s1_ids - set(matching_data.keys())
        extra_s1 = set(matching_data.keys()) - valid_s1_ids

        if missing_s1:
            all_errors.append(f"[matching_results.tsv] Missing {len(missing_s1)} S1 entities from test set (e.g., {list(missing_s1)[:3]})")
        if extra_s1:
            all_errors.append(f"[matching_results.tsv] Contains {len(extra_s1)} unknown S1 entities not in test set (e.g., {list(extra_s1)[:3]})")

        # Check matched entity IDs are valid S2/S3 IDs
        invalid_matches = []
        for s1_id, matches in matching_data.items():
            for m in matches:
                if m not in valid_target_ids:
                    invalid_matches.append((s1_id, m))
                if m.startswith("S1-"):
                    all_errors.append(f"[matching_results.tsv] Entity '{s1_id}' matched with self/S1 ID '{m}'")

        if invalid_matches:
            all_errors.append(f"[matching_results.tsv] Found {len(invalid_matches)} invalid matched target IDs not in test set S2/S3 (e.g., {invalid_matches[:3]})")

    if candidate_data is not None:
        missing_cand_s1 = valid_s1_ids - set(candidate_data.keys())
        if missing_cand_s1:
            all_errors.append(f"[candidate_pairs.tsv] Missing {len(missing_cand_s1)} S1 entities from candidate set")

    # 3. Check Subset Rule: matching IDs MUST be a subset of candidate IDs
    if matching_data is not None and candidate_data is not None:
        non_subset_count = 0
        for s1_id, matches in matching_data.items():
            cands = set(candidate_data.get(s1_id, []))
            not_in_cand = set(matches) - cands
            if not_in_cand:
                non_subset_count += 1
                if non_subset_count <= 5:
                    all_errors.append(f"[Subset Violation] S1 entity '{s1_id}' matches {list(not_in_cand)} which were NOT in candidate_pairs.tsv")

        if non_subset_count > 5:
            all_errors.append(f"[Subset Violation] Total {non_subset_count} entities contain matches outside candidate_pairs.tsv")

    if all_errors:
        print("\n❌ VALIDATION FAILED with the following issues:")
        for idx, err in enumerate(all_errors, start=1):
            print(f"  {idx}. {err}")
        return False
    else:
        print("\n✅ PASS: Submission format and logic validation successful!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate submission TSV files.")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to test dataset directory containing test_source1/2/3.tsv")
    args = parser.parse_args()

    success = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
