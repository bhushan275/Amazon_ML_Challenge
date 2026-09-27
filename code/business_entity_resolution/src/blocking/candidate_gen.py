"""
Candidate Generation Pipeline & CLI Runner (P2 Ownership)
=========================================================
Loads preprocessed source files, executes multi-pass blocking, measures recall ceiling
and reduction ratio against ground truth, and writes formatted candidate_pairs.tsv.

Interface Contract:
- Input: normalized_source{1,2,3}.tsv (or clean_source{1,2,3}.tsv / raw train/test TSVs)
  Schema: entity_id, name_clean/name_norm, address_clean/address_norm, country
- Output: output/candidate_pairs.tsv
  Schema: source1_entity_id\tcandidate_entity_ids (tab-separated, comma-separated candidate IDs)
"""

import os
import sys
import csv
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Ensure standard output can handle utf-8 on Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Support running both as module and as direct script
try:
    from .blocking import (
        MultiPassBlocker,
        compute_blocking_metrics,
        CountryAwareBlocker,
        SortedNeighborhoodBlocker,
        MinHashLSHBlocker,
        TfidfCosineBlocker,
    )
except ImportError:
    from blocking import (
        MultiPassBlocker,
        compute_blocking_metrics,
        CountryAwareBlocker,
        SortedNeighborhoodBlocker,
        MinHashLSHBlocker,
        TfidfCosineBlocker,
    )


# =====================================================================
# 1. TSV Data Loaders & Output Writers
# =====================================================================

def load_source_tsv(filepath: str) -> List[Dict[str, str]]:
    """
    Loads business entity records from a TSV file.
    Supports normalized column names (name_norm, address_norm) and clean names (name_clean, address_clean)
    as well as raw names (business_name, business_address).
    """
    records = []
    if not os.path.exists(filepath):
        return records

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entity_id = row.get("entity_id", "").strip()
            if not entity_id:
                continue

            # Resolve name column across possible schemas
            name = (
                row.get("name_norm", "")
                or row.get("name_clean", "")
                or row.get("business_name", "")
            ).strip()

            # Resolve address column across possible schemas
            addr = (
                row.get("address_norm", "")
                or row.get("address_clean", "")
                or row.get("business_address", "")
            ).strip()

            country = (row.get("country", "") or "UNKNOWN").strip().upper()

            records.append({
                "entity_id": entity_id,
                "name_clean": name,
                "address_clean": addr,
                "country": country,
                "raw_row": row
            })

    return records


def load_ground_truth(filepath: str) -> Dict[str, List[str]]:
    """Loads ground truth mapping: source1_entity_id -> list of matched IDs."""
    gt = {}
    if not os.path.exists(filepath):
        return gt

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row.get("source1_entity_id", "").strip()
            if not s1_id:
                continue
            matched_str = row.get("matched_entity_ids", "").strip()
            if matched_str:
                matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
            else:
                matched_ids = []
            gt[s1_id] = matched_ids

    return gt


def write_candidate_pairs_tsv(
    output_path: str,
    candidate_map: Dict[str, List[str]],
    all_s1_ids: Optional[List[str]] = None
) -> None:
    """
    Writes candidate_pairs.tsv conforming strictly to challenge rules:
    - Tab-separated header: source1_entity_id\tcandidate_entity_ids
    - Exactly one row per Source 1 entity (even if candidate list is empty)
    - Comma-separated list of candidate entity IDs (no surrounding quotes)
    - Deduplicated candidate IDs
    - S2 and S3 IDs only (no S1 self-matches)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ordered_s1_ids = all_s1_ids if all_s1_ids is not None else sorted(candidate_map.keys())

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for s1_id in ordered_s1_ids:
            cands = candidate_map.get(s1_id, [])
            # Deduplicate preserving order and strip whitespace
            seen = set()
            clean_cands = []
            for c in cands:
                c = c.strip()
                if c and c not in seen and not c.startswith("S1-"):
                    seen.add(c)
                    clean_cands.append(c)

            cand_str = ",".join(clean_cands) if clean_cands else ""
            writer.writerow([s1_id, cand_str])


# =====================================================================
# 2. Day 0 Mock Dataset Generator (for Independent Parallel Dev)
# =====================================================================

def generate_mock_normalized_data(output_dir: str) -> None:
    """
    Generates a 15-record mock dataset matching P1's normalized schema
    so P2, P3, and P4 can test end-to-end immediately without upstream waiting.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Source 1 (Reference)
    s1_rows = [
        ("S1-00001", "amazon web services inc", "410 terry ave n seattle wa 98109", "US"),
        ("S1-00002", "tata consultancy services pvt_ltd", "birlati street nariman point mumbai", "INDIA"),
        ("S1-00003", "infosys limited", "electronics city hosur rd bangalore karnataka", "INDIA"),
        ("S1-00004", "alphabet google llc", "1600 amphitheatre pkwy mountain view ca", "US"),
        ("S1-00005", "microsoft corporation", "one microsoft way redmond wa 98052", "US"),
        ("S1-00006", "carrefour sa", "93 avenue de paris massy cedex", "FRANCE"),
        ("S1-00007", "reliance industries pvt_ltd", "maker chambers iv nariman point mumbai", "INDIA"),
        ("S1-00008", "lone star bakery llc", "124 main st dallas tx 75201", "US"),  # Singleton
        ("S1-00009", "totalenergies se", "2 place jean millier la defense paris", "FRANCE"),
        ("S1-00010", "wipro enterprises pvt_ltd", "sarjapur road bangalore karnataka", "INDIA"),
    ]

    # Source 2 (Target 1)
    s2_rows = [
        ("S2-00101", "amazon web services aws", "410 terry avenue north seattle", "US"),  # Matches S1-00001
        ("S2-00102", "tcs ltd", "birlati road nariman point mumbai maharashtra", "INDIA"),  # Matches S1-00002
        ("S2-00103", "infosys tech", "hosur rd electronics city bangalore", "INDIA"),  # Matches S1-00003
        ("S2-00104", "google inc", "1600 amphitheatre parkway mountain view", "US"),  # Matches S1-00004
        ("S2-00105", "microsoft corp", "1 microsoft way redmond", "US"),  # Matches S1-00005
        ("S2-00106", "carrefour hypermarket", "avenue de paris 93 massy", "FRANCE"),  # Matches S1-00006
        ("S2-00107", "random unlinked cafe", "55 5th avenue new york ny", "US"),
        ("S2-00108", "total energies france", "la defense 2 place jean millier", "FRANCE"),  # Matches S1-00009
    ]

    # Source 3 (Target 2)
    s3_rows = [
        ("S3-00201", "aws cloud computing", "terry ave seattle washington", "US"),  # Matches S1-00001
        ("S3-00202", "tata consult svcs", "nariman pt mumbai", "INDIA"),  # Matches S1-00002
        ("S3-00203", "reliance retail ventures", "maker chamber nariman pt mumbai", "INDIA"),  # Matches S1-00007
        ("S3-00204", "wipro tech solutions", "sarjapur road bengaluru", "INDIA"),  # Matches S1-00010
        ("S3-00205", "google cloud platform", "amphitheatre pkwy mountain view", "US"),  # Matches S1-00004
        ("S3-00206", "stand-alone bistro", "rue de la paix paris", "FRANCE"),
    ]

    # Ground truth mapping
    gt_rows = [
        ("S1-00001", "S2-00101,S3-00201"),
        ("S1-00002", "S2-00102,S3-00202"),
        ("S1-00003", "S2-00103"),
        ("S1-00004", "S2-00104,S3-00205"),
        ("S1-00005", "S2-00105"),
        ("S1-00006", "S2-00106"),
        ("S1-00007", "S3-00203"),
        ("S1-00008", ""),  # Singleton
        ("S1-00009", "S2-00108"),
        ("S1-00010", "S3-00204"),
    ]

    def _write_tsv(filepath, header, rows):
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(header)
            w.writerows(rows)

    _write_tsv(out_dir / "normalized_source1.tsv", ["entity_id", "name_norm", "address_norm", "country"], s1_rows)
    _write_tsv(out_dir / "normalized_source2.tsv", ["entity_id", "name_norm", "address_norm", "country"], s2_rows)
    _write_tsv(out_dir / "normalized_source3.tsv", ["entity_id", "name_norm", "address_norm", "country"], s3_rows)
    _write_tsv(out_dir / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gt_rows)


# =====================================================================
# 3. Validation Runner & Benchmark Scorer
# =====================================================================

def evaluate_blocking_on_split(
    s1_records: List[dict],
    target_records: List[dict],
    ground_truth: Dict[str, List[str]],
    val_ratio: float = 0.2,
    seed: int = 42,
    top_k_tfidf: int = 20,
    snm_window: int = 7
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """
    Evaluates blocking strategy on a validation split.
    Reports Recall Ceiling and Reduction Ratio.
    """
    random.seed(seed)
    all_s1 = list(s1_records)
    random.shuffle(all_s1)

    val_count = max(1, int(len(all_s1) * val_ratio))
    val_s1 = all_s1[:val_count]
    val_s1_ids = {r['entity_id'] for r in val_s1}
    val_gt = {k: v for k, v in ground_truth.items() if k in val_s1_ids}

    blocker = MultiPassBlocker(
        top_k_tfidf=top_k_tfidf,
        snm_window_size=snm_window,
        enable_lsh=True,
        enable_snm=True,
        enable_country_keys=True,
        enable_tfidf=True
    )

    val_candidates = blocker.fit_transform(val_s1, target_records)
    metrics = compute_blocking_metrics(val_candidates, val_gt, len(target_records))

    return metrics, val_candidates


# =====================================================================
# 4. Command Line Interface
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Candidate Generation & Multi-Pass Blocking (P2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--s1", help="Path to Source 1 TSV file")
    parser.add_argument("--s2", help="Path to Source 2 TSV file")
    parser.add_argument("--s3", help="Path to Source 3 TSV file")
    parser.add_argument("--data-dir", help="Path to directory containing source1/2/3 TSV files")
    parser.add_argument("--ground-truth", help="Path to ground truth TSV file (for evaluation)")
    parser.add_argument("--output", default="output/candidate_pairs.tsv", help="Output path for candidate_pairs.tsv")
    parser.add_argument("--top-k-tfidf", type=int, default=20, help="Top K neighbors for TF-IDF Cosine pass")
    parser.add_argument("--snm-window", type=int, default=7, help="Window size W for Sorted Neighborhood Method")
    parser.add_argument("--max-candidates", type=int, default=100, help="Max candidates per S1 entity")
    parser.add_argument("--eval-val-split", action="store_true", help="Run validation split benchmark")
    parser.add_argument("--generate-mock", action="store_true", help="Generate mock dataset in mock_data/ and exit")
    return parser.parse_args()


def resolve_file(base_dir: Optional[str], specific_path: Optional[str], candidates: List[str]) -> Optional[str]:
    """Helper to locate a file either from a specific path or within a base directory."""
    if specific_path and os.path.exists(specific_path):
        return specific_path
    if base_dir and os.path.isdir(base_dir):
        for fname in candidates:
            p = os.path.join(base_dir, fname)
            if os.path.exists(p):
                return p
    return None


def main():
    args = parse_args()

    # Mode 1: Generate Mock Data for Day 0
    if args.generate_mock:
        mock_dir = "mock_data"
        print(f"Generating mock normalized dataset in '{mock_dir}'...")
        generate_mock_normalized_data(mock_dir)
        print("Done. Mock files generated: normalized_source1.tsv, normalized_source2.tsv, normalized_source3.tsv, train_ground_truth.tsv")
        return

    # Mode 2: Locate Source Files
    s1_path = resolve_file(args.data_dir, args.s1, ["normalized_source1.tsv", "clean_source1.tsv", "train_source1.tsv", "test_source1.tsv"])
    s2_path = resolve_file(args.data_dir, args.s2, ["normalized_source2.tsv", "clean_source2.tsv", "train_source2.tsv", "test_source2.tsv"])
    s3_path = resolve_file(args.data_dir, args.s3, ["normalized_source3.tsv", "clean_source3.tsv", "train_source3.tsv", "test_source3.tsv"])
    gt_path = resolve_file(args.data_dir, args.ground_truth, ["train_ground_truth.tsv"])

    if not s1_path or not s2_path:
        print("[WARNING] Source files not found. Creating temporary mock dataset to execute demonstration...")
        mock_dir = "mock_data"
        generate_mock_normalized_data(mock_dir)
        s1_path = os.path.join(mock_dir, "normalized_source1.tsv")
        s2_path = os.path.join(mock_dir, "normalized_source2.tsv")
        s3_path = os.path.join(mock_dir, "normalized_source3.tsv")
        gt_path = os.path.join(mock_dir, "train_ground_truth.tsv")

    print("=" * 70)
    print("  STAGE 2: BLOCKING & CANDIDATE GENERATION (P2 LEAD)")
    print("=" * 70)
    print(f"  Source 1 : {s1_path}")
    print(f"  Source 2 : {s2_path}")
    print(f"  Source 3 : {s3_path}")
    if gt_path:
        print(f"  Ground Truth: {gt_path}")
    print(f"  Output TSV  : {args.output}")
    print("-" * 70)

    s1_records = load_source_tsv(s1_path)
    s2_records = load_source_tsv(s2_path)
    s3_records = load_source_tsv(s3_path) if s3_path else []
    target_records = s2_records + s3_records

    print(f"[INFO] Loaded {len(s1_records)} S1 entities and {len(target_records)} target entities (S2: {len(s2_records)}, S3: {len(s3_records)})")

    # Mode 3: Validation Split Benchmark
    if gt_path and (args.eval_val_split or not os.path.basename(s1_path).startswith("test")):
        gt_data = load_ground_truth(gt_path)
        if gt_data:
            print("\n[BENCHMARK] Evaluating Multi-Pass Blocker on Validation Split...")
            metrics, _ = evaluate_blocking_on_split(
                s1_records=s1_records,
                target_records=target_records,
                ground_truth=gt_data,
                val_ratio=0.2,
                top_k_tfidf=args.top_k_tfidf,
                snm_window=args.snm_window
            )
            print(f"  * Validation Recall Ceiling : {metrics['recall_ceiling'] * 100:.2f}% ({metrics['retained_true_pairs']}/{metrics['total_true_pairs']} true pairs)")
            print(f"  * Validation Reduction Ratio: {metrics['reduction_ratio'] * 100:.4f}%")
            print(f"  * Avg Candidates per S1     : {metrics['avg_candidates_per_entity']:.1f}")
            print(f"  * Zero-Candidate Singletons : {metrics['zero_candidate_entities']}")
            print("-" * 70)

    # Mode 4: Full Candidate Generation
    print("\n[BLOCKING] Running Full Multi-Pass Candidate Generation...")
    blocker = MultiPassBlocker(
        top_k_tfidf=args.top_k_tfidf,
        snm_window_size=args.snm_window,
        max_candidates_per_entity=args.max_candidates,
        enable_lsh=True,
        enable_snm=True,
        enable_country_keys=True,
        enable_tfidf=True
    )

    candidate_map = blocker.fit_transform(s1_records, target_records)
    all_s1_ids = [r['entity_id'] for r in s1_records]

    # Save candidate_pairs.tsv
    write_candidate_pairs_tsv(args.output, candidate_map, all_s1_ids=all_s1_ids)

    # Compute overall statistics
    total_cands = sum(len(c) for c in candidate_map.values())
    avg_cands = total_cands / max(len(s1_records), 1)
    cartesian = len(s1_records) * len(target_records)
    reduction = 1.0 - (total_cands / max(cartesian, 1))

    print(f"[SUCCESS] Generated candidates for {len(s1_records)} S1 entities.")
    print(f"  * Total Candidate Pairs : {total_cands:,} (Cartesian product: {cartesian:,})")
    print(f"  * Overall Reduction Ratio: {reduction * 100:.4f}%")
    print(f"  * Avg Candidates per S1 : {avg_cands:.2f}")
    print(f"  * Saved File            : {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
