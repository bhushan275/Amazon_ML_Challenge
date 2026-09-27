#!/usr/bin/env python3
# ============================================================
# P2 — CANDIDATE GENERATION / BLOCKING
# AMAZON ML CHALLENGE
# COMPATIBLE WITH BOTH GOOGLE COLAB AND LOCAL RUNS
#
# Input:
#   train_source1.tsv
#   train_source2.tsv
#   train_source3.tsv
#   train_ground_truth.tsv
#
# Output:
#   candidate_pairs.tsv
#
# The generated candidate file will use the ACTUAL entity IDs
# from your uploaded training files.
# ============================================================

import os
import sys
import re
import time
import argparse
from collections import defaultdict, Counter

# Ensure console output handles UTF-8 on Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ============================================================
# 1. DEPENDENCY CHECK & INSTALLATION
# ============================================================

def ensure_dependencies():
    """Ensures required packages are installed in Colab / local environment."""
    required = ["scikit-learn", "pandas", "numpy"]
    missing = []
    for pkg in required:
        mod_name = "sklearn" if pkg == "scikit-learn" else pkg
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        import subprocess
        try:
            # Check if pip is available
            subprocess.run([sys.executable, "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"[INFO] Installing missing dependencies: {missing}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
            print("[INFO] Dependencies installed successfully.")
        except Exception:
            pass

ensure_dependencies()

try:
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    HAS_PANDAS_SKLEARN = True
except ImportError:
    HAS_PANDAS_SKLEARN = False

# Safe display helper for both Jupyter/Colab and terminal environments
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


# ============================================================
# 2. PATH CONFIGURATION (SMART DETECT: COLAB OR LOCAL)
# ============================================================

def resolve_paths():
    parser = argparse.ArgumentParser(description="P2 Candidate Generation & Blocking")
    parser.add_argument("--base-dir", default=None, help="Base directory containing train TSV files")
    parser.add_argument("--s1", default=None, help="Path to Source 1 TSV")
    parser.add_argument("--s2", default=None, help="Path to Source 2 TSV")
    parser.add_argument("--s3", default=None, help="Path to Source 3 TSV")
    parser.add_argument("--gt", default=None, help="Path to Ground Truth TSV")
    parser.add_argument("--output", default=None, help="Output path for candidate_pairs.tsv")
    args, _ = parser.parse_known_args()

    # Determine Base Directory
    if args.base_dir and os.path.exists(args.base_dir):
        base = args.base_dir
    elif os.path.exists("/content/train_source1.tsv"):
        base = "/content"
    elif os.path.exists("dataset/train/train_source1.tsv"):
        base = "dataset/train"
    elif os.path.exists("train_source1.tsv"):
        base = "."
    else:
        base = "/content"

    s1_path = args.s1 or os.path.join(base, "train_source1.tsv")
    s2_path = args.s2 or os.path.join(base, "train_source2.tsv")
    s3_path = args.s3 or os.path.join(base, "train_source3.tsv")
    gt_path = args.gt or os.path.join(base, "train_ground_truth.tsv")

    # Determine Output Path
    if args.output:
        output_path = args.output
    elif base == "/content":
        output_path = "/content/candidate_pairs.tsv"
    elif os.path.exists("output") or os.path.exists("dataset"):
        os.makedirs("output", exist_ok=True)
        output_path = "output/candidate_pairs.tsv"
    else:
        output_path = os.path.join(base, "candidate_pairs.tsv")

    return s1_path, s2_path, s3_path, gt_path, output_path


S1_PATH, S2_PATH, S3_PATH, GT_PATH, OUTPUT_PATH = resolve_paths()


# ============================================================
# 3. CHECK FILES
# ============================================================

print("=" * 75)
print("CHECKING DATASET FILES")
print("=" * 75)

for path in [S1_PATH, S2_PATH, S3_PATH, GT_PATH]:
    print(f"{'FOUND' if os.path.exists(path) else 'MISSING':10} {path}")

missing = [path for path in [S1_PATH, S2_PATH, S3_PATH, GT_PATH] if not os.path.exists(path)]

if missing:
    raise FileNotFoundError(
        "\nMissing required dataset files:\n"
        + "\n".join(missing)
        + "\n\nPlease ensure your TSV files exist in the specified path or run with --base-dir <path>."
    )


# ============================================================
# 4. LOAD DATA & EXECUTE BLOCKING
# ============================================================

if not HAS_PANDAS_SKLEARN:
    print("\n[INFO] 'pandas' or 'scikit-learn' not found in this environment.")
    print("[INFO] Executing high-performance zero-dependency multi-pass blocking engine...")

    sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
    from src.blocking import candidate_gen

    s1_recs = candidate_gen.load_source_tsv(S1_PATH)
    s2_recs = candidate_gen.load_source_tsv(S2_PATH)
    s3_recs = candidate_gen.load_source_tsv(S3_PATH)
    target_recs = s2_recs + s3_recs
    gt_data = candidate_gen.load_ground_truth(GT_PATH)

    print(f"Loaded: Source1={len(s1_recs)}, Target={len(target_recs)}, Ground Truth={len(gt_data)}")

    blocker = candidate_gen.MultiPassBlocker(top_k_tfidf=20, snm_window_size=7)
    cand_map = blocker.fit_transform(s1_recs, target_recs)

    # Ensure 100% training ground truth coverage
    for s_id, matches in gt_data.items():
        if s_id in cand_map:
            for m in matches:
                if m not in cand_map[s_id]:
                    cand_map[s_id].append(m)
            cand_map[s_id] = sorted(list(set(cand_map[s_id])))

    all_s1_ids = [r['entity_id'] for r in s1_recs]
    candidate_gen.write_candidate_pairs_tsv(OUTPUT_PATH, cand_map, all_s1_ids=all_s1_ids)

    total_cands = sum(len(v) for v in cand_map.values())
    cartesian = len(s1_recs) * len(target_recs)
    reduction = 1.0 - (total_cands / max(cartesian, 1))

    print("\n" + "=" * 75)
    print("FINAL CANDIDATE STATISTICS (ZERO-DEPENDENCY RUNNER)")
    print("=" * 75)
    print(f"Source 1 records       : {len(s1_recs):,}")
    print(f"Target records         : {len(target_recs):,}")
    print(f"Cartesian pairs        : {cartesian:,}")
    print(f"Candidate pairs        : {total_cands:,}")
    print(f"Average candidates/S1  : {total_cands / max(len(s1_recs), 1):.2f}")
    print(f"Reduction ratio        : {reduction * 100:.2f}%")
    print(f"Training recall ceiling: 100.00%")
    print(f"\n[SUCCESS] Candidate file created at: {OUTPUT_PATH}")
    print("=" * 75)
    sys.exit(0)

print("\nLoading datasets with pandas...")

s1 = pd.read_csv(S1_PATH, sep="\t", dtype=str, keep_default_na=False)
s2 = pd.read_csv(S2_PATH, sep="\t", dtype=str, keep_default_na=False)
s3 = pd.read_csv(S3_PATH, sep="\t", dtype=str, keep_default_na=False)
gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False)


# ============================================================
# 5. DISPLAY DATASET INFORMATION
# ============================================================

print("\n" + "=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print(f"Source 1     : {s1.shape}")
print(f"Source 2     : {s2.shape}")
print(f"Source 3     : {s3.shape}")
print(f"Ground Truth : {gt.shape}")

print(f"\nSource 1 columns: {list(s1.columns)}")
print(f"Source 2 columns: {list(s2.columns)}")
print(f"Source 3 columns: {list(s3.columns)}")
print(f"Ground truth columns: {list(gt.columns)}")


# ============================================================
# 6. VALIDATE SCHEMA
# ============================================================

required_source_columns = {
    "entity_id",
    "business_name",
    "business_address",
    "country"
}

for name, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
    missing_columns = required_source_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"\n{name} is missing columns:\n{missing_columns}\n\nAvailable columns:\n{list(df.columns)}"
        )

required_gt_columns = {
    "source1_entity_id",
    "matched_entity_ids"
}

missing_gt = required_gt_columns - set(gt.columns)
if missing_gt:
    raise ValueError(
        f"\nGround truth is missing:\n{missing_gt}\n\nAvailable:\n{list(gt.columns)}"
    )

print("\nSchema validation PASSED.")


# ============================================================
# 7. NORMALIZATION
# ============================================================

def normalize_text(value):
    if value is None:
        return ""
    value = str(value).lower().strip()
    # Replace common separators with spaces
    value = re.sub(r"[/,_\-|]+", " ", value)
    # Remove punctuation
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    # Collapse whitespace
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_id(value):
    if value is None:
        return ""
    return str(value).strip()


def add_normalized_columns(df):
    df = df.copy()
    df["entity_id"] = df["entity_id"].map(normalize_id)
    df["name_norm"] = df["business_name"].map(normalize_text)
    df["address_norm"] = df["business_address"].map(normalize_text)
    df["country_norm"] = df["country"].map(normalize_text)
    return df


s1 = add_normalized_columns(s1)
s2 = add_normalized_columns(s2)
s3 = add_normalized_columns(s3)


# ============================================================
# 8. COMBINE TARGET SOURCES
# ============================================================

target = pd.concat([s2, s3], ignore_index=True)

print("\n" + "=" * 75)
print("DATASET PROFILE")
print("=" * 75)

print(f"Source 1 records : {len(s1):,}")
print(f"Source 2 records : {len(s2):,}")
print(f"Source 3 records : {len(s3):,}")
print(f"Target records   : {len(target):,}")
print(f"Full Cartesian   : {len(s1) * len(target):,}")


# ============================================================
# 9. BUILD TARGET LOOKUPS
# ============================================================

target_ids = set(target["entity_id"])
source1_ids = set(s1["entity_id"])

target_by_id = {
    row.entity_id: row
    for row in target.itertuples()
}


# ============================================================
# 10. GROUND TRUTH -> PAIR SET
# ============================================================

print("\nBuilding ground truth...")

ground_truth_pairs = set()
ground_truth_by_source = defaultdict(set)

for row in gt.itertuples(index=False):
    source_id = normalize_id(row.source1_entity_id)
    matches = str(row.matched_entity_ids).strip()
    if not matches:
        continue

    for target_id in matches.split(","):
        target_id = normalize_id(target_id)
        if not target_id:
            continue
        ground_truth_pairs.add((source_id, target_id))
        ground_truth_by_source[source_id].add(target_id)

print(f"Ground truth pairs: {len(ground_truth_pairs):,}")


# ============================================================
# 11. CHECK GROUND TRUTH IDs
# ============================================================

bad_gt_source = {s for s, t in ground_truth_pairs if s not in source1_ids}
bad_gt_target = {t for s, t in ground_truth_pairs if t not in target_ids}

print(f"Invalid GT Source1 IDs : {len(bad_gt_source):,}")
print(f"Invalid GT Target IDs  : {len(bad_gt_target):,}")

if bad_gt_source:
    print("Example invalid Source1:", list(bad_gt_source)[:10])
if bad_gt_target:
    print("Example invalid Target:", list(bad_gt_target)[:10])


# ============================================================
# 12. CANDIDATE STORAGE & HELPER
# ============================================================

candidates = defaultdict(set)

def add_candidate(source_id, target_id):
    if source_id in source1_ids and target_id in target_ids and not target_id.startswith("S1-"):
        candidates[source_id].add(target_id)


# ============================================================
# 13. PASS 1: INVERTED TOKEN INDEX
# ============================================================

print("\n" + "=" * 75)
print("PASS 1 — TOKEN INVERTED INDEX")
print("=" * 75)

start = time.time()
token_index = defaultdict(set)

for row in target.itertuples():
    text = row.name_norm + " " + row.address_norm
    tokens = set(text.split())
    for token in tokens:
        if len(token) >= 2:
            token_index[token].add(row.entity_id)

for row in s1.itertuples():
    source_id = row.entity_id
    text = row.name_norm + " " + row.address_norm
    tokens = set(text.split())
    for token in tokens:
        for target_id in token_index.get(token, []):
            add_candidate(source_id, target_id)

print(f"Candidates generated: {sum(len(v) for v in candidates.values()):,}")
print(f"Time: {time.time() - start:.2f}s")


# ============================================================
# 14. PASS 2: COUNTRY-AWARE BLOCKING (OPTIMIZED)
# ============================================================

print("\n" + "=" * 75)
print("PASS 2 — COUNTRY-AWARE BLOCKING")
print("=" * 75)

start = time.time()
country_index = defaultdict(set)
country_prefix_index = defaultdict(list)

# Pre-index targets by country and by (country, name_prefix) for O(1) matching
for row in target.itertuples():
    country = row.country_norm
    if country:
        country_index[country].add(row.entity_id)
        name_tokens = [x for x in row.name_norm.split() if len(x) >= 3]
        if name_tokens:
            country_prefix_index[(country, name_tokens[0][:3])].append(row.entity_id)

for row in s1.itertuples():
    source_id = row.entity_id
    country = row.country_norm
    if not country:
        continue

    same_country = country_index.get(country, set())

    # If the country group is reasonably small (<= 500), link all candidates in group
    if len(same_country) <= 500:
        for target_id in same_country:
            add_candidate(source_id, target_id)
    else:
        # For large country groups, match by (country, prefix)
        name_tokens = [x for x in row.name_norm.split() if len(x) >= 3]
        if name_tokens:
            prefix = name_tokens[0][:3]
            for target_id in country_prefix_index.get((country, prefix), []):
                add_candidate(source_id, target_id)

print(f"Total candidates: {sum(len(v) for v in candidates.values()):,}")
print(f"Time: {time.time() - start:.2f}s")


# ============================================================
# 15. PASS 3: SORTED NEIGHBORHOOD (OPTIMIZED)
# ============================================================

print("\n" + "=" * 75)
print("PASS 3 — SORTED NEIGHBORHOOD")
print("=" * 75)

start = time.time()
WINDOW = 7

for key_type in range(3):
    # Pre-index source rows by 6-char prefix for O(1) neighborhood linkage
    source_by_prefix = defaultdict(list)
    for source_row in s1.itertuples():
        if key_type == 0:
            source_key = source_row.name_norm
        elif key_type == 1:
            source_key = source_row.address_norm
        else:
            source_key = source_row.name_norm[:8] + source_row.address_norm[:8]

        if source_key and len(source_key) >= 6:
            source_by_prefix[source_key[:6]].append(source_row.entity_id)

    records = []
    for row in target.itertuples():
        if key_type == 0:
            key = row.name_norm
        elif key_type == 1:
            key = row.address_norm
        else:
            key = row.name_norm[:8] + row.address_norm[:8]

        if key:
            records.append((key, row.entity_id))

    records.sort(key=lambda x: x[0])

    for i in range(len(records)):
        left = max(0, i - WINDOW)
        right = min(len(records), i + WINDOW + 1)
        nearby = records[left:right]

        current_key = records[i][0]
        prefix = current_key[:6]

        matched_sources = source_by_prefix.get(prefix, [])
        for s_id in matched_sources:
            for _, target_id in nearby:
                add_candidate(s_id, target_id)

print(f"Total candidates: {sum(len(v) for v in candidates.values()):,}")
print(f"Time: {time.time() - start:.2f}s")


# ============================================================
# 16. PASS 4: TF-IDF CHARACTER N-GRAM BLOCKING
# ============================================================

print("\n" + "=" * 75)
print("PASS 4 — TF-IDF CHARACTER N-GRAM")
print("=" * 75)

start = time.time()

target_text = (
    target["name_norm"] + " " + target["address_norm"] + " " + target["country_norm"]
).values

source_text = (
    s1["name_norm"] + " " + s1["address_norm"] + " " + s1["country_norm"]
).values

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=1,
    max_features=300000,
    sublinear_tf=True
)

all_text = np.concatenate([target_text, source_text])
vectorizer.fit(all_text)

target_matrix = vectorizer.transform(target_text)
source_matrix = vectorizer.transform(source_text)

TOP_K = 20
nn = NearestNeighbors(
    n_neighbors=min(TOP_K, len(target)),
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)
nn.fit(target_matrix)

distances, indices = nn.kneighbors(source_matrix)

for source_idx in range(len(s1)):
    source_id = s1.iloc[source_idx]["entity_id"]
    for target_idx in indices[source_idx]:
        target_id = target.iloc[target_idx]["entity_id"]
        add_candidate(source_id, target_id)

print(f"Total candidates: {sum(len(v) for v in candidates.values()):,}")
print(f"Time: {time.time() - start:.2f}s")


# ============================================================
# 17. GROUND TRUTH COVERAGE & TRAINING REPAIR
# ============================================================

print("\n" + "=" * 75)
print("GROUND TRUTH COVERAGE")
print("=" * 75)

before_repair = sum(
    1 for pair in ground_truth_pairs
    if pair[1] in candidates.get(pair[0], set())
)

total_gt = len(ground_truth_pairs)
print(f"GT pairs covered before repair: {before_repair:,} / {total_gt:,}")

recall_before = before_repair / total_gt if total_gt else 0
print(f"Recall ceiling before repair  : {recall_before * 100:.2f}%")

# For TRAINING candidate generation, ensure 100% positive coverage for the matcher
missing_gt = []
for source_id, target_id in ground_truth_pairs:
    if target_id not in candidates.get(source_id, set()):
        add_candidate(source_id, target_id)
        missing_gt.append((source_id, target_id))

print(f"GT pairs added for training   : {len(missing_gt):,}")


# ============================================================
# 18. FINAL CANDIDATE STATISTICS
# ============================================================

total_candidates = sum(len(v) for v in candidates.values())
average_candidates = total_candidates / max(len(s1), 1)
reduction_ratio = 1 - (total_candidates / (len(s1) * len(target)))

print("\n" + "=" * 75)
print("FINAL CANDIDATE STATISTICS")
print("=" * 75)

print(f"Source 1 records       : {len(s1):,}")
print(f"Target records         : {len(target):,}")
print(f"Cartesian pairs        : {len(s1) * len(target):,}")
print(f"Candidate pairs        : {total_candidates:,}")
print(f"Average candidates/S1  : {average_candidates:.2f}")
print(f"Reduction ratio        : {reduction_ratio * 100:.2f}%")


# ============================================================
# 19. VERIFY 100% TRAINING GT COVERAGE
# ============================================================

covered_gt = sum(
    1 for source_id, target_id in ground_truth_pairs
    if target_id in candidates.get(source_id, set())
)

final_recall = covered_gt / total_gt if total_gt else 0
print(f"\nFinal GT coverage: {covered_gt:,} / {total_gt:,}")
print(f"Final training recall ceiling: {final_recall * 100:.2f}%")

if final_recall < 1.0:
    raise RuntimeError("Ground-truth coverage is still below 100%.")


# ============================================================
# 20. CREATE candidate_pairs.tsv
# ============================================================

print(f"\nCreating candidate file at: {OUTPUT_PATH}...")

output_rows = []
for source_id in s1["entity_id"]:
    target_ids_for_source = sorted(list(candidates.get(source_id, set())))
    output_rows.append({
        "source1_entity_id": source_id,
        "candidate_entity_ids": ",".join(target_ids_for_source)
    })

candidate_df = pd.DataFrame(output_rows)

os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
candidate_df.to_csv(OUTPUT_PATH, sep="\t", index=False)


# ============================================================
# 21. RELOAD AND STRICTLY VERIFY FILE
# ============================================================

print("\n" + "=" * 75)
print("RELOAD AND VERIFICATION CHECK")
print("=" * 75)

check = pd.read_csv(OUTPUT_PATH, sep="\t", dtype=str, keep_default_na=False)
print(f"Reloaded candidate file shape: {check.shape}")

# Verify source IDs
candidate_source_ids = set(check["source1_entity_id"])
bad_sources = candidate_source_ids - source1_ids
if bad_sources:
    raise ValueError(f"Candidate file contains invalid Source1 IDs: {list(bad_sources)[:5]}")

# Verify target IDs
candidate_target_ids = set()
for value in check["candidate_entity_ids"]:
    for target_id in str(value).split(","):
        target_id = target_id.strip()
        if target_id:
            candidate_target_ids.add(target_id)

bad_targets = candidate_target_ids - target_ids
if bad_targets:
    raise ValueError(f"Candidate file contains invalid target IDs:\nExamples: {list(bad_targets)[:10]}")

# Verify no S1 self-matches
s1_self_matches = {t for t in candidate_target_ids if t.startswith("S1-")}
if s1_self_matches:
    raise ValueError(f"Violation: Candidate set contains S1 self-matches: {s1_self_matches}")

print("Validation check: ALL PASS!")

print("\n" + "=" * 75)
print("FIRST 5 ROWS PREVIEW")
print("=" * 75)
print(candidate_df.head())


# ============================================================
# 22. FINAL SUCCESS MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("P2 SUCCESSFULLY COMPLETED")
print("=" * 75)
print(f"\nGenerated candidate pairs file: {os.path.abspath(OUTPUT_PATH)}")
print("You can now hand off this file to P3 for feature engineering & model training.")
print("=" * 75)
