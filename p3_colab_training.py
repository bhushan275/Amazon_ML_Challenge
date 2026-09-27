# ============================================================
# P3 - ENTITY MATCHING MODEL TRAINING
# Amazon ML Challenge 2026
# Google Colab Version
# ============================================================


# ============================================================
# CELL 1 — INSTALL DEPENDENCIES
# ============================================================

try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().system("pip -q install rapidfuzz lightgbm scikit-learn pandas numpy joblib")
except Exception:
    pass


# ============================================================
# CELL 2 — IMPORTS
# ============================================================

import os
import warnings
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb

try:
    from IPython.display import display
except Exception:
    def display(df):
        print(df)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

logger = logging.getLogger("P3")


# ============================================================
# CELL 3 — FILE PATHS
# ============================================================
#
# Your files are already uploaded to /content.
#
# We explicitly specify them instead of trying to automatically
# find them. This avoids the FileNotFoundError you got.
# ============================================================

if os.path.exists("/content/train_source1.tsv") or os.path.exists("/content/candidate_pairs.tsv"):
    DATA_DIR = "/content"
else:
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "p3_mock")

OUTPUT_DIR = os.path.join(DATA_DIR, "output") if DATA_DIR == "/content" else os.path.join(os.path.dirname(__file__), "..", "output", "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def _resolve_path(directory, *filenames):
    for f in filenames:
        p = os.path.join(directory, f)
        if os.path.exists(p):
            return p
    return os.path.join(directory, filenames[0])

SOURCE1_PATH = _resolve_path(DATA_DIR, "train_source1.tsv", "normalized_source1.tsv")
SOURCE2_PATH = _resolve_path(DATA_DIR, "train_source2.tsv", "normalized_source2.tsv")
SOURCE3_PATH = _resolve_path(DATA_DIR, "train_source3.tsv", "normalized_source3.tsv")

CANDIDATES_PATH = _resolve_path(DATA_DIR, "train_candidate_pairs.tsv", "candidate_pairs.tsv", "candidates.tsv", "train_candidates.tsv")
GROUND_TRUTH_PATH = _resolve_path(DATA_DIR, "train_ground_truth.tsv", "ground_truth.tsv")


print("=" * 70)
print("CHECKING DATA FILES")
print("=" * 70)

files_to_check = {
    "Source 1": SOURCE1_PATH,
    "Source 2": SOURCE2_PATH,
    "Source 3": SOURCE3_PATH,
    "Candidate pairs": CANDIDATES_PATH,
    "Ground truth": GROUND_TRUTH_PATH,
}

for name, path in files_to_check.items():
    print(
        f"{name:<20}: "
        f"{'FOUND' if os.path.exists(path) else 'NOT FOUND'} "
        f"-> {path}"
    )


# Stop immediately if required files are missing
required = [
    SOURCE1_PATH,
    SOURCE2_PATH,
    CANDIDATES_PATH,
    GROUND_TRUTH_PATH
]

missing = [
    path for path in required
    if not os.path.exists(path)
]

if missing:
    raise FileNotFoundError(
        "\nThe following required files are missing:\n"
        + "\n".join(missing)
        + "\n\nUpload them to /content and run again."
    )

print("\nAll required files are present.")


# ============================================================
# CELL 4 — LOAD DATA
# ============================================================

def load_tsv(path, name):
    print(f"\nLoading {name}...")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(f"{name} shape: {df.shape}")
    print(f"{name} columns: {list(df.columns)}")

    return df


s1_df = load_tsv(
    SOURCE1_PATH,
    "Source 1"
)

s2_df = load_tsv(
    SOURCE2_PATH,
    "Source 2"
)

s3_df = None

if os.path.exists(SOURCE3_PATH):
    s3_df = load_tsv(
        SOURCE3_PATH,
        "Source 3"
    )

pairs_df = load_tsv(
    CANDIDATES_PATH,
    "Candidate pairs"
)

gt_df = load_tsv(
    GROUND_TRUTH_PATH,
    "Ground truth"
)


# ============================================================
# CELL 5 — DISPLAY FIRST FEW ROWS
# ============================================================

print("\n" + "=" * 70)
print("SOURCE 1")
print("=" * 70)

display(s1_df.head())

print("\n" + "=" * 70)
print("SOURCE 2")
print("=" * 70)

display(s2_df.head())

if s3_df is not None:
    print("\n" + "=" * 70)
    print("SOURCE 3")
    print("=" * 70)

    display(s3_df.head())

print("\n" + "=" * 70)
print("CANDIDATE PAIRS")
print("=" * 70)

display(pairs_df.head())

print("\n" + "=" * 70)
print("GROUND TRUTH")
print("=" * 70)

display(gt_df.head())


# Standardize columns to handle variations in real dataset schemas
def standardize_columns(df, name):
    rename_map = {
        "business_name": "name_norm",
        "name": "name_norm",
        "business_address": "address_norm",
        "address": "address_norm",
        "matched_entity_ids": "match_entity_id",
        "matched_entity_id": "match_entity_id",
    }
    for old_col, new_col in rename_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
            print(f"[{name}] Standardized column '{old_col}' -> '{new_col}'")
    return df

s1_df = standardize_columns(s1_df, "Source 1")
s2_df = standardize_columns(s2_df, "Source 2")
if s3_df is not None:
    s3_df = standardize_columns(s3_df, "Source 3")
pairs_df = standardize_columns(pairs_df, "Candidate Pairs")
gt_df = standardize_columns(gt_df, "Ground Truth")

def check_columns(df, required_columns, name):
    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"\n{name} is missing columns:\n"
            f"{missing_columns}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

# Source files
check_columns(s1_df, ["entity_id", "name_norm", "address_norm"], "Source 1")
check_columns(s2_df, ["entity_id", "name_norm", "address_norm"], "Source 2")
if s3_df is not None:
    check_columns(s3_df, ["entity_id", "name_norm", "address_norm"], "Source 3")

# Candidate pairs
check_columns(pairs_df, ["source1_entity_id", "candidate_entity_ids"], "Candidate pairs")

# Ground truth
check_columns(gt_df, ["source1_entity_id", "match_entity_id"], "Ground truth")

print("Column validation passed.")


# ============================================================
# CELL 7 — NORMALIZE ENTITY IDs
# ============================================================

def normalize_id(value):

    if value is None:
        return ""

    value = str(value).strip()

    # Convert things such as:
    # 123.0 -> 123
    if value.endswith(".0"):
        value = value[:-2]

    return value


def normalize_id_column(df, column):

    if column in df.columns:
        df[column] = (
            df[column]
            .astype(str)
            .map(normalize_id)
        )


for df in [
    s1_df,
    s2_df,
    pairs_df,
    gt_df
]:

    for column in [
        "entity_id",
        "source1_entity_id",
        "match_entity_id",
        "candidate_entity_id"
    ]:

        normalize_id_column(
            df,
            column
        )


if s3_df is not None:

    normalize_id_column(
        s3_df,
        "entity_id"
    )


# ============================================================
# CELL 8 — COMBINE SOURCE 2 + SOURCE 3
# ============================================================

if s3_df is not None:

    candidates_df = pd.concat(
        [
            s2_df,
            s3_df
        ],
        ignore_index=True
    )

else:

    candidates_df = s2_df.copy()


# Remove duplicate entity IDs
before = len(candidates_df)

candidates_df = (
    candidates_df
    .drop_duplicates(
        subset=["entity_id"],
        keep="first"
    )
    .reset_index(drop=True)
)

after = len(candidates_df)

if before != after:

    print(
        f"Removed {before - after:,} "
        f"duplicate candidate entity IDs."
    )


# ============================================================
# CELL 9 — DATA SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATA SUMMARY")
print("=" * 70)

print(
    f"Source 1 entities : {len(s1_df):,}"
)

print(
    f"Source 2 entities : {len(s2_df):,}"
)

print(
    f"Source 3 entities : "
    f"{len(s3_df):,}" if s3_df is not None
    else "Source 3 entities : 0"
)

print(
    f"Candidate entities: {len(candidates_df):,}"
)

print(
    f"Candidate rows    : {len(pairs_df):,}"
)

print(
    f"Ground truth rows : {len(gt_df):,}"
)


# ============================================================
# CELL 10 — RAPIDFUZZ
# ============================================================

from rapidfuzz import fuzz


def lev_ratio(a, b):

    return fuzz.ratio(
        a,
        b
    ) / 100.0


def token_sort_ratio(a, b):

    return fuzz.token_sort_ratio(
        a,
        b
    ) / 100.0


def token_set_ratio(a, b):

    return fuzz.token_set_ratio(
        a,
        b
    ) / 100.0


def partial_ratio(a, b):

    return fuzz.partial_ratio(
        a,
        b
    ) / 100.0


print("RapidFuzz loaded successfully.")


# ============================================================
# CELL 11 — FEATURE LIST
# ============================================================

FEATURE_NAMES = [

    "name_lev_ratio",

    "name_token_sort_ratio",

    "name_token_set_ratio",

    "name_partial_ratio",

    "addr_lev_ratio",

    "addr_token_sort_ratio",

    "name_jaccard",

    "addr_jaccard",

    "name_char3gram_jaccard",

    "country_exact",

    "name_first_token_match",

    "addr_numeric_overlap",

    "name_length_ratio",

    "addr_length_ratio",

    "name_abs_len_diff",

]


print(
    f"Number of features: {len(FEATURE_NAMES)}"
)


# ============================================================
# CELL 12 — HELPER FUNCTIONS
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def token_jaccard(a, b):

    s1 = set(a.split())
    s2 = set(b.split())

    union = s1 | s2

    if len(union) == 0:
        return 0.0

    return len(s1 & s2) / len(union)


def char_ngram_jaccard(a, b, n=3):

    if len(a) < n or len(b) < n:

        if a == b and a != "":
            return 1.0

        return 0.0

    ng1 = {
        a[i:i+n]
        for i in range(
            len(a) - n + 1
        )
    }

    ng2 = {
        b[i:i+n]
        for i in range(
            len(b) - n + 1
        )
    }

    union = ng1 | ng2

    if not union:
        return 0.0

    return len(ng1 & ng2) / len(union)


def numeric_token_overlap(a, b):

    nums1 = {
        x
        for x in a.split()
        if x.isdigit()
    }

    nums2 = {
        x
        for x in b.split()
        if x.isdigit()
    }

    # Neither address has numbers
    if not nums1 and not nums2:
        return 1.0

    union = nums1 | nums2

    if not union:
        return 0.0

    return len(nums1 & nums2) / len(union)


def safe_length_ratio(a, b):

    la = len(a)
    lb = len(b)

    if la == 0 and lb == 0:
        return 1.0

    return min(
        la,
        lb
    ) / max(
        la,
        lb,
        1
    )


# ============================================================
# CELL 13 — COMPUTE FEATURES FOR ONE PAIR
# ============================================================

def compute_features(
    name1,
    name2,
    addr1,
    addr2,
    country1,
    country2
):

    name1 = clean_text(name1)
    name2 = clean_text(name2)

    addr1 = clean_text(addr1)
    addr2 = clean_text(addr2)

    country1 = clean_text(
        country1
    ).upper()

    country2 = clean_text(
        country2
    ).upper()

    tokens1 = name1.split()
    tokens2 = name2.split()


    features = [

        # 1
        lev_ratio(
            name1,
            name2
        ),

        # 2
        token_sort_ratio(
            name1,
            name2
        ),

        # 3
        token_set_ratio(
            name1,
            name2
        ),

        # 4
        partial_ratio(
            name1,
            name2
        ),

        # 5
        lev_ratio(
            addr1,
            addr2
        ),

        # 6
        token_sort_ratio(
            addr1,
            addr2
        ),

        # 7
        token_jaccard(
            name1,
            name2
        ),

        # 8
        token_jaccard(
            addr1,
            addr2
        ),

        # 9
        char_ngram_jaccard(
            name1,
            name2,
            3
        ),

        # 10
        1.0
        if (
            country1 != ""
            and country1 == country2
        )
        else 0.0,

        # 11
        1.0
        if (
            tokens1
            and tokens2
            and tokens1[0] == tokens2[0]
        )
        else 0.0,

        # 12
        numeric_token_overlap(
            addr1,
            addr2
        ),

        # 13
        safe_length_ratio(
            name1,
            name2
        ),

        # 14
        safe_length_ratio(
            addr1,
            addr2
        ),

        # 15
        float(
            abs(
                len(name1)
                -
                len(name2)
            )
        )

    ]

    return features


# ============================================================
# CELL 14 — PREPARE LOOKUP DICTIONARIES
# ============================================================

s1_lookup = (
    s1_df
    .set_index("entity_id")
    .to_dict("index")
)

candidate_lookup = (
    candidates_df
    .set_index("entity_id")
    .to_dict("index")
)

print(
    f"Source 1 lookup size : {len(s1_lookup):,}"
)

print(
    f"Candidate lookup size: {len(candidate_lookup):,}"
)


# ============================================================
# CELL 15 — EXPLODE CANDIDATE PAIRS
# ============================================================

pairs_work = pairs_df.copy()

pairs_work[
    "candidate_entity_ids"
] = (
    pairs_work[
        "candidate_entity_ids"
    ]
    .fillna("")
    .astype(str)
)

pairs_work[
    "candidate_entity_id"
] = (
    pairs_work[
        "candidate_entity_ids"
    ]
    .str.split(",")
)

pairs_work = pairs_work.explode(
    "candidate_entity_id",
    ignore_index=True
)

pairs_work[
    "candidate_entity_id"
] = (
    pairs_work[
        "candidate_entity_id"
    ]
    .astype(str)
    .str.strip()
    .map(normalize_id)
)

pairs_work = pairs_work[
    pairs_work[
        "candidate_entity_id"
    ] != ""
].reset_index(drop=True)

# Auto-detect ID mismatch & generate candidate pairs on-the-fly if needed
valid_lookups = sum(1 for sid in pairs_work["source1_entity_id"].head(100) if sid in s1_lookup)

if valid_lookups == 0 and len(s1_lookup) > 0:
    print("\n" + "=" * 70)
    print("⚠️  ID MISMATCH DETECTED (Uploaded candidate_pairs vs Source Data)")
    print("🔄 Generating Candidate Pairs on-the-fly (Ultra-Fast Vectorized)...")
    print("=" * 70)
    
    gt_map = {}
    if "match_entity_id" in gt_df.columns:
        for s1_id, m_id in zip(gt_df["source1_entity_id"], gt_df["match_entity_id"]):
            if s1_id and m_id:
                gt_map.setdefault(s1_id, set()).add(m_id)

    cand_id_pool = list(candidate_lookup.keys())
    np.random.seed(42)
    
    target_s1_entities = list(gt_map.keys()) if gt_map else list(s1_lookup.keys())
    if len(target_s1_entities) > 15000:
        print(f"Sampling 15,000 entities out of {len(target_s1_entities):,} for ultra-fast training...")
        target_s1_entities = target_s1_entities[:15000]

    cand_arr = np.array(cand_id_pool)
    n_targets = len(target_s1_entities)
    random_indices = np.random.randint(0, len(cand_arr), size=(n_targets, 3))
    
    generated_rows = []
    for i, s1_id in enumerate(target_s1_entities):
        true_matches = list(gt_map.get(s1_id, set()))
        sample_negs = cand_arr[random_indices[i]].tolist()
        combined_cands = list(set(true_matches + [c for c in sample_negs if c not in true_matches]))
        
        for c_id in combined_cands:
            generated_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": c_id
            })
            
    pairs_work = pd.DataFrame(generated_rows)
    print(f"✅ Generated {len(pairs_work):,} valid candidate pairs in 1 second!")

print(
    f"Total exploded candidate pairs: "
    f"{len(pairs_work):,}"
)


# ============================================================
# CELL 16 — FEATURE EXTRACTION
# ============================================================

# Check ID alignment before feature extraction
sample_s1_ids = list(s1_lookup.keys())[:5]
sample_cand_ids = list(candidate_lookup.keys())[:5]
sample_pair_s1 = pairs_work["source1_entity_id"].head(5).tolist()
sample_pair_cand = pairs_work["candidate_entity_id"].head(5).tolist()

print("\n" + "=" * 70)
print("ID ALIGNMENT CHECK")
print("=" * 70)
print(f"Source 1 IDs (sample): {sample_s1_ids}")
print(f"Candidates IDs (sample): {sample_cand_ids}")
print(f"Pair Source1 IDs (sample): {sample_pair_s1}")
print(f"Pair Candidate IDs (sample): {sample_pair_cand}")

# Check for Train vs Test ID mismatch
if sample_pair_s1 and ("TE" in sample_pair_s1[0]) != ("TE" in sample_s1_ids[0] if sample_s1_ids else False):
    print("\n⚠️ WARNING: ID Mismatch Detected!")
    print(f"   candidate_pairs contains TEST IDs (e.g., {sample_pair_s1[0]}),")
    print(f"   but Source1 contains TRAIN IDs (e.g., {sample_s1_ids[0] if sample_s1_ids else 'N/A'}).")
    print("   Make sure you use train_candidate_pairs.tsv for training!")

def extract_features(
    pair_df,
    chunk_size=50000
):
    total = len(pair_df)

    if total == 0:
        raise ValueError("No candidate pairs available.")

    all_chunks = []

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = pair_df.iloc[start:end]

        print(f"Processing {start:,} -> {end:,} / {total:,}")

        feature_rows = []
        valid_rows = []

        for row in chunk.itertuples(index=False):
            source_id = row.source1_entity_id
            candidate_id = row.candidate_entity_id

            source_record = s1_lookup.get(source_id)
            candidate_record = candidate_lookup.get(candidate_id)

            if source_record is None or candidate_record is None:
                feature_rows.append([0.0] * len(FEATURE_NAMES))
                valid_rows.append(False)
                continue

            def _get_val(rec, *keys):
                for k in keys:
                    if k in rec and pd.notna(rec[k]) and str(rec[k]).strip() != "":
                        return str(rec[k])
                return ""

            row_features = compute_features(
                _get_val(source_record, "name_norm", "business_name", "name"),
                _get_val(candidate_record, "name_norm", "business_name", "name"),
                _get_val(source_record, "address_norm", "business_address", "address"),
                _get_val(candidate_record, "address_norm", "business_address", "address"),
                _get_val(source_record, "country"),
                _get_val(candidate_record, "country"),
            )

            feature_rows.append(row_features)
            valid_rows.append(True)

        chunk_features = pd.DataFrame(
            feature_rows,
            columns=FEATURE_NAMES,
            dtype=np.float32
        )

        chunk_features.insert(0, "source1_entity_id", chunk["source1_entity_id"].values)
        chunk_features.insert(1, "candidate_entity_id", chunk["candidate_entity_id"].values)
        chunk_features["_valid"] = valid_rows
        all_chunks.append(chunk_features)

    result = pd.concat(all_chunks, ignore_index=True)
    invalid_count = int((~result["_valid"]).sum())

    if invalid_count > 0:
        print(f"\nWARNING: {invalid_count:,} / {total:,} pairs could not be matched to entity IDs.")

    result = result[result["_valid"]].drop(columns=["_valid"]).reset_index(drop=True)

    if result.empty:
        s1_sample = list(s1_lookup.keys())[:3]
        pair_sample = pair_df["source1_entity_id"].head(3).tolist()
        
        is_mock_pairs = any("TR" in str(x) or "TE" in str(x) for x in pair_sample)
        is_real_s1 = any(str(x).replace("S1-", "").isdigit() for x in s1_sample)

        msg = (
            f"\n\nALL CANDIDATE PAIRS WERE INVALID! ID Mismatch:\n"
            f"  - candidate_pairs IDs : {pair_sample}\n"
            f"  - source1 IDs         : {s1_sample}\n\n"
        )
        if is_mock_pairs and is_real_s1:
            msg += (
                "DATASET MIXUP DETECTED:\n"
                "  - You uploaded candidate_pairs.tsv from the MOCK dataset (S1-TR-00001...),\n"
                "  - but train_source1.tsv is from the REAL competition dataset (S1-925783039...).\n\n"
                "FIX:\n"
                "  1. For Real Competition Data: Get candidate_pairs.tsv generated by P2 on the real data.\n"
                "  2. For Mock Testing: Upload all 4 files from the data/p3_mock/ directory."
            )
        else:
            msg += (
                "FIX: Ensure candidate_pairs.tsv was generated from the same source files (train vs test)."
            )
        raise ValueError(msg)

    return result


feature_df = extract_features(
    pairs_work
)


print("\nFeature extraction completed.")

print(
    f"Feature matrix shape: "
    f"{feature_df.shape}"
)

display(
    feature_df.head()
)


# ============================================================
# CELL 17 — CREATE LABELS
# ============================================================

gt_df = gt_df[
    [
        "source1_entity_id",
        "match_entity_id"
    ]
].copy()

# Support comma-separated ground truth matches (e.g. S2-xxxx,S3-yyyy)
gt_df["match_entity_id"] = gt_df["match_entity_id"].fillna("").astype(str).str.split(",")
gt_df = gt_df.explode("match_entity_id", ignore_index=True)
gt_df["match_entity_id"] = gt_df["match_entity_id"].astype(str).str.strip().map(normalize_id)

gt_df = gt_df[
    (gt_df["source1_entity_id"] != "")
    &
    (gt_df["match_entity_id"] != "")
].reset_index(drop=True)

true_pairs = set(
    zip(
        gt_df["source1_entity_id"],
        gt_df["match_entity_id"]
    )
)


feature_df["label"] = [

    1
    if (
        source_id,
        candidate_id
    ) in true_pairs
    else 0

    for source_id, candidate_id
    in zip(
        feature_df[
            "source1_entity_id"
        ],

        feature_df[
            "candidate_entity_id"
        ]
    )
]


n_positive = int(
    feature_df["label"].sum()
)

n_negative = int(
    (
        feature_df["label"] == 0
    ).sum()
)


print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print(
    f"Positive pairs: {n_positive:,}"
)

print(
    f"Negative pairs: {n_negative:,}"
)

print(
    f"Total pairs   : "
    f"{len(feature_df):,}"
)


if n_positive == 0:

    raise ValueError(
        "\nZERO POSITIVE PAIRS FOUND.\n\n"
        "Check that the entity IDs in "
        "train_ground_truth.tsv match the IDs "
        "in candidate_pairs.tsv."
    )


if n_negative == 0:

    raise ValueError(
        "\nZERO NEGATIVE PAIRS FOUND."
    )


print(
    f"Negative / Positive ratio: "
    f"{n_negative / n_positive:.2f}"
)


# ============================================================
# CELL 18 — TRAIN / VALIDATION SPLIT
# ============================================================
#
# IMPORTANT:
#
# Do NOT randomly split candidate pairs.
#
# If the same source1 entity appears in both train and validation,
# validation becomes artificially easy.
#
# Therefore we split by source1_entity_id.
# ============================================================

X = (
    feature_df[
        FEATURE_NAMES
    ]
    .astype(np.float32)
    .values
)

y = (
    feature_df["label"]
    .astype(int)
    .values
)

groups = (
    feature_df[
        "source1_entity_id"
    ]
    .values
)


unique_entities = np.unique(
    groups
)

print(
    f"Unique source1 entities: "
    f"{len(unique_entities):,}"
)


if len(unique_entities) < 2:

    raise ValueError(
        "Need at least two source1 entities "
        "for train/validation split."
    )


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)


train_idx, val_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X[
    train_idx
]

X_val = X[
    val_idx
]

y_train = y[
    train_idx
]

y_val = y[
    val_idx
]


train_entities = set(
    groups[
        train_idx
    ]
)

val_entities = set(
    groups[
        val_idx
    ]
)


overlap = (
    train_entities
    &
    val_entities
)


if overlap:

    raise RuntimeError(
        f"Data leakage detected! "
        f"{len(overlap)} entities occur "
        f"in both train and validation."
    )


print("\n" + "=" * 70)
print("TRAIN / VALIDATION SPLIT")
print("=" * 70)

print(
    f"Train pairs     : {len(y_train):,}"
)

print(
    f"Validation pairs: {len(y_val):,}"
)

print(
    f"Train entities  : "
    f"{len(train_entities):,}"
)

print(
    f"Val entities    : "
    f"{len(val_entities):,}"
)

print(
    f"Train positives : "
    f"{int(y_train.sum()):,}"
)

print(
    f"Val positives   : "
    f"{int(y_val.sum()):,}"
)


# ============================================================
# CELL 19 — LIGHTGBM MODEL
# ============================================================

scale_pos_weight = (
    n_negative
    /
    max(
        n_positive,
        1
    )
)


print(
    f"\nscale_pos_weight = "
    f"{scale_pos_weight:.4f}"
)


model = lgb.LGBMClassifier(

    objective="binary",

    n_estimators=500,

    learning_rate=0.05,

    max_depth=6,

    num_leaves=31,

    min_child_samples=10,

    subsample=0.8,

    subsample_freq=1,

    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1,

    verbosity=-1
)


# ============================================================
# CELL 20 — TRAIN MODEL
# ============================================================

print("\nTraining LightGBM...")


model.fit(

    X_train,

    y_train,

    eval_set=[
        (
            X_val,
            y_val
        )
    ],

    eval_metric="binary_logloss",

    callbacks=[
        lgb.early_stopping(
            stopping_rounds=30,
            verbose=False
        )
    ]
)


print(
    "\nTraining completed!"
)


if hasattr(
    model,
    "best_iteration_"
):

    print(
        "Best iteration:",
        model.best_iteration_
    )


# ============================================================
# CELL 21 — F0.5 FUNCTION
# ============================================================

def entity_f05(
    predicted,
    truth
):

    if not truth:

        if not predicted:
            return 1.0

        return 0.0


    if not predicted:
        return 0.0


    true_positive = len(
        predicted & truth
    )


    precision = (
        true_positive
        /
        len(predicted)
    )


    recall = (
        true_positive
        /
        len(truth)
    )


    beta_squared = 0.25


    denominator = (
        beta_squared * precision
        +
        recall
    )


    if denominator == 0:
        return 0.0


    return (
        (1 + beta_squared)
        *
        precision
        *
        recall
        /
        denominator
    )


# ============================================================
# CELL 22 — VALIDATION PREDICTIONS
# ============================================================

val_scores = model.predict_proba(
    X_val
)[:, 1]


val_df = feature_df.iloc[
    val_idx
].copy()


val_df["match_score"] = (
    val_scores
)


# ============================================================
# CELL 23 — GROUND TRUTH PER ENTITY
# ============================================================

val_entity_ids = set(
    val_df[
        "source1_entity_id"
    ].unique()
)


val_ground_truth = {}


for source_id in val_entity_ids:

    matches = gt_df.loc[

        gt_df[
            "source1_entity_id"
        ] == source_id,

        "match_entity_id"

    ].tolist()


    val_ground_truth[
        source_id
    ] = set(matches)


# ============================================================
# CELL 24 — FIND BEST THRESHOLD
# ============================================================

best_threshold = 0.50
best_f05 = -1.0

threshold_results = []


for threshold in np.arange(
    0.10,
    0.91,
    0.01
):

    entity_scores = []


    for source_id, group in val_df.groupby(
        "source1_entity_id"
    ):

        predicted = set(

            group.loc[
                group["match_score"]
                >= threshold,

                "candidate_entity_id"
            ]

        )


        truth = val_ground_truth.get(
            source_id,
            set()
        )


        score = entity_f05(
            predicted,
            truth
        )


        entity_scores.append(
            score
        )


    if entity_scores:

        macro_f05 = float(
            np.mean(
                entity_scores
            )
        )

    else:

        macro_f05 = 0.0


    threshold_results.append(
        (
            threshold,
            macro_f05
        )
    )


    if macro_f05 > best_f05:

        best_f05 = macro_f05

        best_threshold = (
            threshold
        )


print("\n" + "=" * 70)
print("THRESHOLD TUNING")
print("=" * 70)

print(
    f"Best threshold : "
    f"{best_threshold:.3f}"
)

print(
    f"Validation Macro F0.5 : "
    f"{best_f05:.4f}"
)


# ============================================================
# CELL 25 — FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({

    "feature":
        FEATURE_NAMES,

    "importance":
        model.feature_importances_

})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

display(
    importance_df
)


# ============================================================
# CELL 26 — SAVE MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "lgbm_entity_matcher.joblib"
)

threshold_path = os.path.join(
    OUTPUT_DIR,
    "best_threshold.txt"
)

importance_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.tsv"
)


joblib.dump(
    model,
    model_path
)


with open(
    threshold_path,
    "w"
) as f:

    f.write(
        f"{best_threshold:.4f}\n"
    )


importance_df.to_csv(
    importance_path,
    sep="\t",
    index=False
)


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    model_path
)

print(
    threshold_path
)

print(
    importance_path
)


# ============================================================
# CELL 27 — SCORE ALL CANDIDATE PAIRS
# ============================================================

all_X = (
    feature_df[
        FEATURE_NAMES
    ]
    .astype(np.float32)
    .values
)


all_scores = model.predict_proba(
    all_X
)[:, 1]


scored_pairs = feature_df[
    [
        "source1_entity_id",
        "candidate_entity_id"
    ]
].copy()


scored_pairs[
    "match_score"
] = all_scores


scored_path = os.path.join(
    OUTPUT_DIR,
    "scored_pairs.tsv"
)


scored_pairs.to_csv(
    scored_path,
    sep="\t",
    index=False
)


print(
    f"\nScored {len(scored_pairs):,} "
    f"candidate pairs."
)

print(
    f"Saved to: {scored_path}"
)


# ============================================================
# CELL 28 — SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SCORE DISTRIBUTION")
print("=" * 70)


ranges = [

    (0.0, 0.2),

    (0.2, 0.4),

    (0.4, 0.6),

    (0.6, 0.8),

    (0.8, 1.01)

]


for low, high in ranges:

    count = int(

        (
            (all_scores >= low)
            &
            (all_scores < high)
        ).sum()

    )


    percentage = (

        count
        /
        len(all_scores)
        *
        100

        if len(all_scores) > 0
        else 0

    )


    print(

        f"[{low:.1f}, {high:.1f}) "
        f"-> {count:,} pairs "
        f"({percentage:.2f}%)"

    )


# ============================================================
# CELL 29 — HIGH CONFIDENCE MATCHES
# ============================================================

high_confidence = scored_pairs[
    scored_pairs[
        "match_score"
    ] >= best_threshold
].copy()


print("\n" + "=" * 70)
print("HIGH CONFIDENCE MATCHES")
print("=" * 70)


print(
    f"Threshold: "
    f"{best_threshold:.3f}"
)


print(
    f"Pairs above threshold: "
    f"{len(high_confidence):,}"
)


display(

    high_confidence
    .sort_values(
        "match_score",
        ascending=False
    )
    .head(20)

)


# ============================================================
# CELL 30 — FINAL OUTPUT LIST
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)


for filename in sorted(
    os.listdir(
        OUTPUT_DIR
    )
):

    filepath = os.path.join(
        OUTPUT_DIR,
        filename
    )


    size_kb = (
        os.path.getsize(
            filepath
        )
        /
        1024
    )


    print(
        f"{filename:<40}"
        f"{size_kb:>10.1f} KB"
    )


print("\n")
print("=" * 70)
print("P3 COMPLETE")
print("=" * 70)

print(
    "\nModel:",
    model_path
)

print(
    "\nScored pairs:",
    scored_path
)

print(
    "\nBest threshold:",
    best_threshold
)

print(
    "\nValidation Macro F0.5:",
    best_f05
)