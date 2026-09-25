# Business Entity Resolution Pipeline - Amazon ML Challenge 2026

## Executive Summary
This repository implements an end-to-end Machine Learning solution for cross-source Business Entity Resolution (ER). Given business records from 3 independent data sources with noisy, inconsistent, and abbreviated fields, the pipeline identifies which records across sources refer to the same real-world business entity.

The solution optimizes the competition evaluation metric (**Macro F_0.5 Score**), prioritizing high precision while maintaining high recall ceilings via multi-pass candidate blocking.

---

## Directory Layout
```
code/business_entity_resolution/
├── CONTRACTS.md               # Inter-stage file interface schemas
├── README.md                  # Detailed run and reproduction instructions
├── requirements.txt           # Pinned python dependencies
└── src/
    ├── preprocessing/         # Stage 1: Unicode/legal suffix/address cleaning
    │   ├── normalizer.py
    │   └── __init__.py
    ├── blocking/              # Stage 2: Multi-pass fuzzy candidate generation
    │   ├── candidate_generator.py
    │   └── __init__.py
    ├── matching/              # Stage 3: Feature engineering & GBDT classifier
    │   ├── feature_extractor.py
    │   ├── model.py
    │   └── __init__.py
    └── evaluation/            # Stage 4: Threshold tuning & macro F_0.5 evaluation
        ├── evaluator.py
        ├── pipeline.py
        └── __init__.py
```

---

## Quick Start & Reproduction

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Benchmark Data (Optional)
If running on local benchmark data:
```bash
python scripts/generate_synthetic_data.py
```

### 3. Run Pipeline End-to-End
```bash
python scripts/run_pipeline.py
```

### 4. Validate Final Submission Outputs
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## Pipeline Architecture Details

### Stage 1: Preprocessing & Normalization (Person 1)
- Strips accents (Unicode NFKD normalization) and lowercases text.
- Standardizes legal entity suffixes (e.g., `Corporation` -> `corp`, `Private Limited` -> `pvt_ltd`).
- Expands street address abbreviations (`st` -> `street`, `rd` -> `road`) while protecting landmark phrases (`near`, `opposite`).
- Preserves open-set country representation without hardcoding to `{US, India}`.

### Stage 2: Blocking & Candidate Generation (Person 2)
- Combines Token Inverted Indexing, Phonetic (Soundex) matching, First-word + Country keying, and TF-IDF Cosine Nearest Neighbors.
- Generates candidate pair set (`candidate_pairs.tsv`) ensuring high recall ceiling while reducing pair comparison search space by > 99%.

### Stage 3: Feature Engineering & Matching Model (Person 3)
- Extracts pairwise similarity features: Levenshtein ratio, Jaro-Winkler, Token Jaccard, Char 3-gram Jaccard, TF-IDF Cosine similarity, Country match, First-token match, Numeric token overlap, and Length ratios.
- Fits LightGBM / Gradient Boosting classifier (MIT/Apache 2.0 compliant, < 8B parameters).

### Stage 4: Evaluation, Threshold Tuning & Submission (Person 4)
- Tunes acceptance probability threshold `t*` on held-out validation split to maximize macro F_0.5 score.
- Handles singletons cleanly (unmatched entities emit empty string in TSV).
- Exports formatted `matching_results.tsv` and `candidate_pairs.tsv`.
