# Architecture & Inter-Stage Interface Contracts

## Overview
The Business Entity Resolution pipeline is designed as a modular 4-stage system. Each stage has strict input and output file contracts, allowing parallel development, independent unit testing, and reproducible end-to-end execution.

```
[Raw S1, S2, S3 TSV Files]
       │
       ▼ (Stage 1: Preprocessing & Normalization)
[clean_source1.tsv, clean_source2.tsv, clean_source3.tsv]
       │
       ▼ (Stage 2: Blocking & Candidate Generation)
[candidate_pairs.tsv]
       │
       ▼ (Stage 3: Feature Engineering & Matching Model)
[Scored Pairs / Pair Probabilities]
       │
       ▼ (Stage 4: Threshold Calibration & Evaluation)
[matching_results.tsv & Submission Zip]
```

---

## Stage 1: Preprocessing & Normalization
- **Owner**: Person 1
- **Inputs**: `dataset/{train,test}/source1/2/3.tsv`
- **Output Files**: `output/clean_source1.tsv`, `output/clean_source2.tsv`, `output/clean_source3.tsv`
- **Schema**:
  - `entity_id` (string): Unique identifier (e.g., `S1-00001`, `S2-00047`, `S3-00812`)
  - `name_clean` (string): Normalized business name (lowercased, legal suffix dictionary mapping, unicode stripped)
  - `address_clean` (string): Normalized address (street abbreviation mapping, landmark preserved)
  - `tokens_name` (string): Space-separated n-grams/tokens of name
  - `country` (string): Preserved open-set country string label

---

## Stage 2: Blocking & Candidate Generation
- **Owner**: Person 2
- **Inputs**: `clean_source1.tsv`, `clean_source2.tsv`, `clean_source3.tsv`
- **Output File**: `output/candidate_pairs.tsv`
- **Schema**:
  - `source1_entity_id` (string): S1 entity identifier
  - `candidate_entity_ids` (string): Comma-separated list of candidate S2/S3 entity IDs (empty string for singletons)
- **Constraints**:
  - Tab-separated file (`\t`).
  - Exactly one row per S1 entity.
  - Candidate IDs must contain ONLY S2 or S3 IDs present in the target files.
  - No duplicate entity IDs within a list.

---

## Stage 3: Feature Engineering & Matching Model
- **Owner**: Person 3
- **Inputs**: `clean_source1/2/3.tsv`, `candidate_pairs.tsv`, `train_ground_truth.tsv` (for training)
- **Output**: Matrix of candidate pair similarity features and predicted match probabilities (`pair_score`).
- **Feature Schema**:
  - String metrics: Levenshtein distance/ratio, Jaro-Winkler, Token Jaccard, Token Cosine, LCS ratio.
  - Vector similarities: TF-IDF Cosine similarity (name, address, combined).
  - Categorical & Structure: Country match (binary), First token match, Numeric token overlap ratio, Length ratio.

---

## Stage 4: Threshold Calibration & Submission Packaging
- **Owner**: Person 4
- **Inputs**: Scored candidate pairs from Stage 3, `candidate_pairs.tsv` from Stage 2.
- **Output Files**:
  - `output/matching_results.tsv` (final matches scored on leaderboard)
  - `output/candidate_pairs.tsv` (blocking output)
  - `<team_name>_submission.zip`
- **Constraints**:
  - Final matched IDs in `matching_results.tsv` MUST be a subset of candidate IDs in `candidate_pairs.tsv`.
  - Calibrated for **Macro F_0.5 Score** (penalizes false merges 2× more than false negatives).
  - Validated using `utils/validate_submission.py`.
