"""
End-to-End Execution Pipeline Manager
Orchestrates Stage 1 (Preprocessing) -> Stage 2 (Blocking) -> Stage 3 (Matching) -> Stage 4 (Evaluation & Packaging).
"""

import os
import csv
import random
from pathlib import Path

from ..preprocessing import preprocess_record
from ..blocking import MultiPassBlocker, compute_blocking_metrics
from ..matching import FeatureExtractor, EntityMatchingModel
from .evaluator import compute_macro_f05, tune_threshold


def load_tsv(filepath: str) -> list:
    """Loads records from a tab-separated TSV file."""
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append(dict(row))
    return records


def load_ground_truth(filepath: str) -> dict:
    """Loads ground truth mapping s1_id -> list of matched_ids."""
    gt = {}
    if not os.path.exists(filepath):
        return gt
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row['source1_entity_id']
            m_str = row.get('matched_entity_ids', '').strip()
            gt[s1_id] = [m.strip() for m in m_str.split(',') if m.strip()] if m_str else []
    return gt


def save_output_tsv(filepath: str, data_map: dict, id_col: str, list_col: str):
    """Saves output mapping s1_id -> list of entity IDs to TSV per spec."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([id_col, list_col])
        for s1_id in sorted(data_map.keys()):
            id_list = data_map[s1_id]
            list_str = ",".join(id_list) if id_list else ""
            writer.writerow([s1_id, list_str])


class EntityResolutionPipeline:
    def __init__(self, top_k_tfidf: int = 15):
        self.blocker = MultiPassBlocker(top_k_tfidf=top_k_tfidf)
        self.feature_extractor = FeatureExtractor()
        self.model = EntityMatchingModel()
        self.best_threshold = 0.5

    def train_and_validate(self, train_dir: str, val_split_ratio: float = 0.2):
        """
        Runs Stage 1-4 on training data with held-out validation split.
        """
        print("=== STAGE 1: Preprocessing Training Data ===")
        s1_raw = load_tsv(os.path.join(train_dir, "train_source1.tsv"))
        s2_raw = load_tsv(os.path.join(train_dir, "train_source2.tsv"))
        s3_raw = load_tsv(os.path.join(train_dir, "train_source3.tsv"))
        gt_full = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

        s1_clean = [preprocess_record(r) for r in s1_raw]
        s2_clean = [preprocess_record(r) for r in s2_raw]
        s3_clean = [preprocess_record(r) for r in s3_raw]
        target_clean = s2_clean + s3_clean
        target_dict = {r['entity_id']: r for r in target_clean}

        # Train / Validation split
        random.seed(42)
        all_s1_ids = [r['entity_id'] for r in s1_clean]
        random.shuffle(all_s1_ids)

        val_size = int(len(all_s1_ids) * val_split_ratio)
        val_s1_ids = set(all_s1_ids[:val_size])
        train_s1_ids = set(all_s1_ids[val_size:])

        train_s1_clean = [r for r in s1_clean if r['entity_id'] in train_s1_ids]
        val_s1_clean = [r for r in s1_clean if r['entity_id'] in val_s1_ids]

        train_gt = {k: v for k, v in gt_full.items() if k in train_s1_ids}
        val_gt = {k: v for k, v in gt_full.items() if k in val_s1_ids}

        print(f"[INFO] Train split: {len(train_s1_clean)} S1 entities | Validation split: {len(val_s1_clean)} S1 entities")

        print("\n=== STAGE 2: Blocking & Candidate Generation ===")
        train_candidates = self.blocker.fit_transform(train_s1_clean, target_clean)
        val_candidates = self.blocker.fit_transform(val_s1_clean, target_clean)

        val_blocking_metrics = compute_blocking_metrics(val_candidates, val_gt, len(target_clean))
        print(f"[METRICS] Validation Blocking Recall Ceiling: {val_blocking_metrics['recall_ceiling']:.4f}")
        print(f"[METRICS] Validation Reduction Ratio      : {val_blocking_metrics['reduction_ratio']:.4f}")

        print("\n=== STAGE 3: Feature Engineering & Model Training ===")
        all_recs = s1_clean + target_clean
        self.feature_extractor.fit(all_recs)

        # Build feature dataset for training matching classifier
        X_train, y_train = [], []
        for s1_rec in train_s1_clean:
            s1_id = s1_rec['entity_id']
            cands = train_candidates.get(s1_id, [])
            true_set = set(train_gt.get(s1_id, []))

            for cand_id in cands:
                cand_rec = target_dict.get(cand_id)
                if not cand_rec:
                    continue
                feats = self.feature_extractor.extract_pair_features(s1_rec, cand_rec)
                label = 1 if cand_id in true_set else 0
                X_train.append(feats)
                y_train.append(label)

        print(f"[INFO] Training classifier on {len(X_train)} candidate pairs (Positives: {sum(y_train)})...")
        self.model.fit(X_train, y_train)

        print("\n=== STAGE 4: Threshold Calibration on Validation Set ===")
        val_pair_scores = {}
        for s1_rec in val_s1_clean:
            s1_id = s1_rec['entity_id']
            cands = val_candidates.get(s1_id, [])
            if not cands:
                val_pair_scores[s1_id] = []
                continue

            pair_feats = []
            valid_cand_ids = []
            for cand_id in cands:
                cand_rec = target_dict.get(cand_id)
                if cand_rec:
                    pair_feats.append(self.feature_extractor.extract_pair_features(s1_rec, cand_rec))
                    valid_cand_ids.append(cand_id)

            if pair_feats:
                probs = self.model.predict_proba(pair_feats)
                val_pair_scores[s1_id] = list(zip(valid_cand_ids, probs))
            else:
                val_pair_scores[s1_id] = []

        self.best_threshold, val_f05 = tune_threshold(val_pair_scores, val_gt)
        print(f"[RESULTS] Optimal Decision Threshold: {self.best_threshold:.3f}")
        print(f"[RESULTS] Validation Macro F_0.5 Score: {val_f05:.4f}")

    def run_inference(self, test_dir: str, output_dir: str):
        """
        Runs end-to-end pipeline on unseen test data and generates submission files.
        """
        print("\n=== RUNNING INFERENCE ON TEST DATASET ===")
        s1_raw = load_tsv(os.path.join(test_dir, "test_source1.tsv"))
        s2_raw = load_tsv(os.path.join(test_dir, "test_source2.tsv"))
        s3_raw = load_tsv(os.path.join(test_dir, "test_source3.tsv"))

        print(f"[INFO] Loaded Test S1: {len(s1_raw)} | S2: {len(s2_raw)} | S3: {len(s3_raw)}")

        s1_clean = [preprocess_record(r) for r in s1_raw]
        s2_clean = [preprocess_record(r) for r in s2_raw]
        s3_clean = [preprocess_record(r) for r in s3_raw]

        target_clean = s2_clean + s3_clean
        target_dict = {r['entity_id']: r for r in target_clean}

        # Stage 2: Candidate Blocking
        test_candidates = self.blocker.fit_transform(s1_clean, target_clean)

        # Save candidate_pairs.tsv
        cand_filepath = os.path.join(output_dir, "candidate_pairs.tsv")
        save_output_tsv(cand_filepath, test_candidates, "source1_entity_id", "candidate_entity_ids")
        print(f"✅ Saved candidate pairs to {cand_filepath}")

        # Stage 3 & 4: Model Scoring & Thresholding
        final_matches = {}
        for s1_rec in s1_clean:
            s1_id = s1_rec['entity_id']
            cands = test_candidates.get(s1_id, [])
            if not cands:
                final_matches[s1_id] = []
                continue

            feats = []
            cand_ids = []
            for c_id in cands:
                c_rec = target_dict.get(c_id)
                if c_rec:
                    feats.append(self.feature_extractor.extract_pair_features(s1_rec, c_rec))
                    cand_ids.append(c_id)

            if feats:
                probs = self.model.predict_proba(feats)
                matched_ids = [c_id for c_id, p in zip(cand_ids, probs) if p >= self.best_threshold]
                final_matches[s1_id] = matched_ids
            else:
                final_matches[s1_id] = []

        # Save matching_results.tsv
        match_filepath = os.path.join(output_dir, "matching_results.tsv")
        save_output_tsv(match_filepath, final_matches, "source1_entity_id", "matched_entity_ids")
        print(f"✅ Saved final matches to {match_filepath}")
