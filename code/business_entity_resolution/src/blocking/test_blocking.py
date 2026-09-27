"""
Comprehensive Unit & Integration Tests for P2 Blocking Module
=============================================================
Tests:
1. Phonetic algorithms (Soundex, Metaphone)
2. CountryAwareBlocker
3. SortedNeighborhoodBlocker (SNM)
4. MinHashLSHBlocker
5. TfidfCosineBlocker (both sklearn and pure python fallback)
6. MultiPassBlocker end-to-end
7. Recall Ceiling and Reduction Ratio computation
8. Output TSV format compliance (tab delimiter, one row per S1, S2/S3 IDs only, no duplicates)
"""

import os
import sys
import unittest
import tempfile
import csv
from pathlib import Path

# Add code directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.blocking.blocking import (
    generate_soundex_key,
    generate_metaphone_key,
    CountryAwareBlocker,
    SortedNeighborhoodBlocker,
    MinHashLSHBlocker,
    TfidfCosineBlocker,
    MultiPassBlocker,
    compute_blocking_metrics
)
from src.blocking.candidate_gen import (
    load_source_tsv,
    load_ground_truth,
    write_candidate_pairs_tsv,
    generate_mock_normalized_data,
    evaluate_blocking_on_split
)


class TestBlockingModule(unittest.TestCase):

    def setUp(self):
        self.s1_records = [
            {"entity_id": "S1-001", "name_clean": "apple inc tech", "address_clean": "1 infinite loop cupertino ca", "country": "US"},
            {"entity_id": "S1-002", "name_clean": "tata motors limited", "address_clean": "bombay house homi mody st mumbai", "country": "INDIA"},
            {"entity_id": "S1-003", "name_clean": "bnp paribas bank", "address_clean": "16 boulevard des italiens paris", "country": "FRANCE"},
            {"entity_id": "S1-004", "name_clean": "rare independent boutique", "address_clean": "99 secluded lane austin tx", "country": "US"},  # singleton
        ]

        self.target_records = [
            {"entity_id": "S2-001", "name_clean": "apple computer corporation", "address_clean": "infinite loop 1 cupertino", "country": "US"},
            {"entity_id": "S2-002", "name_clean": "tata motors auto", "address_clean": "homi mody street bombay mumbai", "country": "INDIA"},
            {"entity_id": "S3-001", "name_clean": "bnp paribas sa", "address_clean": "blvd des italiens 16 paris", "country": "FRANCE"},
            {"entity_id": "S3-002", "name_clean": "unrelated hardware corp", "address_clean": "500 market st san francisco", "country": "US"},
            {"entity_id": "S2-003", "name_clean": "apple store retail", "address_clean": "cupertino california", "country": "US"},
        ]

        self.ground_truth = {
            "S1-001": ["S2-001", "S2-003"],
            "S1-002": ["S2-002"],
            "S1-003": ["S3-001"],
            "S1-004": [],  # singleton
        }

    def test_phonetic_keys(self):
        # Soundex
        self.assertEqual(generate_soundex_key("Robert"), "R163")
        self.assertEqual(generate_soundex_key("Rupert"), "R163")
        self.assertEqual(generate_soundex_key("Amazon"), "A525")

        # Metaphone
        meta_phil = generate_metaphone_key("Philip")
        meta_fil = generate_metaphone_key("Filip")
        self.assertEqual(meta_phil, meta_fil)

    def test_country_aware_blocker(self):
        blocker = CountryAwareBlocker()
        blocker.fit(self.target_records)
        cands = blocker.query(self.s1_records)

        # Apple should match S2-001 or S2-003
        self.assertTrue("S2-001" in cands["S1-001"] or "S2-003" in cands["S1-001"])
        # Tata should match S2-002
        self.assertTrue("S2-002" in cands["S1-002"])

    def test_sorted_neighborhood_blocker(self):
        blocker = SortedNeighborhoodBlocker(window_size=3)
        cands = blocker.block(self.s1_records, self.target_records)

        # Ensure candidates are only S2 or S3
        for s1_id, t_ids in cands.items():
            for t_id in t_ids:
                self.assertFalse(t_id.startswith("S1-"))
                self.assertTrue(t_id.startswith("S2-") or t_id.startswith("S3-"))

    def test_minhash_lsh_blocker(self):
        blocker = MinHashLSHBlocker(num_perm=64, num_bands=16)
        blocker.fit(self.target_records)
        cands = blocker.query(self.s1_records)

        # Apple inc tech should match apple computer or apple store
        self.assertTrue("S2-001" in cands["S1-001"] or "S2-003" in cands["S1-001"])

    def test_tfidf_cosine_blocker(self):
        blocker = TfidfCosineBlocker(top_k=3)
        cands = blocker.block(self.s1_records, self.target_records)

        self.assertIn("S2-001", cands["S1-001"])
        self.assertIn("S2-002", cands["S1-002"])
        self.assertIn("S3-001", cands["S1-003"])

    def test_multi_pass_blocker_recall_and_reduction(self):
        blocker = MultiPassBlocker(top_k_tfidf=5, snm_window_size=5)
        candidate_map = blocker.fit_transform(self.s1_records, self.target_records)

        # Check every S1 entity is present in candidate_map
        for r in self.s1_records:
            self.assertIn(r['entity_id'], candidate_map)

        # Check all candidates are strictly S2 or S3 IDs
        for s1_id, cands in candidate_map.items():
            for c_id in cands:
                self.assertFalse(c_id.startswith("S1-"))
                self.assertTrue(c_id.startswith("S2-") or c_id.startswith("S3-"))
            # Check no duplicates
            self.assertEqual(len(cands), len(set(cands)))

        # Evaluate metrics
        metrics = compute_blocking_metrics(candidate_map, self.ground_truth, len(self.target_records))
        self.assertEqual(metrics["recall_ceiling"], 1.0)  # 100% recall on this test set
        self.assertGreaterEqual(metrics["reduction_ratio"], 0.0)

    def test_candidate_pairs_tsv_format(self):
        blocker = MultiPassBlocker()
        candidate_map = blocker.fit_transform(self.s1_records, self.target_records)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "candidate_pairs.tsv")
            write_candidate_pairs_tsv(out_file, candidate_map, [r['entity_id'] for r in self.s1_records])

            # Verify file format
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                rows = list(reader)

            # Check header
            self.assertEqual(rows[0], ["source1_entity_id", "candidate_entity_ids"])
            # Check row count: header + 4 S1 entities
            self.assertEqual(len(rows), 5)

            # Check each row
            seen_ids = set()
            for r in rows[1:]:
                self.assertEqual(len(r), 2)
                s1_id = r[0]
                self.assertNotIn(s1_id, seen_ids)
                seen_ids.add(s1_id)
                cand_str = r[1]
                if cand_str:
                    cands = cand_str.split(",")
                    for cid in cands:
                        self.assertFalse(cid.startswith("S1-"))
                        self.assertTrue(cid.startswith("S2-") or cid.startswith("S3-"))


if __name__ == "__main__":
    unittest.main()
