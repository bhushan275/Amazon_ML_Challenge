"""
Test cases for end-to-end pipeline execution and module interfaces.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_PATH, TRAIN_PATH, TEST_PATH, OUTPUT_PATH, RAW_SOURCE_FILES
from src.data.loaders import load_source, REQUIRED_COLUMNS


class TestConfig(unittest.TestCase):
    """Test configuration paths and settings."""

    def test_config_paths(self):
        """Test loading configuration path settings."""
        self.assertIsInstance(DATASET_PATH, Path)
        self.assertIsInstance(TRAIN_PATH, Path)
        self.assertIsInstance(TEST_PATH, Path)
        self.assertIsInstance(OUTPUT_PATH, Path)
        self.assertIn("source1", RAW_SOURCE_FILES)


class TestLoadSource(unittest.TestCase):
    """Test data loading and validation routines."""

    def test_load_source_missing_columns_validation(self):
        """Test load_source raises ValueError when required columns are missing."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = Path(tmp_dir) / "train_source1.tsv"
            bad_file.write_text("entity_id\tbusiness_name\tbusiness_address\nE1\tAcme\t123 Main\n", encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                load_source("source1", split="train", data_dir=tmp_dir)

            self.assertIn("missing required columns", str(ctx.exception))

    def test_load_source_valid_columns(self):
        """Test load_source successfully loads TSV when all required columns exist."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            valid_file = Path(tmp_dir) / "train_source1.tsv"
            valid_file.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nE1\tAcme\t123 Main\tUS\n", encoding="utf-8")

            loaded_df = load_source("source1", split="train", data_dir=tmp_dir)
            self.assertEqual(len(loaded_df), 1)
            self.assertTrue(set(REQUIRED_COLUMNS).issubset(set(loaded_df.columns)))


from src.data.normalize import (
    normalize_name,
    normalize_address,
    normalize_country,
    normalize_record,
    SUFFIXES
)


class TestNormalize(unittest.TestCase):
    """Test text, name, address, and country normalization routines."""

    def test_normalize_name(self):
        """Test name normalization with legal suffix stripping."""
        self.assertEqual(normalize_name("B+ Retail Inc"), "b retail")
        self.assertEqual(normalize_name("Acme Corp."), "acme")
        self.assertEqual(normalize_name("Global Solutions Pvt Ltd"), "global")
        self.assertEqual(normalize_name(None), "")

    def test_normalize_address(self):
        """Test address normalization and abbreviation expansion."""
        res = normalize_address("1795 Westchester Drive, High Point, NC 27262")
        self.assertEqual(res["postal_code"], "27262")
        self.assertIn("drive", res["street"])

        res2 = normalize_address("123 Main St., Apt 4B")
        self.assertIn("street", res2["address_norm"])

    def test_normalize_country(self):
        """Test country normalization and generalization to unseen countries."""
        self.assertEqual(normalize_country("US"), "US")
        self.assertEqual(normalize_country("United States of America"), "US")
        self.assertEqual(normalize_country("India"), "IN")
        self.assertEqual(normalize_country("France"), "FR")
        self.assertEqual(normalize_country("Germany"), "DE")
        self.assertEqual(normalize_country("Sweden"), "SWEDEN")

    def test_normalize_record(self):
        """Test full record normalization to dictionary schema."""
        row = {
            "entity_id": "E100",
            "business_name": "Prabhav Business Center Inc",
            "business_address": "797 Lake Town Block A, Kolkata 700089",
            "country": "India"
        }
        rec = normalize_record(row)
        self.assertEqual(rec["entity_id"], "E100")
        self.assertEqual(rec["name_norm"], "prabhav business center")
        self.assertEqual(rec["country"], "IN")
        self.assertEqual(rec["postal_code"], "700089")


from src.data.split import make_split, train_val_split


class TestSplit(unittest.TestCase):
    """Test train/val splitting logic."""

    def test_make_split_reproducibility(self):
        """Test make_split generates identical, reproducible splits given explicit random seed."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            gt_file = Path(tmp_dir) / "train_ground_truth.tsv"
            # Create dummy ground truth data with matched vs unmatched entities
            gt_data = "source1_entity_id\tmatched_entity_ids\n"
            gt_data += "S1-1\tS2-1,S3-1\n"
            gt_data += "S1-2\t\n"
            gt_data += "S1-3\tS2-3\n"
            gt_data += "S1-4\t\n"
            gt_data += "S1-5\tS3-5\n"
            gt_file.write_text(gt_data, encoding="utf-8")

            train1, val1 = make_split(gt_file, val_fraction=0.4, seed=42, save_to_disk=True)
            train2, val2 = make_split(gt_file, val_fraction=0.4, seed=42, save_to_disk=False)

            self.assertEqual(list(train1["source1_entity_id"]), list(train2["source1_entity_id"]))
            self.assertEqual(list(val1["source1_entity_id"]), list(val2["source1_entity_id"]))
            self.assertEqual(len(train1) + len(val1), 5)
            self.assertTrue((Path(tmp_dir) / "split_train.tsv").exists())
            self.assertTrue((Path(tmp_dir) / "split_val.tsv").exists())


if __name__ == "__main__":
    unittest.main()


