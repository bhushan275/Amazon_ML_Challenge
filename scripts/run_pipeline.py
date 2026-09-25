#!/usr/bin/env python3
"""
Master Execution Pipeline for Amazon ML Challenge 2026 - Business Entity Resolution.

Runs the complete 4-stage workflow:
1. Data Verification / Synthetic Data Generation
2. Stage 1: Preprocessing & Normalization
3. Stage 2: Multi-pass Candidate Blocking
4. Stage 3: Feature Extraction & Model Training
5. Stage 4: Threshold Calibration, Inference, and Submission Validation
"""

import sys
import os
import subprocess
from pathlib import Path

# Add code/business_entity_resolution to python path
BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = BASE_DIR / "code" / "business_entity_resolution"
sys.path.insert(0, str(CODE_DIR))

from src.evaluation.pipeline import EntityResolutionPipeline


def main():
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026 - BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 70)

    train_dir = BASE_DIR / "dataset" / "train"
    test_dir = BASE_DIR / "dataset" / "test"
    output_dir = BASE_DIR / "output"

    # Step 1: Ensure datasets exist
    s1_train_file = train_dir / "train_source1.tsv"
    if not s1_train_file.exists():
        print("\n[INFO] Dataset not found. Generating synthetic dataset...")
        gen_script = BASE_DIR / "scripts" / "generate_synthetic_data.py"
        subprocess.run([sys.executable, str(gen_script)], check=True)

    # Step 2: Initialize & Execute Pipeline
    pipeline = EntityResolutionPipeline(top_k_tfidf=15)
    pipeline.train_and_validate(train_dir=str(train_dir), val_split_ratio=0.2)
    pipeline.run_inference(test_dir=str(test_dir), output_dir=str(output_dir))

    # Step 3: Run Official Validator
    print("\n" + "=" * 70)
    print("  RUNNING SUBMISSION VALIDATOR")
    print("=" * 70)

    validator_script = BASE_DIR / "utils" / "validate_submission.py"
    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_file),
        "--candidate", str(candidate_file),
        "--test-dir", str(test_dir)
    ]

    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n🎉 Pipeline completed successfully and submission passed format validation!")
    else:
        print("\n❌ Pipeline completed but submission failed validation check.")
        sys.exit(1)


if __name__ == "__main__":
    main()
