#!/usr/bin/env python3
"""
Creates final submission ZIP package for Amazon ML Challenge 2026.
Matches exact folder hierarchy required by competition rules:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       ├── CONTRACTS.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import zipfile
import argparse
from pathlib import Path


def create_submission_zip(team_name: str = "team_alpha"):
    base_dir = Path(".").resolve()
    zip_filename = f"{team_name}_submission.zip"
    zip_path = base_dir / zip_filename

    print(f"Creating submission package: {zip_path}")

    files_to_pack = [
        # Output files
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        # Documentation
        ("Documentation_template.md", "Documentation_template.md"),
        # Code files
        ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
        ("code/business_entity_resolution/CONTRACTS.md", "code/business_entity_resolution/CONTRACTS.md"),
        ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
    ]

    # Recursively include code/business_entity_resolution/src
    src_dir = base_dir / "code" / "business_entity_resolution" / "src"
    if src_dir.exists():
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".py"):
                    full_path = Path(root) / file
                    rel_zip_path = full_path.relative_to(base_dir)
                    files_to_pack.append((str(rel_zip_path), str(rel_zip_path)))

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src_file, zip_arc in files_to_pack:
            file_path = base_dir / src_file
            if file_path.exists():
                zf.write(file_path, arcname=zip_arc)
                print(f"  + Added: {zip_arc}")
            else:
                print(f"  ! Warning: File missing, skipped: {src_file}")

    print(f"\n✅ Submission package created successfully at: {zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package submission ZIP archive.")
    parser.add_argument("--team-name", default="team_alpha", help="Team name for zip filename")
    args = parser.parse_args()

    create_submission_zip(args.team_name)
