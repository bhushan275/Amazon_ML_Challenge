# Business Entity Resolution Pipeline

Modular Python project structure for candidate blocking, feature engineering, pairwise matching, and entity clustering.

## Project Structure

```text
business_entity_resolution/
├── .gitignore               # Excludes dataset/, output/, .venv/, cache files
├── requirements.txt         # Core dependencies (pandas, unidecode, rapidfuzz, pytest, etc.)
├── README.md                # Project documentation
├── dataset/                 # Dataset folder (gitignored)
├── output/                  # Predictions and submission TSVs (gitignored)
├── notebooks/               # Jupyter exploration notebooks
├── tests/                   # Unit and integration test suite
│   ├── __init__.py
│   └── test_pipeline.py
├── utils/                   # Root utility scripts
│   ├── __init__.py
│   └── validate_submission.py
├── stubs/                   # Stubs and type definitions package
│   └── __init__.py
└── src/                     # Core source code package
    ├── __init__.py
    ├── config.py            # Global paths, parameters, hyperparameter settings
    ├── pipeline.py          # Pipeline CLI runner
    ├── utils/               # IO and formatting helpers
    │   ├── __init__.py
    │   └── io_helpers.py
    ├── data/                # Data loaders, cleaning, normalization, splitting
    │   ├── __init__.py
    │   ├── loaders.py
    │   ├── normalize.py
    │   └── split.py
    ├── blocking/            # Candidate pair retrieval & indexing
    │   ├── __init__.py
    │   └── blocker.py
    ├── features/            # Similarity metric computation & feature vectors
    │   ├── __init__.py
    │   └── builder.py
    ├── model/               # Pair classification models (LightGBM/XGBoost)
    │   ├── __init__.py
    │   └── matcher.py
    ├── evaluation/          # Pair-level & cluster-level metrics (Recall, Precision, F1)
    │   ├── __init__.py
    │   └── metrics.py
    └── postprocess/         # Match thresholding & graph-based entity clustering
        ├── __init__.py
        └── clustering.py
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run pipeline CLI:
   ```bash
   python -m src.pipeline --mode train
   ```

3. Run test suite:
   ```bash
   pytest tests/
   ```
