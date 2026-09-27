"""
Wrapper to execute make_stubs.py script from root workspace.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent / "business_entity_resolution"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_stubs import main

if __name__ == "__main__":
    main()
