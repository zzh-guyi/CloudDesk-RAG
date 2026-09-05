"""
Evaluation entry point - scripts/evaluate.py
"""
import sys
import os
from pathlib import Path

# Add project root and local_packages to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "local_packages"))

from app.eval.evaluator import main

if __name__ == "__main__":
    main()
