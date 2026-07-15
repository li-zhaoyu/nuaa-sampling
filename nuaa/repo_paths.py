"""Resolve repository root paths for standalone and monorepo layouts."""
from __future__ import annotations

from pathlib import Path

# nuaa/repo_paths.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT
OUTPUT_DIR = REPO_ROOT / "outputs"
FIGURE_DIR = REPO_ROOT / "figures"
CONFIG_DIR = REPO_ROOT / "configs"
