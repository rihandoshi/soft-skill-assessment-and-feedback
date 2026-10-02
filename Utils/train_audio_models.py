"""Backward-compatible wrapper for the historical legacy-only benchmark."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from archive.legacy_benchmark import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
