"""Backward-compatible wrapper for legacy handcrafted feature extraction."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_audio_features import extract_main  # noqa: E402


def main():
    # The old command extracted handcrafted features only. Keep that behavior;
    # transcript-derived features are excluded in the new waveform-only branch.
    args = ["--extractors", "legacy"]
    args.extend(arg for arg in sys.argv[1:] if arg != "--no-transcript-features")
    return extract_main(args)

if __name__ == "__main__":
    raise SystemExit(main())
