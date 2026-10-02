"""Guard against accidentally regenerating RecruitView's fixed participant split."""
import sys

if __name__ == "__main__":
    print("Refusing to regenerate the fixed train/val/test split. The historical generator is archived at "
          "archive/split_dataset_generator.py; use the existing Split Dataset files as the source of truth.",
          file=sys.stderr)
    raise SystemExit(2)
