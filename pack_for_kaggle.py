"""
Pack ELIEI code + data into a single zip for Kaggle upload.

Run:  python pack_for_kaggle.py

Creates:  ELIEI_Kaggle.zip  in the current directory.
Upload as a Kaggle Dataset, then run the Kaggle notebook.

The zip layout:
    data/train/high/  (training high-light images)
    data/train/low/   (training low-light images)
    data/eval/high/   (evaluation high-light images)
    data/eval/low/    (evaluation low-light images)
    confs/IR-RGB-kaggle.yaml
    models/ ...
    data/   ...
    utils/  ...
    train.py, test.py, evaluate.py, smoke_test.py, ...
"""

import os
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))   # ELIEI_Implementation/
PARENT = os.path.dirname(BASE)                       # Low-lighten image/

# ── Source code files (relative to BASE, will be placed at zip root) ──
CODE_FILES = [
    "train.py",
    "test.py",
    "evaluate.py",
    "smoke_test.py",
    "requirements.txt",
    "README.md",
    "confs/IR-RGB-kaggle.yaml",
    "confs/IR-RGB.yaml",
    "models/__init__.py",
    "models/encoder.py",
    "models/flow.py",
    "models/loss.py",
    "models/model.py",
    "data/__init__.py",
    "data/dataset.py",
    "utils/__init__.py",
    "utils/utils.py",
]

# ── Dataset directories ──
# Mapping: source path (relative to PARENT) -> zip path
DATA_MAPPING = [
    ("train-20241116T084056Z-001/train/high", "data/train/high"),
    ("train-20241116T084056Z-001/train/low",  "data/train/low"),
    ("eval-20241116T084057Z-001/eval/high",   "data/eval/high"),
    ("eval-20241116T084057Z-001/eval/low",    "data/eval/low"),
]

OUT = os.path.join(BASE, "ELIEI_Kaggle.zip")


def main():
    count = 0
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Source code files
        print("Packing source code...")
        for rel in CODE_FILES:
            fullpath = os.path.join(BASE, rel)
            if os.path.isfile(fullpath):
                zf.write(fullpath, rel)
                count += 1
                print(f"  + {rel}")
            else:
                print(f"  ! MISSING: {rel}")

        # 2. Dataset images
        print("\nPacking dataset images...")
        for src_rel, dst_prefix in DATA_MAPPING:
            src_dir = os.path.join(PARENT, src_rel)
            if not os.path.isdir(src_dir):
                print(f"  ! MISSING DIR: {src_rel}")
                continue
            n_files = 0
            for root, _, files in os.walk(src_dir):
                for f in files:
                    if f.startswith('.'):
                        continue
                    fp = os.path.join(root, f)
                    # Flatten: put directly in dst_prefix/filename
                    arcname = dst_prefix + "/" + f
                    zf.write(fp, arcname)
                    count += 1
                    n_files += 1
            print(f"  + {dst_prefix}/ ({n_files} images)")

    size_mb = os.path.getsize(OUT) / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"Done! {count} files -> {OUT}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"{'='*60}")
    print()
    print("Next steps:")
    print("  1. Go to https://www.kaggle.com/datasets/new")
    print("  2. Upload ELIEI_Kaggle.zip as a new dataset")
    print("  3. Name it 'eliei-dataset'")
    print("  4. Create a new Kaggle Notebook")
    print("  5. Add the dataset to the notebook")
    print("  6. Paste/upload the Kaggle notebook")
    print("  7. Enable GPU accelerator -> Run All")


if __name__ == "__main__":
    main()
