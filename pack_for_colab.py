"""
Pack ELIEI code + data into a single zip for Google Colab upload.

Run:  python pack_for_colab.py

Creates:  ELIEI_Colab.zip  in the current directory.
Upload this single zip to Google Drive root, then run the Colab notebook.
"""

import os
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)

# Files/dirs to include (relative to PARENT so that paths are preserved)
INCLUDE = [
    # Code
    "ELIEI_Implementation/train.py",
    "ELIEI_Implementation/test.py",
    "ELIEI_Implementation/evaluate.py",
    "ELIEI_Implementation/smoke_test.py",
    "ELIEI_Implementation/requirements.txt",
    "ELIEI_Implementation/confs/IR-RGB.yaml",
    "ELIEI_Implementation/models/__init__.py",
    "ELIEI_Implementation/models/encoder.py",
    "ELIEI_Implementation/models/flow.py",
    "ELIEI_Implementation/models/loss.py",
    "ELIEI_Implementation/models/model.py",
    "ELIEI_Implementation/data/__init__.py",
    "ELIEI_Implementation/data/dataset.py",
    "ELIEI_Implementation/utils/__init__.py",
    "ELIEI_Implementation/utils/utils.py",
]

# Data directories
DATA_DIRS = [
    "train-20241116T084056Z-001/train/high",
    "train-20241116T084056Z-001/train/low",
    "eval-20241116T084057Z-001/eval/high",
    "eval-20241116T084057Z-001/eval/low",
]

OUT = os.path.join(BASE, "ELIEI_Colab.zip")


def main():
    count = 0
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Source files
        for rel in INCLUDE:
            fullpath = os.path.join(PARENT, rel)
            if os.path.isfile(fullpath):
                zf.write(fullpath, rel)
                count += 1
                print(f"  + {rel}")
            else:
                print(f"  ! MISSING: {rel}")

        # 2. Dataset images
        for ddir in DATA_DIRS:
            folder = os.path.join(PARENT, ddir)
            if not os.path.isdir(folder):
                print(f"  ! MISSING DIR: {ddir}")
                continue
            for root, _, files in os.walk(folder):
                for f in files:
                    fp = os.path.join(root, f)
                    arcname = os.path.relpath(fp, PARENT).replace("\\", "/")
                    zf.write(fp, arcname)
                    count += 1
            print(f"  + {ddir}/ ({len(os.listdir(folder))} files)")

    size_mb = os.path.getsize(OUT) / 1024 / 1024
    print(f"\nDone! {count} files -> {OUT}  ({size_mb:.1f} MB)")
    print("Upload this file to Google Drive root, then run the Colab notebook.")


if __name__ == "__main__":
    main()
