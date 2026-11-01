"""
Merges HERIDAL (converted to YOLO format) and SARD (already YOLO format
from Kaggle/Roboflow) into a single unified dataset with one "person" class.

This is a PRIMARY + SUPPLEMENT merge, not a fixed-ratio blend -- both
datasets keep their full image counts. HERIDAL is real drone-captured
wilderness imagery (your primary domain match); SARD adds pose diversity
(running/walking/standing/sitting/lying, simulated injured/exhausted).

Usage:
    python scripts/merge_datasets.py \
        --heridal data/heridal \
        --sard data/sard \
        --output data/processed/merged
"""
import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

# SARD's Kaggle/Roboflow export may use a different class ID for "person" --
# adjust SARD_PERSON_CLASS_ID if your specific SARD download uses something
# other than 0. Check the dataset's data.yaml `names:` list to confirm.
SARD_PERSON_CLASS_ID = 0


def copy_split(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir, prefix, remap_class=None):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(src_img_dir.glob("*")):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        label_path = src_label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        new_name = f"{prefix}_{img_path.name}"
        new_label_name = f"{prefix}_{img_path.stem}.txt"

        shutil.copy(img_path, dst_img_dir / new_name)

        if remap_class is not None:
            lines = label_path.read_text().strip().splitlines()
            remapped = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                # Only keep person-class boxes; remap to class 0
                if int(parts[0]) == remap_class:
                    parts[0] = "0"
                    remapped.append(" ".join(parts))
            (dst_label_dir / new_label_name).write_text("\n".join(remapped))
        else:
            shutil.copy(label_path, dst_label_dir / new_label_name)

        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heridal", type=str, default="data/heridal")
    parser.add_argument("--sard", type=str, default="data/sard")
    parser.add_argument("--output", type=str, default="data/processed/merged")
    args = parser.parse_args()

    heridal_dir = Path(args.heridal)
    sard_dir = Path(args.sard)
    output_dir = Path(args.output)

    total = {"train": 0, "val": 0}

    for split in ["train", "val"]:
        out_img = output_dir / split / "images"
        out_label = output_dir / split / "labels"

        # HERIDAL
        h_img = heridal_dir / split / "images"
        h_lbl = heridal_dir / split / "labels"
        if h_img.exists():
            n = copy_split(h_img, h_lbl, out_img, out_label, prefix="heridal")
            total[split] += n
            print(f"HERIDAL {split}: {n} images")

        # SARD -- adjust subfolder names below to match your actual SARD export
        # (Kaggle/Roboflow exports commonly use train/valid/test with images/labels subfolders)
        sard_split_name = "valid" if split == "val" else split
        s_img = sard_dir / sard_split_name / "images"
        s_lbl = sard_dir / sard_split_name / "labels"
        if s_img.exists():
            n = copy_split(s_img, s_lbl, out_img, out_label, prefix="sard",
                            remap_class=SARD_PERSON_CLASS_ID)
            total[split] += n
            print(f"SARD {split}: {n} images")
        else:
            print(f"WARNING: {s_img} not found -- check your SARD folder structure "
                  f"and update sard_split_name / paths in this script if needed.")

    print(f"\nMerged dataset written to: {output_dir}")
    print(f"Total train: {total['train']} | Total val: {total['val']}")


if __name__ == "__main__":
    main()
