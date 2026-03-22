"""
Converts Door Classification/RGB + Door Detection_Segmentation into a
merged YOLO detection dataset at door_combined_yolo/.

Both datasets share the same image names (Door0001.png, etc.), so:
- Class label  → from Classification/RGB folder structure (Closed/Open/Semi)
- Bounding box → from segmentation mask PNG (door pixels = 192, 224, 192)

Run once before training:
    python prepare_dataset.py
"""

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).parent
CLASS_DIR = BASE / "Door Classification" / "RGB"
SEG_IMAGES = BASE / "Door Detection_Segmentation" / "Images"
SEG_MASKS  = BASE / "Door Detection_Segmentation" / "Annotations"
OUT_DIR    = BASE / "door_combined_yolo"

DOOR_COLOR = np.array([128, 0, 0], dtype=np.uint8)
CLASS_MAP  = {"Closed": 0, "Open": 1, "Semi": 2}
SPLITS     = ["train", "val", "test"]


def build_classification_lookup():
    """Returns {image_name: (class_idx, split)}"""
    lookup = {}
    for split in SPLITS:
        for cls_name, cls_idx in CLASS_MAP.items():
            folder = CLASS_DIR / split / cls_name
            if not folder.exists():
                continue
            for img_path in folder.iterdir():
                lookup[img_path.name] = (cls_idx, split)
    return lookup


def extract_bbox_from_mask(mask_path):
    """
    Returns normalized (cx, cy, w, h) for the door region in the mask,
    or None if no door pixels found.
    """
    mask = np.array(Image.open(mask_path).convert("RGB"))
    door_pixels = np.all(mask == DOOR_COLOR, axis=-1)
    rows = np.where(door_pixels.any(axis=1))[0]
    cols = np.where(door_pixels.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    h_img, w_img = mask.shape[:2]
    r_min, r_max = rows[0], rows[-1]
    c_min, c_max = cols[0], cols[-1]
    cx = ((c_min + c_max) / 2) / w_img
    cy = ((r_min + r_max) / 2) / h_img
    bw = (c_max - c_min) / w_img
    bh = (r_max - r_min) / h_img
    return cx, cy, bw, bh


def write_label(label_path, class_idx, bbox):
    cx, cy, w, h = bbox
    label_path.write_text(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def make_dirs():
    for split in SPLITS:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)


def write_data_yaml():
    yaml_content = (
        f"path: {OUT_DIR}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "\n"
        "nc: 3\n"
        "names: ['closed', 'open', 'semi']\n"
    )
    (OUT_DIR / "data.yaml").write_text(yaml_content)


def main():
    print("Building classification lookup...")
    lookup = build_classification_lookup()
    print(f"  Found {len(lookup)} labelled images")

    make_dirs()

    stats = {"processed": 0, "no_mask": 0, "no_label": 0, "full_box": 0}
    split_counts = {s: 0 for s in SPLITS}

    # --- Process segmentation images (precise bounding boxes) ---
    seg_names = set()
    for img_path in sorted(SEG_IMAGES.iterdir()):
        name = img_path.name
        seg_names.add(name)

        mask_path = SEG_MASKS / name
        if not mask_path.exists():
            stats["no_mask"] += 1
            continue

        bbox = extract_bbox_from_mask(mask_path)
        if bbox is None:
            stats["no_mask"] += 1
            continue

        if name in lookup:
            cls_idx, split = lookup[name]
        else:
            # No class label — skip (can't train detection without class)
            stats["no_label"] += 1
            continue

        dest_img = OUT_DIR / split / "images" / name
        dest_lbl = OUT_DIR / split / "labels" / (img_path.stem + ".txt")
        shutil.copy2(img_path, dest_img)
        write_label(dest_lbl, cls_idx, bbox)
        stats["processed"] += 1
        split_counts[split] += 1

    # --- Process classification images NOT in segmentation set (full-image boxes) ---
    for split in SPLITS:
        for cls_name, cls_idx in CLASS_MAP.items():
            folder = CLASS_DIR / split / cls_name
            if not folder.exists():
                continue
            for img_path in folder.iterdir():
                if img_path.name in seg_names:
                    continue  # already handled above
                dest_img = OUT_DIR / split / "images" / img_path.name
                dest_lbl = OUT_DIR / split / "labels" / (img_path.stem + ".txt")
                if dest_img.exists():
                    continue  # duplicate name across classes (shouldn't happen)
                shutil.copy2(img_path, dest_img)
                write_label(dest_lbl, cls_idx, (0.5, 0.5, 1.0, 1.0))
                stats["processed"] += 1
                stats["full_box"] += 1
                split_counts[split] += 1

    write_data_yaml()

    print(f"\nDone!")
    print(f"  Processed : {stats['processed']}")
    print(f"  Full-image boxes (no mask): {stats['full_box']}")
    print(f"  Skipped (no mask match)   : {stats['no_mask']}")
    print(f"  Skipped (no class label)  : {stats['no_label']}")
    print(f"\nSplit distribution:")
    for s, n in split_counts.items():
        print(f"  {s:5s}: {n}")
    print(f"\nDataset written to: {OUT_DIR}")
    print(f"data.yaml          : {OUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
