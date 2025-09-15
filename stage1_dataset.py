import os
import shutil
import random
from pathlib import Path

base_dir = Path("data")
stage1_dir = Path("stage1")

for split in ["train", "val", "test"]:
    for cls in ["fruit", "leaf"]:
        (stage1_dir / split / cls).mkdir(parents=True, exist_ok=True)

def split_and_copy(src_dir, dst_cls):
    all_images = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                all_images.append(Path(root) / file)

    random.shuffle(all_images)
    n = len(all_images)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }

    for split, files in splits.items():
        for file in files:
            shutil.copy(file, stage1_dir / split / dst_cls / file.name)

split_and_copy(base_dir / "fruit", "fruit")

split_and_copy(base_dir / "leaves", "leaf")

print("Stage-1 dataset created at:", stage1_dir)
