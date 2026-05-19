#!/usr/bin/env python3
"""Create a train-only geometric augmentation split for baseline-compatible training.

The validation and test splits are copied from an existing baseline split directory.
Only the training split is expanded with generated images, which keeps evaluation
directly comparable with the baseline run.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parent.parent


def resolve_image_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [
        (ROOT_DIR / path).resolve(),
        (ROOT_DIR / "notebooks" / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def gamma_jitter(image: Image.Image) -> Image.Image:
    gamma = random.uniform(0.8, 1.2)
    table = [min(255, int(((i / 255.0) ** gamma) * 255.0)) for i in range(256)]
    return image.point(table)


def gaussian_noise(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    sigma = random.uniform(0.01, 0.03) * 255.0
    arr = np.clip(arr + np.random.normal(0.0, sigma, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.80, 1.00),
                ratio=(0.90, 1.10),
                antialias=True,
            ),
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=10, fill=0)],
                p=0.60,
            ),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.06, 0.06),
                        scale=(0.95, 1.05),
                        fill=0,
                    )
                ],
                p=0.50,
            ),
            transforms.RandomApply(
                [transforms.ElasticTransform(alpha=30.0, sigma=5.0, fill=0)],
                p=0.30,
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.15, contrast=0.08)],
                p=0.60,
            ),
            transforms.RandomApply([transforms.Lambda(gamma_jitter)], p=0.40),
            transforms.RandomApply([transforms.Lambda(gaussian_noise)], p=0.30),
            transforms.Resize((image_size, image_size), antialias=True),
        ]
    )


def normalize_split(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["image_path"] = frame["image_path"].map(lambda p: str(resolve_image_path(p)))
    frame["exists"] = frame["image_path"].map(lambda p: Path(p).exists())
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate geometric augmentations for the training split only."
    )
    parser.add_argument(
        "--baseline-splits-dir",
        type=Path,
        default=ROOT_DIR / "outputs" / "advanced_baselines_full",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs" / "geometric_aug_train_3000_dataset",
    )
    parser.add_argument("--target-per-class", type=int, default=3000)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    train_df = normalize_split(pd.read_csv(args.baseline_splits_dir / "train_split.csv"))
    val_df = normalize_split(pd.read_csv(args.baseline_splits_dir / "val_split.csv"))
    test_df = normalize_split(pd.read_csv(args.baseline_splits_dir / "test_split.csv"))

    if not train_df["exists"].all():
        missing = train_df.loc[~train_df["exists"], "image_path"].head().to_list()
        raise FileNotFoundError(f"Missing train images, examples: {missing}")
    if not val_df["exists"].all() or not test_df["exists"].all():
        raise FileNotFoundError("Missing validation/test images in baseline splits.")

    transform = build_transform(args.image_size)
    class_counts = train_df["label"].value_counts().sort_index()
    needs = (args.target_per_class - class_counts).clip(lower=0).astype(int)
    total_needed = int(needs.sum())

    print(f"baseline train samples : {len(train_df):,}")
    print(f"target per class       : {args.target_per_class:,}")
    print(f"new augmented images   : {total_needed:,}")
    print(f"output dir             : {args.output_dir}")
    print("\naugmentation plan:")
    for label, need in needs[needs > 0].sort_values(ascending=False).items():
        print(f"  {label:<22} current={class_counts[label]:>5,}  add={need:>5,}")

    augmented_rows: list[dict] = []
    aug_index = 0
    for label, need in needs[needs > 0].items():
        source_rows = train_df[train_df["label"] == label].reset_index(drop=True)
        source_indices = list(range(len(source_rows)))
        repeated = (source_indices * (need // len(source_indices) + 1))[:need]
        random.shuffle(repeated)

        for idx in tqdm(repeated, desc=f"augment {label}", leave=False):
            row = source_rows.iloc[idx]
            source_path = Path(row["image_path"])
            image = Image.open(source_path).convert("L")
            image = ImageOps.exif_transpose(image)
            augmented = transform(image)

            aug_name = f"geom_aug_{aug_index:06d}_{source_path.stem}.png"
            aug_path = images_dir / aug_name
            augmented.save(aug_path)

            new_row = row.to_dict()
            new_row["image"] = aug_name
            new_row["image_path"] = str(aug_path.resolve())
            new_row["exists"] = True
            new_row["is_augmented"] = True
            new_row["source_image"] = row["image"]
            augmented_rows.append(new_row)
            aug_index += 1

    train_df = train_df.copy()
    train_df["is_augmented"] = False
    train_df["source_image"] = train_df["image"]
    augmented_df = pd.DataFrame(augmented_rows)
    combined_train = pd.concat([train_df, augmented_df], ignore_index=True)

    combined_train.to_csv(args.output_dir / "train_split.csv", index=False)
    val_df.to_csv(args.output_dir / "val_split.csv", index=False)
    test_df.to_csv(args.output_dir / "test_split.csv", index=False)

    class_distribution = (
        combined_train["label"].value_counts().sort_index().rename_axis("label").reset_index(name="train_count")
    )
    class_distribution.to_csv(args.output_dir / "train_class_distribution.csv", index=False)

    print("\nfinal train distribution:")
    print(class_distribution.to_string(index=False))
    print(f"\ntrain split: {args.output_dir / 'train_split.csv'}")
    print(f"val split  : {args.output_dir / 'val_split.csv'}")
    print(f"test split : {args.output_dir / 'test_split.csv'}")


if __name__ == "__main__":
    main()
