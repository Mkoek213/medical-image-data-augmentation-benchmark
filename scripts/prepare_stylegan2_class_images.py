#!/usr/bin/env python3
"""Prepare train-split images for StyleGAN2-ADA class-conditional runs.

This script intentionally reads from the saved classifier train split, not from
the full labels file. That prevents train/test leakage when synthetic images are
later used in benchmark classifier training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def resolve_image_path(row: pd.Series, train_split: Path) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    candidates = []

    if "image_path" in row and pd.notna(row["image_path"]):
        raw = Path(str(row["image_path"]))
        candidates.extend(
            [
                raw,
                train_split.parent / raw,
                repo_root / raw,
                repo_root / "notebooks" / raw,
            ]
        )

    if "image" in row and pd.notna(row["image"]):
        image_name = str(row["image"])
        candidates.extend(
            [
                Path(image_name),
                train_split.parent / image_name,
                repo_root / "data_single_label_full" / "images" / image_name,
                repo_root / "data" / "images" / image_name,
            ]
        )

    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve image path for row: {row.to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", type=Path, default=Path("outputs/advanced_baselines_full/train_split.csv"))
    parser.add_argument("--class-name", default="Hernia")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stylegan2_staging/hernia_512"))
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_split)
    class_df = train_df[train_df["label"] == args.class_name].copy()
    if class_df.empty:
        raise ValueError(f"No rows for class {args.class_name!r} in {args.train_split}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for idx, row in tqdm(class_df.iterrows(), total=len(class_df), desc=f"stage {args.class_name}"):
        try:
            src = resolve_image_path(row, args.train_split)
        except FileNotFoundError:
            skipped += 1
            continue

        dst = args.output_dir / f"{idx:06d}_{src.stem}.png"
        if dst.exists():
            copied += 1
            continue

        img = Image.open(src).convert("RGB")
        if img.size != (args.resolution, args.resolution):
            img = img.resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
        img.save(dst)
        copied += 1

    print(f"class       : {args.class_name}")
    print(f"source split : {args.train_split}")
    print(f"output dir   : {args.output_dir}")
    print(f"staged       : {copied}")
    print(f"skipped      : {skipped}")


if __name__ == "__main__":
    main()
