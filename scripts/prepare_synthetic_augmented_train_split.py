#!/usr/bin/env python3
"""Add synthetic images to the baseline train split while keeping val/test fixed."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def resolve_existing_image_path(raw_path: str | Path, image_name: str | None, split_dir: Path) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    candidates = []

    raw = Path(str(raw_path))
    candidates.extend(
        [
            raw,
            split_dir / raw,
            repo_root / raw,
            repo_root / "notebooks" / raw,
        ]
    )
    if image_name:
        candidates.extend(
            [
                repo_root / "data_single_label_full" / "images" / image_name,
                repo_root / "data" / "images" / image_name,
            ]
        )

    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.exists():
            return candidate.resolve()

    return raw.resolve()


def normalize_existing_paths(df: pd.DataFrame, split_dir: Path) -> pd.DataFrame:
    df = df.copy()
    if "image_path" not in df.columns:
        if "image" not in df.columns:
            raise ValueError("Split CSV must contain image_path or image column")
        df["image_path"] = df["image"]
    df["image_path"] = df.apply(
        lambda row: str(resolve_existing_image_path(row["image_path"], row.get("image"), split_dir)),
        axis=1,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-splits-dir", type=Path, default=Path("outputs/advanced_baselines_full"))
    parser.add_argument("--synthetic-dir", type=Path, default=None)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--glob", default="*.png")
    args = parser.parse_args()

    train_df = normalize_existing_paths(
        pd.read_csv(args.baseline_splits_dir / "train_split.csv"),
        args.baseline_splits_dir,
    )
    val_df = normalize_existing_paths(
        pd.read_csv(args.baseline_splits_dir / "val_split.csv"),
        args.baseline_splits_dir,
    )
    test_df = normalize_existing_paths(
        pd.read_csv(args.baseline_splits_dir / "test_split.csv"),
        args.baseline_splits_dir,
    )

    if args.manifest is not None:
        manifest = pd.read_csv(args.manifest)
        required = {"image_path", "label"}
        if not required.issubset(manifest.columns):
            raise ValueError(f"Manifest must contain columns {required}. Found: {set(manifest.columns)}")

        if args.max_images is not None:
            manifest = manifest.groupby("label", group_keys=False).head(args.max_images)

        synthetic_rows = []
        for idx, row in manifest.reset_index(drop=True).iterrows():
            path = Path(str(row["image_path"])).resolve()
            if not path.exists():
                continue
            label = str(row["label"])
            synthetic_rows.append(
                {
                    "image": row.get("image", path.name),
                    "label": label,
                    "patient_id": row.get("patient_id", f"synthetic_{label}_{idx:06d}"),
                    "image_path": str(path),
                }
            )
    else:
        if args.synthetic_dir is None or args.class_name is None:
            raise ValueError("Provide either --manifest or both --synthetic-dir and --class-name")

        synthetic_paths = sorted(args.synthetic_dir.glob(args.glob))
        if args.max_images is not None:
            synthetic_paths = synthetic_paths[: args.max_images]
        if not synthetic_paths:
            raise ValueError(f"No synthetic images found in {args.synthetic_dir} with glob {args.glob!r}")

        synthetic_rows = []
        for idx, path in enumerate(synthetic_paths):
            synthetic_rows.append(
                {
                    "image": path.name,
                    "label": args.class_name,
                    "patient_id": f"synthetic_{args.class_name}_{idx:06d}",
                    "image_path": str(path.resolve()),
                }
            )

    if not synthetic_rows:
        raise ValueError("No usable synthetic images found")

    augmented_train = pd.concat([train_df, pd.DataFrame(synthetic_rows)], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    augmented_train.to_csv(args.output_dir / "train_split.csv", index=False)
    val_df.to_csv(args.output_dir / "val_split.csv", index=False)
    test_df.to_csv(args.output_dir / "test_split.csv", index=False)

    counts = augmented_train["label"].value_counts().sort_index()
    counts.rename_axis("label").reset_index(name="train_count").to_csv(
        args.output_dir / "train_class_distribution.csv", index=False
    )

    print(f"manifest     : {args.manifest if args.manifest else 'single class mode'}")
    print(f"synthetic dir: {args.synthetic_dir if args.synthetic_dir else '--'}")
    print(f"added images : {len(synthetic_rows)}")
    print(f"train split  : {args.output_dir / 'train_split.csv'}")
    print(f"val split    : {args.output_dir / 'val_split.csv'}")
    print(f"test split   : {args.output_dir / 'test_split.csv'}")
    print("final train distribution:")
    print(counts.to_string())


if __name__ == "__main__":
    main()
