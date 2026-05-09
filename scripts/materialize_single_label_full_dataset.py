#!/usr/bin/env python3
"""Create a flat full single-label NIH ChestX-ray14 dataset.

Input is the original NIH layout with Data_Entry_2017.csv and images_*/images.
Output is a simple directory:

    output_dir/
    ├── images/
    └── labels.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path


NO_FINDING = "No Finding"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data_full"),
        help="Original NIH dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_single_label_full"),
        help="Output directory for flat images/ plus labels.csv.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="Use copy for independent files or hardlink to save disk space.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing image files in output-dir/images.",
    )
    return parser.parse_args()


def disease_labels(label_value: str) -> list[str]:
    labels = [label.strip() for label in label_value.split("|") if label.strip()]
    return [label for label in labels if label != NO_FINDING]


def single_label(label_value: str) -> str | None:
    diseases = disease_labels(label_value)
    if len(diseases) > 1:
        return None
    if not diseases:
        return NO_FINDING
    return diseases[0]


def build_image_index(source_dir: Path) -> dict[str, Path]:
    index = {}
    for image_dir in sorted(source_dir.glob("images_*/images")):
        for image_path in image_dir.glob("*.png"):
            index[image_path.name] = image_path
    return index


def materialize_image(source: Path, target: Path, copy_mode: str, overwrite: bool) -> str:
    if target.exists():
        if not overwrite:
            return "skipped"
        target.unlink()

    if copy_mode == "hardlink":
        target.hardlink_to(source)
    else:
        shutil.copy2(source, target)
    return "copied"


def main() -> None:
    args = parse_args()
    metadata_path = args.source_dir / "Data_Entry_2017.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "labels.csv"
    removed_path = args.output_dir / "removed_multilabel.csv"
    missing_path = args.output_dir / "missing_images.txt"
    class_counts_path = args.output_dir / "class_counts.csv"

    image_index = build_image_index(args.source_dir)

    total_rows = 0
    kept_rows = 0
    removed_rows = 0
    copied = 0
    skipped = 0
    missing = []
    class_counts: Counter[str] = Counter()

    with metadata_path.open(newline="") as metadata_handle, labels_path.open(
        "w", newline=""
    ) as labels_handle, removed_path.open("w", newline="") as removed_handle:
        reader = csv.DictReader(metadata_handle)
        labels_writer = csv.DictWriter(
            labels_handle, fieldnames=["image", "label", "patient_id"]
        )
        labels_writer.writeheader()

        removed_writer = csv.DictWriter(
            removed_handle,
            fieldnames=[
                "image",
                "finding_labels",
                "patient_id",
                "disease_count",
            ],
        )
        removed_writer.writeheader()

        for row in reader:
            total_rows += 1
            image_name = row["Image Index"]
            label = single_label(row["Finding Labels"])

            if label is None:
                removed_rows += 1
                removed_writer.writerow(
                    {
                        "image": image_name,
                        "finding_labels": row["Finding Labels"],
                        "patient_id": row["Patient ID"],
                        "disease_count": len(disease_labels(row["Finding Labels"])),
                    }
                )
                continue

            source_image = image_index.get(image_name)
            if source_image is None:
                missing.append(image_name)
                continue

            status = materialize_image(
                source_image,
                images_dir / image_name,
                copy_mode=args.copy_mode,
                overwrite=args.overwrite,
            )
            if status == "copied":
                copied += 1
            else:
                skipped += 1

            labels_writer.writerow(
                {
                    "image": image_name,
                    "label": label,
                    "patient_id": row["Patient ID"],
                }
            )
            class_counts[label] += 1
            kept_rows += 1

    with class_counts_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "count"])
        writer.writeheader()
        for label, count in class_counts.most_common():
            writer.writerow({"label": label, "count": count})

    if missing:
        missing_path.write_text("".join(f"{image_name}\n" for image_name in missing))
    elif missing_path.exists():
        missing_path.unlink()

    print(f"Source rows: {total_rows}")
    print(f"Kept single-label rows: {kept_rows}")
    print(f"Removed multi-label rows: {removed_rows}")
    print(f"Missing source images: {len(missing)}")
    print(f"Images copied/linked: {copied}")
    print(f"Images skipped: {skipped}")
    print(f"Output directory: {args.output_dir}")
    print(f"Labels CSV: {labels_path}")
    print(f"Class counts CSV: {class_counts_path}")


if __name__ == "__main__":
    main()
