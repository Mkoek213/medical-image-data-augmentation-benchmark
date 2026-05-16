"""
Chest X-Ray Geometric Augmentation Pipeline
============================================
Augments minority classes in a CXR dataset until every class reaches TARGET_COUNT images.

Input structure:
    data/
        images/         <- source PNGs
        labels.csv      <- columns: image, label, patient_id

Output structure:
    augmented_data/
        images/         <- original images copied + new augmented PNGs
        labels.csv      <- full manifest (originals + augmented rows)

Augmentation choices (research-backed for chest X-rays):
    - Random resized crop          : best single augmentation for CXR representation
    - Rotation +-10 deg            : simulates positioning variation (never >15 deg)
    - Shift (translate) +-6%       : simulates patient centering variation
    - Zoom 0.95-1.10x              : simulates detector distance variation
    - Elastic deformation          : mild; simulates soft-tissue variability
    - Brightness +-15%             : simulates exposure variation
    - Gamma correction 0.8-1.2     : simulates detector response variation
    - Gaussian noise (low sigma)   : simulates machine-to-machine sensor noise

Deliberately excluded:
    - Horizontal flip  : creates dextrocardia appearance (cardiac laterality error)
    - Vertical flip    : anatomically invalid
    - Blur             : destroys subtle density differences that carry diagnostic signal
    - Heavy contrast   : not statistically significant in ChestX-ray14 studies
"""

import os
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────

TARGET_COUNT   = 3000          # minimum images per class after augmentation
INPUT_DIR      = Path("data")
OUTPUT_DIR     = Path("augmented_data")
IMAGES_SUBDIR  = "images"
LABELS_FILE    = "labels.csv"
SEED           = 42
IMAGE_SIZE     = 1024          # resize to this before augmentation; set to 0 to keep original

# ─── Augmentation Pipeline ────────────────────────────────────────────────────

def build_pipeline(image_size):
    """
    Returns an Albumentations pipeline tuned for chest X-rays.
    All parameters are at the conservative end of what the literature supports.
    """
    transforms = []

    # 1. Random resized crop — best single augmentation for CXR
    #    Crops 80-100% of the image then resizes back; forces the model
    #    to be invariant to field-of-view differences across machines.
    if image_size:
        transforms.append(
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.80, 1.00),   # crop 80-100% of the image area
                ratio=(0.90, 1.10),   # slight aspect ratio jitter
                p=0.70,
            )
        )

    # 2. Rotation +-10 deg — small angles only; >15 deg is anatomically implausible
    transforms.append(
        A.Rotate(
            limit=10,
            border_mode=cv2.BORDER_REFLECT_101,   # reflect to avoid black corners
            p=0.60,
        )
    )

    # 3. Shift (translation) +-6% — simulates patient centering variation
    #    ShiftScaleRotate lets us control shift independently of rotation.
    transforms.append(
        A.ShiftScaleRotate(
            shift_limit=0.06,    # up to 6% of image dimension in each direction
            scale_limit=0.05,    # +-5% zoom
            rotate_limit=0,      # rotation already handled above
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.50,
        )
    )

    # 4. Elastic deformation — mild settings are critical; high alpha/sigma
    #    destroys the density gradients that encode pathology
    transforms.append(
        A.ElasticTransform(
            alpha=30,            # displacement magnitude (keep low for CXR)
            sigma=5,             # smoothness of the displacement field
            p=0.30,
        )
    )

    # 5. Brightness — simulates X-ray exposure variation across operators/machines
    #    Contrast is kept very mild: not statistically significant in CXR studies
    transforms.append(
        A.RandomBrightnessContrast(
            brightness_limit=0.15,   # +-15% brightness
            contrast_limit=0.08,     # +-8% contrast (minimal, per literature)
            p=0.60,
        )
    )

    # 6. Gamma correction — simulates detector response curves across machines
    transforms.append(
        A.RandomGamma(
            gamma_limit=(80, 120),   # gamma 0.8-1.2
            p=0.40,
        )
    )

    # 7. Gaussian noise — low sigma only; simulates sensor/detector noise variation
    transforms.append(
        A.GaussNoise(
            std_range=(0.01, 0.03),  # very mild; expressed as fraction of pixel range
            p=0.30,
        )
    )

    # Final resize to ensure consistent output dimensions
    if image_size:
        transforms.append(A.Resize(height=image_size, width=image_size))

    return A.Compose(transforms)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_image(path):
    """Load a grayscale PNG and return as uint8 RGB numpy array (required by albumentations)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def save_image(img_rgb, path):
    """Convert back to grayscale and save as PNG."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(str(path), gray)


def make_aug_filename(source_name, aug_index):
    """
    Generate a unique filename for an augmented image.
    Example: 00000003_000.png -> aug_000042_00000003_000.png
    """
    stem, ext = os.path.splitext(source_name)
    return f"aug_{aug_index:06d}_{stem}{ext}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(input_dir, output_dir, target_count, image_size, seed):
    random.seed(seed)
    np.random.seed(seed)

    src_images_dir  = input_dir / IMAGES_SUBDIR
    src_labels_path = input_dir / LABELS_FILE
    out_images_dir  = output_dir / IMAGES_SUBDIR
    out_labels_path = output_dir / LABELS_FILE

    # ── Validate input ────────────────────────────────────────────────────────
    if not src_images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {src_images_dir}")
    if not src_labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {src_labels_path}")

    out_images_dir.mkdir(parents=True, exist_ok=True)

    # ── Load labels ───────────────────────────────────────────────────────────
    df = pd.read_csv(src_labels_path)
    if not {"image", "label"}.issubset(df.columns):
        raise ValueError(f"labels.csv must contain 'image' and 'label' columns. Found: {set(df.columns)}")

    print(f"\n{'='*60}")
    print(f"  Chest X-Ray Augmentation Pipeline")
    print(f"{'='*60}")
    print(f"  Source      : {input_dir}")
    print(f"  Destination : {output_dir}")
    print(f"  Target/class: {target_count:,}")
    print(f"  Image size  : {image_size if image_size else 'original'}")
    print(f"{'='*60}\n")

    # ── Step 1: Compute augmentation needs per class ─────────────────────────
    # Build a map from image filename -> patient_id for id propagation
    patient_map = df.set_index("image")["patient_id"].to_dict()
    class_counts = df.groupby("label")["image"].apply(list).to_dict()
    needs = {
        cls: max(0, target_count - len(imgs))
        for cls, imgs in class_counts.items()
    }

    print("\nStep 1/2 — Class distribution & augmentation plan:\n")
    print(f"  {'Class':<25} {'Real':>6}  {'Need':>6}  {'Target':>7}")
    print(f"  {'-'*50}")
    for cls in sorted(class_counts.keys(), key=lambda c: len(class_counts[c])):
        real = len(class_counts[cls])
        need = needs[cls]
        print(f"  {cls:<25} {real:>6,}  {need:>6,}  {target_count:>7,}")

    total_to_generate = sum(needs.values())
    if total_to_generate == 0:
        print("\n  All classes already meet the target. Nothing to augment.")
    else:
        print(f"\n  Total new images to generate: {total_to_generate:,}\n")

    # ── Step 2: Generate augmented images ────────────────────────────────────
    print("Step 2/2 — Augmenting ...")
    pipeline  = build_pipeline(image_size)
    aug_index = 0
    output_rows = []

    for cls, images_in_class in class_counts.items():
        n_needed = needs[cls]
        if n_needed == 0:
            print(f"  [{cls}] already at target — skipping")
            continue

        print(f"  [{cls}] generating {n_needed:,} images from {len(images_in_class):,} originals ...")

        # Cycle through source images with repetition if n_needed > len(class)
        source_pool = (images_in_class * (n_needed // len(images_in_class) + 1))[:n_needed]
        random.shuffle(source_pool)

        for src_name in tqdm(source_pool, desc=f"    {cls}", leave=False):
            src_path = src_images_dir / src_name

            try:
                img = load_image(src_path)
            except FileNotFoundError as e:
                print(f"\n  WARNING: {e} — skipping")
                continue

            augmented = pipeline(image=img)["image"]

            aug_name = make_aug_filename(src_name, aug_index)
            save_image(augmented, out_images_dir / aug_name)

            output_rows.append({
                "image":      aug_name,
                "label":      cls,
                "patient_id": patient_map.get(src_name, 0),
            })

            aug_index += 1

    # ── Write output labels.csv (augmented images only) ───────────────────────
    out_df = pd.DataFrame(output_rows, columns=["image", "label", "patient_id"])
    out_df.to_csv(out_labels_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Done! Final class distribution:")
    print(f"{'='*60}")
    final_counts = out_df.groupby("label").size().sort_values()
    for cls, count in final_counts.items():
        bar = "█" * (count // 200)
        print(f"  {cls:<25} {count:>6,}  {bar}")
    print(f"\n  Output labels : {out_labels_path}")
    print(f"  Output images : {out_images_dir}")
    print(f"  Total images  : {len(out_df):,}")
    print(f"{'='*60}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment chest X-ray classes to a minimum target count."
    )
    parser.add_argument(
        "--input",  type=Path, default=INPUT_DIR,
        help=f"Root input directory containing images/ and labels.csv  (default: {INPUT_DIR})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help=f"Root output directory  (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_COUNT,
        help=f"Minimum images per class after augmentation  (default: {TARGET_COUNT})",
    )
    parser.add_argument(
        "--size",   type=int, default=IMAGE_SIZE,
        help="Resize images to this square size before augmenting. Set 0 to keep original.  (default: 1024)",
    )
    parser.add_argument(
        "--seed",   type=int, default=SEED,
        help=f"Random seed for reproducibility  (default: {SEED})",
    )

    args = parser.parse_args()
    image_size = args.size if args.size > 0 else None

    main(
        input_dir    = args.input,
        output_dir   = args.output,
        target_count = args.target,
        image_size   = image_size,
        seed         = args.seed,
    )