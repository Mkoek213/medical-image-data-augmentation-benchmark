"""
StyleGAN2-ADA test run on Hernia class images.

Requirements:
    - CUDA GPU (this script will not run on CPU/MPS — StyleGAN2-ADA uses
      custom CUDA kernels). Run on a cloud GPU (Colab, RunPod, Lambda, etc.)
    - ~4 GB VRAM for 256×256  |  ~10 GB for 512×512  |  ~24 GB for 1024×1024

Setup (run once):
    git clone https://github.com/NVlabs/stylegan2-ada-pytorch
    pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3

Then run this script:
    python scripts/test_stylegan2_hernia.py
    python scripts/test_stylegan2_hernia.py --res 512 --kimg 50
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
STYLEGAN_DIR = REPO_ROOT / "stylegan2-ada-pytorch"   # cloned repo expected here
DATA_DIR    = REPO_ROOT / "data"
LABELS_CSV  = DATA_DIR / "labels.csv"
IMAGES_DIR  = DATA_DIR / "images"

TARGET_CLASS = "Hernia"

# ── Helpers ───────────────────────────────────────────────────────────────────

def check_setup():
    if not STYLEGAN_DIR.exists():
        print("ERROR: stylegan2-ada-pytorch not found.")
        print("Run from the repo root:")
        print("  git clone https://github.com/NVlabs/stylegan2-ada-pytorch")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        print("StyleGAN2-ADA requires a CUDA GPU.")
        print("Run this on a cloud GPU instance (Colab, RunPod, Lambda, etc.)")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")


def prepare_images(res: int) -> Path:
    """Copy and resize Hernia images into a staging folder."""
    df = pd.read_csv(LABELS_CSV)
    hernia = df[df["label"] == TARGET_CLASS]
    print(f"Found {len(hernia)} {TARGET_CLASS} images in labels.csv")

    staging = REPO_ROOT / f"stylegan_staging_{TARGET_CLASS.lower()}_{res}px"
    staging.mkdir(exist_ok=True)

    copied = 0
    for _, row in hernia.iterrows():
        src = IMAGES_DIR / row["image"]
        dst = staging / row["image"]
        if dst.exists():
            copied += 1
            continue
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  WARNING: could not read {src}, skipping")
            continue
        # StyleGAN2 expects RGB PNGs
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if res != img.shape[0] or res != img.shape[1]:
            img_rgb = cv2.resize(img_rgb, (res, res), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(dst), img_rgb)
        copied += 1

    print(f"Staged {copied} images at {res}×{res} → {staging}")
    return staging


def build_dataset(staging: Path, res: int) -> Path:
    """Convert staging folder into a StyleGAN2 zip dataset."""
    zip_out = REPO_ROOT / f"stylegan_dataset_{TARGET_CLASS.lower()}_{res}px.zip"
    if zip_out.exists():
        print(f"Dataset zip already exists: {zip_out}")
        return zip_out

    cmd = [
        sys.executable,
        str(STYLEGAN_DIR / "dataset_tool.py"),
        f"--source={staging}",
        f"--dest={zip_out}",
        f"--width={res}",
        f"--height={res}",
    ]
    print(f"\nBuilding dataset zip...")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return zip_out


def run_training(dataset_zip: Path, res: int, kimg: int, outdir: Path):
    """Run StyleGAN2-ADA training."""
    outdir.mkdir(parents=True, exist_ok=True)

    # cfg presets: cifar (fast, small), paper256, paper512, paper1024
    cfg = {256: "paper256", 512: "paper512", 1024: "paper1024"}.get(res, "auto")

    cmd = [
        sys.executable,
        str(STYLEGAN_DIR / "train.py"),
        f"--outdir={outdir}",
        f"--data={dataset_zip}",
        "--gpus=1",
        f"--cfg={cfg}",
        f"--kimg={kimg}",          # total thousands of images to train on
        "--mirror=0",              # no horizontal flip — dextrocardia risk
        "--aug=ada",               # adaptive augmentation (key for small datasets)
        "--target=0.6",            # ADA target r_t: 0.6 is recommended for <1k images
        "--snap=10",               # save network snapshot every 10 kimg
        "--metrics=none",          # skip FID etc. for test runs (very slow)
    ]
    print(f"\nStarting training ({kimg} kimg at {res}×{res}) ...")
    print("  " + " ".join(cmd))
    print()
    subprocess.run(cmd, check=True, cwd=STYLEGAN_DIR)


def generate_samples(outdir: Path, n: int = 16):
    """Find the latest checkpoint and generate sample images."""
    pkls = sorted(outdir.glob("**/*.pkl"))
    if not pkls:
        print("No checkpoints found yet — training may not have completed a snapshot.")
        return

    latest = pkls[-1]
    samples_dir = outdir / "samples"
    samples_dir.mkdir(exist_ok=True)

    seeds = ",".join(str(i) for i in range(n))
    cmd = [
        sys.executable,
        str(STYLEGAN_DIR / "generate.py"),
        f"--outdir={samples_dir}",
        f"--trunc=0.7",            # truncation psi: lower = higher quality, less variety
        f"--seeds={seeds}",
        f"--network={latest}",
    ]
    print(f"\nGenerating {n} samples from {latest.name} ...")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=STYLEGAN_DIR)
    print(f"Samples saved to {samples_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--res", type=int, default=256, choices=[128, 256, 512, 1024],
        help="Training resolution. 256 recommended for a test run (default: 256)",
    )
    parser.add_argument(
        "--kimg", type=int, default=25,
        help="Thousands of images to train on. 25 kimg ≈ 10-30 min on a T4. "
             "You won't get good samples this quickly — this is just to verify the pipeline. "
             "A real run needs 1000-5000 kimg. (default: 25)",
    )
    parser.add_argument(
        "--only-prep", action="store_true",
        help="Only prepare the dataset zip, don't train. Useful for prepping on CPU then "
             "uploading to a GPU machine.",
    )
    parser.add_argument(
        "--only-generate", action="store_true",
        help="Skip training, just generate samples from the latest checkpoint.",
    )
    args = parser.parse_args()

    outdir = REPO_ROOT / f"stylegan_output_{TARGET_CLASS.lower()}_{args.res}px"

    # Data prep can run without CUDA
    staging = prepare_images(args.res)
    dataset_zip = build_dataset(staging, args.res)

    if args.only_prep:
        print(f"\nDataset ready: {dataset_zip}")
        print("Upload this zip to your GPU machine and run:")
        print(f"  python train.py --outdir=<outdir> --data={dataset_zip.name} "
              f"--gpus=1 --cfg=paper{args.res} --kimg={args.kimg} "
              f"--mirror=0 --aug=ada --target=0.6 --metrics=none")
        return

    check_setup()  # CUDA check — only required for actual training

    if not args.only_generate:
        run_training(dataset_zip, args.res, args.kimg, outdir)

    generate_samples(outdir)


if __name__ == "__main__":
    main()
