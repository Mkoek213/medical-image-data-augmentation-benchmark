#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

RESOLUTION="${RESOLUTION:-256}"
BASELINE_SPLITS_DIR="${BASELINE_SPLITS_DIR:-outputs/advanced_baselines_full}"

STYLEGAN_RUN_ROOT="${STYLEGAN_RUN_ROOT:-outputs/stylegan2_all_minority_${RESOLUTION}px}"
MANIFEST="${MANIFEST:-${STYLEGAN_RUN_ROOT}/synthetic_manifest.csv}"
SPLITS_DIR="${SPLITS_DIR:-outputs/stylegan2_all_minority_${RESOLUTION}px_aug_train_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stylegan2_all_minority_${RESOLUTION}px_aug_train_full15}"

EPOCHS="${EPOCHS:-15}"
PATIENCE="${PATIENCE:-15}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Nie znaleziono interpretera: $PYTHON" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Brakuje manifestu obrazow syntetycznych: $MANIFEST" >&2
  echo "Najpierw uruchom:" >&2
  echo "  ./scripts/run_stylegan2_class_training.sh" >&2
  exit 1
fi

echo "Etap 1/2: przygotowanie splitu train z obrazami StyleGAN2 dla wszystkich klas mniejszosciowych"
"$PYTHON" scripts/prepare_synthetic_augmented_train_split.py \
  --baseline-splits-dir "$BASELINE_SPLITS_DIR" \
  --manifest "$MANIFEST" \
  --output-dir "$SPLITS_DIR"

echo "Etap 2/2: trening klasyfikatorow na 15 epok"
"$PYTHON" scripts/train_baselines.py \
  --splits-dir "$SPLITS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --early-stopping-patience "$PATIENCE" \
  --num-workers "$NUM_WORKERS"
