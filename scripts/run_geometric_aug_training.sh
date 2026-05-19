#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

BASELINE_SPLITS_DIR="${BASELINE_SPLITS_DIR:-outputs/advanced_baselines_full}"
DATASET_DIR="${DATASET_DIR:-outputs/geometric_aug_train_3000_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/geometric_aug_train_3000_full15}"

TARGET_PER_CLASS="${TARGET_PER_CLASS:-3000}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-15}"
PATIENCE="${PATIENCE:-15}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Nie znaleziono interpretera: $PYTHON" >&2
  echo "Ustaw PYTHON=/sciezka/do/python albo uruchom z katalogu repo z aktywnym .venv." >&2
  exit 1
fi

if [[ ! -f "$DATASET_DIR/train_split.csv" || ! -f "$DATASET_DIR/val_split.csv" || ! -f "$DATASET_DIR/test_split.csv" ]]; then
  echo "Generuje split treningowy z augmentacja geometryczna: $DATASET_DIR"
  "$PYTHON" scripts/prepare_geometric_augmented_train_split.py \
    --baseline-splits-dir "$BASELINE_SPLITS_DIR" \
    --output-dir "$DATASET_DIR" \
    --target-per-class "$TARGET_PER_CLASS" \
    --image-size "$IMAGE_SIZE" \
    --seed "$SEED"
else
  echo "Uzywam istniejacego splitu z augmentacja: $DATASET_DIR"
fi

echo "Start treningu na 15 epok:"
echo "  splits:  $DATASET_DIR"
echo "  output:  $OUTPUT_DIR"
echo "  epochs:  $EPOCHS"

"$PYTHON" scripts/train_baselines.py \
  --splits-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --early-stopping-patience "$PATIENCE" \
  --num-workers "$NUM_WORKERS"
