#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
STYLEGAN_DIR="${STYLEGAN_DIR:-stylegan2-ada-pytorch}"

# Domyslnie trenujemy osobny generator dla kazdej klasy ponizej TARGET_PER_CLASS.
# Ustaw CLASS_NAME=Hernia, jesli chcesz uruchomic tylko jedna klase.
CLASS_NAME="${CLASS_NAME:-ALL}"
TARGET_PER_CLASS="${TARGET_PER_CLASS:-3000}"
# The classifier resizes every image to 512px during loading, so training the
# GAN at 256px is the practical default for this benchmark environment.
RESOLUTION="${RESOLUTION:-256}"
KIMG="${KIMG:-40}"
AUGPIPE="${AUGPIPE:-color}"
SNAP="${SNAP:-5}"
TRAIN_SPLIT="${TRAIN_SPLIT:-outputs/advanced_baselines_full/train_split.csv}"
RETRAIN_STYLEGAN="${RETRAIN_STYLEGAN:-0}"

if [[ "$CLASS_NAME" == "ALL" ]]; then
  RUN_ROOT="${RUN_ROOT:-outputs/stylegan2_all_minority_${RESOLUTION}px}"
else
  CLASS_SLUG="$(echo "$CLASS_NAME" | tr '[:upper:] ' '[:lower:]_')"
  RUN_ROOT="${RUN_ROOT:-outputs/stylegan2_${CLASS_SLUG}_${RESOLUTION}px}"
fi

PLAN_FILE="${PLAN_FILE:-${RUN_ROOT}/minority_plan.tsv}"
MANIFEST="${MANIFEST:-${RUN_ROOT}/synthetic_manifest.csv}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Nie znaleziono interpretera: $PYTHON" >&2
  exit 1
fi

if [[ ! -d "$STYLEGAN_DIR" ]]; then
  echo "Brakuje repozytorium StyleGAN2-ADA: $STYLEGAN_DIR" >&2
  echo "Uruchom z katalogu projektu:" >&2
  echo "  git clone https://github.com/NVlabs/stylegan2-ada-pytorch" >&2
  echo "  pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3" >&2
  exit 1
fi

echo "Sprawdzam kompatybilnosc StyleGAN2-ADA z aktualnym PyTorch"
"$PYTHON" - "$STYLEGAN_DIR/torch_utils/misc.py" "$STYLEGAN_DIR/torch_utils/ops/upfirdn2d.py" <<'PY'
import sys
from pathlib import Path

misc_path = Path(sys.argv[1])
misc_text = misc_path.read_text(encoding="utf-8")
misc_old = "super().__init__(dataset)"
misc_new = "super().__init__()"
if misc_old in misc_text:
    misc_path.write_text(misc_text.replace(misc_old, misc_new), encoding="utf-8")
    print(f"patched: {misc_path}")
else:
    print(f"ok: {misc_path}")

upfirdn_path = Path(sys.argv[2])
upfirdn_text = upfirdn_path.read_text(encoding="utf-8")
upfirdn_old = """def _init():
    global _inited, _plugin
    if not _inited:
        sources = ['upfirdn2d.cpp', 'upfirdn2d.cu']"""
upfirdn_new = """def _init():
    global _inited, _plugin
    if not _inited:
        _inited = True
        sources = ['upfirdn2d.cpp', 'upfirdn2d.cu']"""
if upfirdn_old in upfirdn_text:
    upfirdn_path.write_text(upfirdn_text.replace(upfirdn_old, upfirdn_new), encoding="utf-8")
    print(f"patched: {upfirdn_path}")
else:
    print(f"ok: {upfirdn_path}")
PY

mkdir -p "$RUN_ROOT"

echo "Przygotowuje plan nadproblkowania StyleGAN2"
echo "ADA augpipe: $AUGPIPE"
echo "Resolution : ${RESOLUTION}px"
echo "KIMG       : $KIMG"
echo "Snapshot   : every $SNAP ticks"
"$PYTHON" - "$TRAIN_SPLIT" "$TARGET_PER_CLASS" "$CLASS_NAME" "$PLAN_FILE" <<'PY'
import sys
from pathlib import Path
import pandas as pd

train_split = Path(sys.argv[1])
target = int(sys.argv[2])
class_name = sys.argv[3]
plan_file = Path(sys.argv[4])

df = pd.read_csv(train_split)
counts = df["label"].value_counts().sort_index()

rows = []
if class_name == "ALL":
    for label, count in counts.items():
        need = max(0, target - int(count))
        if need > 0:
            rows.append((label, int(count), need))
else:
    count = int(counts.get(class_name, 0))
    need = max(0, target - count)
    if need == 0:
        print(f"Klasa {class_name} ma juz {count} probek, nic do generowania.")
    else:
        rows.append((class_name, count, need))

plan_file.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=["label", "current_count", "need"]).to_csv(
    plan_file, sep="\t", index=False
)
print(pd.DataFrame(rows, columns=["label", "current_count", "need"]).to_string(index=False))
PY

if [[ ! -s "$PLAN_FILE" ]]; then
  echo "Plan jest pusty, nie ma klas do nadproblkowania." >&2
  exit 1
fi

echo "image,label,patient_id,image_path" > "$MANIFEST"

CFG="auto"
case "$RESOLUTION" in
  256) CFG="paper256" ;;
  512) CFG="paper512" ;;
  1024) CFG="paper1024" ;;
esac

tail -n +2 "$PLAN_FILE" | while IFS=$'\t' read -r LABEL CURRENT_COUNT NEED; do
  CLASS_SLUG="$(echo "$LABEL" | tr '[:upper:] ' '[:lower:]_')"
  CLASS_ROOT="${RUN_ROOT}/${CLASS_SLUG}"
  STAGING_DIR="${CLASS_ROOT}/staging_train_images"
  DATASET_ZIP="${CLASS_ROOT}/dataset_${CLASS_SLUG}_${RESOLUTION}px.zip"
  TRAIN_OUT_DIR="${CLASS_ROOT}/training"
  GENERATED_DIR="${CLASS_ROOT}/generated_samples"

  mkdir -p "$CLASS_ROOT" "$TRAIN_OUT_DIR" "$GENERATED_DIR"

  echo "========================================================================"
  echo "Klasa: $LABEL | obecnie: $CURRENT_COUNT | generuje: $NEED"
  echo "Etap 1/4: staging obrazow treningowych"
  "$PYTHON" scripts/prepare_stylegan2_class_images.py \
    --train-split "$TRAIN_SPLIT" \
    --class-name "$LABEL" \
    --output-dir "$STAGING_DIR" \
    --resolution "$RESOLUTION"

  STAGED_COUNT="$(find "$STAGING_DIR" -maxdepth 1 -type f -name '*.png' | wc -l)"
  if [[ "$STAGED_COUNT" -eq 0 ]]; then
    echo "Staging jest pusty dla klasy $LABEL: $STAGING_DIR" >&2
    exit 1
  fi

  echo "Etap 2/4: budowa datasetu StyleGAN2 zip z $STAGED_COUNT obrazow"
  rm -f "$DATASET_ZIP"
  "$PYTHON" "$STYLEGAN_DIR/dataset_tool.py" \
    --source="$STAGING_DIR" \
    --dest="$DATASET_ZIP" \
    --width="$RESOLUTION" \
    --height="$RESOLUTION"

  LATEST_NETWORK="$(find "$TRAIN_OUT_DIR" -name 'network-snapshot-*.pkl' | sort | tail -n 1 || true)"
  if [[ "$RETRAIN_STYLEGAN" == "1" || -z "$LATEST_NETWORK" ]]; then
    echo "Etap 3/4: trening StyleGAN2-ADA dla klasy $LABEL"
    "$PYTHON" "$STYLEGAN_DIR/train.py" \
      --outdir="$TRAIN_OUT_DIR" \
      --data="$DATASET_ZIP" \
      --gpus=1 \
      --cfg="$CFG" \
      --kimg="$KIMG" \
      --mirror=0 \
      --aug=ada \
      --augpipe="$AUGPIPE" \
      --target=0.6 \
      --snap="$SNAP" \
      --metrics=none
    LATEST_NETWORK="$(find "$TRAIN_OUT_DIR" -name 'network-snapshot-*.pkl' | sort | tail -n 1)"
  else
    echo "Etap 3/4: uzywam istniejacego checkpointu: $LATEST_NETWORK"
  fi

  if [[ -z "$LATEST_NETWORK" ]]; then
    echo "Nie znaleziono checkpointu network-snapshot-*.pkl w $TRAIN_OUT_DIR" >&2
    exit 1
  fi

  echo "Etap 4/4: generowanie $NEED obrazow syntetycznych dla klasy $LABEL"
  rm -rf "$GENERATED_DIR"
  mkdir -p "$GENERATED_DIR"
  "$PYTHON" "$STYLEGAN_DIR/generate.py" \
    --outdir="$GENERATED_DIR" \
    --trunc=0.7 \
    --seeds="0-$((NEED - 1))" \
    --network="$LATEST_NETWORK"

  "$PYTHON" - "$LABEL" "$GENERATED_DIR" "$MANIFEST" <<'PY'
import sys
from pathlib import Path

label = sys.argv[1]
generated_dir = Path(sys.argv[2])
manifest = Path(sys.argv[3])

with manifest.open("a", encoding="utf-8") as f:
    for idx, path in enumerate(sorted(generated_dir.glob("*.png"))):
        f.write(f"{path.name},{label},synthetic_{label}_{idx:06d},{path.resolve()}\n")
PY
done

echo "========================================================================"
echo "Gotowe generowanie StyleGAN2 dla klas mniejszosciowych."
echo "Plan     : $PLAN_FILE"
echo "Manifest : $MANIFEST"
