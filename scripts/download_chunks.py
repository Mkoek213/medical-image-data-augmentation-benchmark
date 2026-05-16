"""
Downloads eligible images from the NIH Chest X-rays archive on Kaggle.
Uses HTTP range requests against the GCS-hosted archive.zip so only
eligible files (single-label, non-"No Finding", not already present)
are actually transferred — ~25% of the 45 GB archive.
"""

import csv
import io
import json
import time
import zipfile
from pathlib import Path

import requests

DATASET_SLUG = "nih-chest-xrays/data"
KAGGLE_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ENTRY_CSV = REPO_ROOT / "Data_Entry_2017.csv"
LABELS_CSV = REPO_ROOT / "data" / "labels.csv"
IMAGES_DIR = REPO_ROOT / "data" / "images"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ── HTTP range-request backed file object ────────────────────────────────────

class RemoteFile(io.RawIOBase):
    """Seekable file-like object backed by HTTP range requests."""

    CHUNK = 256 * 1024  # 256 KB read buffer

    def __init__(self, url: str, session: requests.Session):
        self._url = url
        self._s = session
        self._pos = 0
        self._size: int | None = None

    # ---- size -----------------------------------------------------------------
    @property
    def size(self) -> int:
        if self._size is None:
            r = self._s.head(self._url)
            r.raise_for_status()
            self._size = int(r.headers["content-length"])
        return self._size

    # ---- io.RawIOBase interface ------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def readinto(self, b: bytearray) -> int:
        n = len(b)
        if n == 0 or self._pos >= self.size:
            return 0
        end = min(self._pos + n - 1, self.size - 1)
        data = self._fetch(self._pos, end)
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self.size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    # ---- helpers ---------------------------------------------------------------
    def _fetch(self, start: int, end: int, retries: int = 5) -> bytes:
        for attempt in range(retries):
            try:
                r = self._s.get(
                    self._url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=60,
                )
                if r.status_code in (200, 206):
                    return r.content
                if r.status_code == 416:  # range not satisfiable
                    return b""
                r.raise_for_status()
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return b""


# ── Kaggle helpers ────────────────────────────────────────────────────────────

def get_archive_url() -> str:
    creds = json.loads(KAGGLE_JSON.read_text())
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {creds['key']}"
    r = s.get(KAGGLE_DOWNLOAD_URL, allow_redirects=False, timeout=30)
    r.raise_for_status()
    location = r.headers.get("location")
    if not location:
        raise RuntimeError(f"No redirect from Kaggle — got {r.status_code}: {r.text[:200]}")
    return location


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_label_index() -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    with open(DATA_ENTRY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            fname = row["Image Index"].strip()
            label = row["Finding Labels"].strip()
            pid = str(row["Patient ID"]).strip()
            index[fname] = (label, pid)
    return index


def load_existing() -> set[str]:
    if not LABELS_CSV.exists():
        return set()
    with open(LABELS_CSV, newline="") as f:
        return {row["image"].strip() for row in csv.DictReader(f)}


KEEP_CLASSES = {
    "Nodule", "Pneumothorax", "Mass", "Consolidation", "Pleural_Thickening",
    "Cardiomegaly", "Emphysema", "Fibrosis", "Edema", "Pneumonia", "Hernia",
}


def is_eligible(label: str) -> bool:
    return label in KEEP_CLASSES


def append_labels(rows: list[tuple[str, str, str]]) -> None:
    write_header = not LABELS_CSV.exists() or LABELS_CSV.stat().st_size == 0
    with open(LABELS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["image", "label", "patient_id"])
        writer.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading Data_Entry_2017.csv...")
    label_index = load_label_index()
    print(f"  {len(label_index):,} entries.")

    existing = load_existing()
    print(f"  {len(existing):,} already downloaded, skipping.")

    want: dict[str, tuple[str, str]] = {
        fname: entry
        for fname, entry in label_index.items()
        if is_eligible(entry[0]) and fname not in existing
    }
    print(f"  {len(want):,} images to download.\n")

    if not want:
        print("Nothing to do.")
        return

    print("Getting archive URL from Kaggle...")
    gcs_url = get_archive_url()
    print("  Got signed URL (valid 72 h).\n")

    session = requests.Session()
    # GCS signed URL already has auth baked in; no extra headers needed

    print("Opening remote zip (reading central directory)...")
    remote = RemoteFile(gcs_url, session)
    buffered = io.BufferedReader(remote, buffer_size=512 * 1024)
    zf = zipfile.ZipFile(buffered)

    # Build a map from bare filename → ZipInfo for fast lookup
    name_to_info: dict[str, zipfile.ZipInfo] = {}
    for info in zf.infolist():
        bare = Path(info.filename).name
        if bare.endswith(".png"):
            name_to_info[bare] = info
    print(f"  Archive contains {len(name_to_info):,} PNG entries.")

    to_fetch = {fname: info for fname, info in name_to_info.items() if fname in want}
    print(f"  {len(to_fetch):,} to extract.\n")

    new_rows: list[tuple[str, str, str]] = []
    ok = fail = 0

    for i, (fname, info) in enumerate(sorted(to_fetch.items()), 1):
        label, pid = want[fname]
        print(f"  [{i}/{len(to_fetch)}] {fname}  [{label}]", end="  ", flush=True)
        try:
            data = zf.read(info)
            dest = IMAGES_DIR / fname
            dest.write_bytes(data)
            new_rows.append((fname, label, pid))
            ok += 1
            print("OK")
        except Exception as exc:
            fail += 1
            print(f"FAILED: {exc}")

        # Flush labels every 50 images so progress is safe to interrupt
        if len(new_rows) % 50 == 0 and new_rows:
            append_labels(new_rows)
            new_rows = []

    if new_rows:
        append_labels(new_rows)

    print(f"\nDone. {ok} saved, {fail} failed.")


if __name__ == "__main__":
    main()
