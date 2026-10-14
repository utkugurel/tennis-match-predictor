"""Integrity checks for the frozen raw-data snapshot in ``data/``.

The upstream repository (JeffSackmann/tennis_atp) is no longer available, so
the local copy is the only source for rebuilds. ``data/MANIFEST.json`` records
the size, SHA-256 and row count of every raw file; builds refuse to run if any
file is missing or differs from it.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MANIFEST_PATH = DATA_DIR / "MANIFEST.json"

SNAPSHOT_ID = "sackmann-atp-2024-12-30"
SNAPSHOT_META = {
    "snapshot_id": SNAPSHOT_ID,
    "source": "https://github.com/JeffSackmann/tennis_atp",
    "source_status": "Upstream repository returns 404 since mid-2026; local copy frozen.",
    "attribution": "Jeff Sackmann / Tennis Abstract (http://www.tennisabstract.com/)",
    "license": "CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)",
    "coverage": "ATP tour-level matches 1968-2024",
}
MANIFEST_PATTERNS = ["*.csv", "README.md", "matches_data_dictionary.txt"]
RESTORE_HINT = (
    "Restore the raw data snapshot into data/ as described in the README section "
    "'Raw data snapshot' (README.md#raw-data-snapshot)."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path: Path) -> int:
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def snapshot_files(data_dir: Path = DATA_DIR) -> List[Path]:
    files = set()
    for pattern in MANIFEST_PATTERNS:
        files.update(data_dir.glob(pattern))
    return sorted(files)


def describe_file(path: Path) -> Dict:
    return {
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "lines": count_rows(path),
    }


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> Dict:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run scripts/make_manifest.py once on the trusted snapshot. "
            + RESTORE_HINT
        )
    return json.loads(manifest_path.read_text())


def manifest_hash(manifest_path: Path = MANIFEST_PATH) -> str:
    return sha256_file(manifest_path)


def verify_snapshot(data_dir: Path = DATA_DIR, manifest_path: Path = MANIFEST_PATH) -> Dict:
    """Raise if any file listed in the manifest is missing or modified."""
    manifest = load_manifest(manifest_path)
    problems = []
    for name, expected in manifest["files"].items():
        path = data_dir / name
        if not path.exists():
            problems.append(f"missing: {name}")
            continue
        if path.stat().st_size != expected["size"]:
            problems.append(f"size mismatch: {name}")
            continue
        if sha256_file(path) != expected["sha256"]:
            problems.append(f"checksum mismatch: {name}")
    if problems:
        raise RuntimeError(
            "Raw data does not match data/MANIFEST.json:\n  " + "\n  ".join(problems)
            + "\n" + RESTORE_HINT
        )
    return manifest
