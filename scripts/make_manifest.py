"""Record checksums of the raw data snapshot in data/MANIFEST.json.

Run once on the trusted copy of the data. Re-running it on modified files
would bless those modifications, so it refuses to overwrite an existing
manifest unless --force is given.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis.snapshot import DATA_DIR, MANIFEST_PATH, SNAPSHOT_META, describe_file, snapshot_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    args = parser.parse_args()

    if MANIFEST_PATH.exists() and not args.force:
        sys.exit(f"{MANIFEST_PATH} already exists. Use --force to overwrite.")

    files = {path.name: describe_file(path) for path in snapshot_files(DATA_DIR)}
    if not files:
        sys.exit(f"No data files found in {DATA_DIR}.")

    manifest = {
        **SNAPSHOT_META,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {MANIFEST_PATH} with {len(files)} files.")


if __name__ == "__main__":
    main()
