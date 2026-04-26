"""Download Higgs UCI CSV.gz with progress and resume.

Usage:
    python scripts/download_higgs.py
"""
from __future__ import annotations
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
DEST = Path(".data_cache/higgs/HIGGS.csv.gz")
EXPECTED_MIN = 2_500_000_000  # ~2.6GB minimum (actual ~2.81 GB)


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size >= EXPECTED_MIN:
        print(f"[higgs] already present ({DEST.stat().st_size/1e9:.2f} GB)")
        return 0

    if DEST.exists():
        print(f"[higgs] partial file at {DEST.stat().st_size/1e6:.1f} MB; removing")
        DEST.unlink()

    print(f"[higgs] downloading {URL}")
    t0 = time.time()
    last = [0.0]

    def progress(blocks: int, block_size: int, total: int) -> None:
        downloaded = blocks * block_size
        now = time.time()
        if now - last[0] > 5.0:
            last[0] = now
            mb = downloaded / 1e6
            mbps = downloaded / (now - t0) / 1e6
            pct = (100.0 * downloaded / total) if total > 0 else 0.0
            print(f"  [{pct:5.1f}%] {mb:8.1f} MB @ {mbps:5.2f} MB/s",
                  flush=True)

    urllib.request.urlretrieve(URL, DEST, reporthook=progress)
    print(f"[higgs] done: {DEST.stat().st_size/1e9:.2f} GB in "
          f"{time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
