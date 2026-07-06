#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval_engine import ensure_chroma_index


def main() -> None:
    try:
        count = ensure_chroma_index()
    except Exception as exc:
        print(f"Chroma index build skipped/failed: {exc}")
        return
    print(f"Chroma index built: collection=paperflow_corpus, count={count}")


if __name__ == "__main__":
    main()
