#!/usr/bin/env python3
"""Execute a single .ipynb in-place (cells in order)."""
import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("notebook", type=Path)
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    path = args.notebook.resolve()
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 2
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3")
    client.execute()
    nbformat.write(nb, path)
    print(f"OK: {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
