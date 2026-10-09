"""Generate the 4x4 Sudoku train/val/test splits and print summary statistics.

Usage:
    python scripts/generate_data.py [--overwrite true] [--seed 1234] ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.config import DataConfig, add_dataclass_args, dataclass_from_args  # noqa: E402
from src.data.dataset import SPLITS, build_datasets, split_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataclass_args(parser, DataConfig)
    parser.add_argument("--overwrite", action="store_true", help="regenerate even if files exist")
    args = parser.parse_args()
    cfg = dataclass_from_args(DataConfig, args)
    build_datasets(cfg, overwrite=args.overwrite)
    for split in SPLITS:
        with np.load(split_path(cfg, split)) as z:
            ne, bt = z["n_empty"], z["backtracks"]
            counts = {int(k): int((ne == k).sum()) for k in np.unique(ne)}
            print(
                f"{split:5s}: n={len(ne):6d} | empty-cell counts {counts} | "
                f"backtracks mean={bt.mean():.2f} max={bt.max()} | frac with 0 backtracks={(bt == 0).mean():.2f}"
            )


if __name__ == "__main__":
    main()
