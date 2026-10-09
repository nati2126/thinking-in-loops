"""Train any model by name.

Examples:
    python scripts/train.py --model ut
    python scripts/train.py --model same_compute --epochs 20
    python scripts/train.py --model ut --tbptt_k 1 --run_name ut_k1
    python scripts/train.py --model ut --resume true
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    DataConfig,
    ModelConfig,
    TrainConfig,
    add_dataclass_args,
    dataclass_from_args,
)
from src.training.trainer import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataclass_args(parser, TrainConfig)
    add_dataclass_args(parser, ModelConfig)
    add_dataclass_args(parser, DataConfig, prefix="data_")
    args = parser.parse_args()
    summary = train(
        dataclass_from_args(TrainConfig, args),
        dataclass_from_args(ModelConfig, args),
        dataclass_from_args(DataConfig, args, prefix="data_"),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
