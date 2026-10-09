"""Export a training checkpoint to a small weights-only file for the demo app.

Usage:
    python scripts/export_model.py                                   # checkpoints/ut_s0/best.pt -> app/model/ut.pt
    python scripts/export_model.py --checkpoint checkpoints/ut/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.config import REPO_ROOT  # noqa: E402
from src.training.checkpoint import load_checkpoint, load_model  # noqa: E402


def _repo_relative(path: Path) -> str:
    """Path relative to the repo (so no local absolute paths end up in the published file)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "ut_s0" / "best.pt"))
    parser.add_argument("--out", type=str, default=str(REPO_ROOT / "app" / "model" / "ut.pt"))
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    model, _ = load_model(args.checkpoint)  # validates that the weights load
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": ckpt["model_name"],
        "model_cfg": ckpt["model_cfg"],
        "model_state": {k: v.float().cpu() for k, v in ckpt["model_state"].items()},
        "val_metrics": ckpt.get("val_metrics", {}),
        "source": _repo_relative(Path(args.checkpoint)),
    }
    torch.save(payload, out)
    print(f"exported {ckpt['model_name']} ({model.num_parameters():,} params, epoch {ckpt.get('epoch')}) "
          f"-> {out} [{out.stat().st_size / 1024:.0f} KB]")


if __name__ == "__main__":
    main()
