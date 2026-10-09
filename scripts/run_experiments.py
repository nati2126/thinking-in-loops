"""Train every model needed for Phase 4 (skipping finished runs) and run all experiments.

Runs (each for every seed):
    ut            Universal Transformer, TBPTT K=3 (main model)
    same_param    1-layer transformer
    same_compute  8-layer transformer
    ut_k1         UT with TBPTT K=1   (ablation)
    ut_kall       UT with full backprop through all loops (ablation)

Outputs CSV + PNG files in results/ and results/summary.json.

Usage:
    python scripts/run_experiments.py                 # everything, default seeds
    python scripts/run_experiments.py --seeds 0       # one seed (faster)
    python scripts/run_experiments.py --retrain       # ignore existing checkpoints
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.config import DataConfig, ModelConfig, TrainConfig, resolve_device  # noqa: E402
from src.data.augment import symmetry_overlap  # noqa: E402
from src.data.dataset import build_datasets, load_split  # noqa: E402
from src.eval import experiments as ex  # noqa: E402
from src.eval.metrics import predict_all_loops  # noqa: E402
from src.training.checkpoint import load_checkpoint, load_model  # noqa: E402
from src.training.trainer import train  # noqa: E402

RUNS: dict[str, dict] = {
    "ut": {"model": "ut", "tbptt_k": 3},
    "same_param": {"model": "same_param"},
    "same_compute": {"model": "same_compute"},
    "ut_k1": {"model": "ut", "tbptt_k": 1},
    "ut_kall": {"model": "ut", "tbptt_k": 0},
}


def is_finished(ckpt_dir: Path, epochs: int) -> bool:
    last = ckpt_dir / "last.pt"
    return last.exists() and load_checkpoint(last)["epoch"] >= epochs - 1


def read_log(path: Path) -> list[dict[str, float]]:
    with path.open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--max_eval_loops", type=int, default=32)
    parser.add_argument("--retrain", action="store_true", help="retrain even if a finished checkpoint exists")
    parser.add_argument("--results_dir", type=str, default=TrainConfig.results_dir)
    parser.add_argument("--checkpoint_dir", type=str, default=TrainConfig.checkpoint_dir)
    args = parser.parse_args()

    data_cfg, model_cfg = DataConfig(), ModelConfig()
    base_train = TrainConfig(epochs=args.epochs, results_dir=args.results_dir, checkpoint_dir=args.checkpoint_dir)
    device = resolve_device(base_train.device)
    out_dir = Path(args.results_dir)
    build_datasets(data_cfg)
    test = load_split(data_cfg, "test")
    t_start = time.time()

    results: list[ex.RunResult] = []
    last_results: list[ex.RunResult] = []  # final-epoch checkpoints of looped models (extra analysis)
    for name in args.runs:
        for seed in args.seeds:
            run_name = f"{name}_s{seed}"
            cfg = dataclasses.replace(base_train, seed=seed, run_name=run_name, **RUNS[name])
            ckpt_dir = Path(cfg.checkpoint_dir) / run_name
            if args.retrain or not is_finished(ckpt_dir, cfg.epochs):
                cfg.resume = not args.retrain
                print(f"=== training {run_name} ===", flush=True)
                train(cfg, model_cfg, data_cfg)
            else:
                print(f"=== {run_name}: finished checkpoint found, skipping training ===")
            model, _ = load_model(ckpt_dir / "best.pt", device)
            preds = predict_all_loops(model, test.puzzles, args.max_eval_loops if model.is_recurrent else None)
            log = read_log(Path(cfg.results_dir) / "runs" / run_name / "log.csv")
            results.append(ex.RunResult(name, cfg.model, seed, model.num_parameters(),
                                        model_cfg.max_loops_train if model.is_recurrent else 1, preds, log))  # fmt: skip
            if model.is_recurrent:
                last_model, _ = load_model(ckpt_dir / "last.pt", device)
                last_preds = predict_all_loops(last_model, test.puzzles, model_cfg.max_loops_train)
                last_results.append(dataclasses.replace(results[-1], preds=last_preds))
                del last_model, last_preds
            del model, preds
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    seen = symmetry_overlap(load_split(data_cfg, "train").puzzles, test.puzzles)
    print(f"{int(seen.sum())}/{len(test)} test puzzles are a symmetry image of a training puzzle")
    summary = {
        "seeds": args.seeds,
        "test_seen_up_to_symmetry": int(seen.sum()),
        "epochs": args.epochs,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "exp1_comparison": ex.exp1_comparison(results, test, out_dir),
        "exp1b_novel_vs_seen": ex.exp1b_novel_vs_seen(results, test, seen, out_dir),
        "exp2_loops_vs_difficulty": ex.exp2_loops_vs_difficulty(results, test, out_dir),
        "exp2b_halting_threshold": ex.exp2b_halting_threshold(results, test, out_dir),
        "exp2c_checkpoint_choice": ex.exp2c_checkpoint_choice(results, last_results, test, out_dir),
        "exp3_test_time_compute": ex.exp3_test_time_compute(results, test, out_dir),
        "exp4_tbptt_ablation": ex.exp4_tbptt_ablation(results, test, out_dir, model_cfg, base_train.batch_size, device),
        "total_time_s": time.time() - t_start,  # this invocation only (skipped runs cost ~0)
        "total_train_time_s": sum(e["epoch_time_s"] for r in results for e in r.log),
    }
    ex.plot_training_curves(results, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for key in ("exp1_comparison", "exp4_tbptt_ablation"):
        print(f"\n{key}:")
        for row in summary[key]:
            print("  " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    print(f"\nall experiments done in {summary['total_time_s'] / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
