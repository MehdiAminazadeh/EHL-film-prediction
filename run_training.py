
import os
import time
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path

# Torch (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# RegKit imports
# NOTE: this uses the new Config object from utils.config_parser we defined earlier.
from utils.config_parser import load_config         # returns Config with .data/.experiment/.logging/.active
from utils.logger import ExperimentLogger
from pipelines.train_core import train_model        # strategy router




def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic behavior (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(
        description="RegKit — Unified runner for tabular ML models"
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to unified YAML config (default: configs/config.yaml)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override experiment.model_name (e.g., xgboost, lightgbm, catboost, ridge, ...)",
    )
    p.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Override experiment.num_runs in config",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override data.seed (and global RNGs). If omitted, uses config.data.seed",
    )
    p.add_argument(
        "--verbose",
        type=int,
        default=None,
        help="Override logging.verbosity (0=silent, 1=info, 2=debug)",
    )
    return p.parse_args()




def main():
    args = parse_args()

    # Load merged config object
    cfg = load_config(args.config)  # -> Config with .data, .experiment, .logging, .active

    # CLI overrides
    if args.model is not None:
        cfg.active["model_name"] = args.model.lower()
        cfg.experiment["model_name"] = args.model.lower()  # keep mirror if you store it
    if args.num_runs is not None:
        cfg.experiment["num_runs"] = int(args.num_runs)
    if args.verbose is not None:
        cfg.logging["verbosity"] = int(args.verbose)
    if args.seed is not None:
        cfg.data["seed"] = int(args.seed)

    # Reproducibility
    seed = int(cfg.data.get("seed", 42))
    set_global_seed(seed)

    # Device report
    device = get_device()
    if TORCH_AVAILABLE:
        print(f"📡 Using device: {device} | torch {torch.__version__}")
    else:
        print("📡 Using device: cpu | torch not installed")

    # Load data
    csv_path = "./data/" + cfg.data["csv_path"]
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)

    # Summary banner
    model_name = cfg.active["model_name"]
    num_runs = int(cfg.experiment.get("num_runs", 1))
    verbosity = int(cfg.logging.get("verbosity", 1))
    log_file = cfg.logging.get("log_to_file", "experiment_log.xlsx")

    print("\n🔧 Running Configuration Summary")
    print(f"• Model: {model_name}")
    print(f"• Config: {os.path.abspath(args.config)}")
    print(f"• Num Runs: {num_runs}")
    print(f"• Seed: {seed}")
    print(f"• Logging → file: {log_file} | verbosity: {verbosity}")
    print("────────────────────────────────────────────")

    # Logger
    logger = ExperimentLogger(
        enabled=cfg.logging.get("enable", True),
        excel_path=log_file,
        verbosity=verbosity,
    )

    all_results = []
    t0 = time.time()

    try:
        for run_idx in range(1, num_runs + 1):
            print(f"\n🚀 Run {run_idx}/{num_runs} — model={model_name}")
            run_start = time.time()

            # Delegate to train_core (returns dict with metrics & params)
            result = train_model(
                cfg=cfg,
                data_cfg=cfg.data,
                active=cfg.active,
                df=df,
                logger=logger,
            )

            run_time = time.time() - run_start
            result = result or {}
            result.setdefault("Run", run_idx)
            result.setdefault("Time_sec", round(run_time, 2))
            all_results.append(result)

            # Minimal run summary
            cv_r2 = result.get("CV_R2") or result.get("R2")
            overfit = result.get("Overfitting", None)
            if cv_r2 is not None:
                print(f"✅ Run {run_idx} finished | CV_R2={cv_r2} | Overfitting={overfit} | Time={run_time:.2f}s")
            else:
                print(f"✅ Run {run_idx} finished | Time={run_time:.2f}s")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Printing partial results…")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        raise
    finally:
        total = time.time() - t0
        print("\n🏁 All Runs Summary")
        if not all_results:
            print("No results collected.")
        else:
            best = None
            for r in all_results:
                ridx = r.get("Run", "?")
                cv_r2 = r.get("CV_R2") or r.get("R2")
                overfit = r.get("Overfitting", "N/A")
                print(f"• Run {ridx}: CV_R2={cv_r2} | Overfitting={overfit} | Time={r.get('Time_sec','?')}s")
                if cv_r2 is not None and (best is None or cv_r2 > (best.get("CV_R2") or -1e9)):
                    best = r

            if best is not None:
                print("\n🏆 Best Run")
                print(f"• Run: {best.get('Run')}")
                print(f"• CV_R2: {best.get('CV_R2', best.get('R2'))}")
                print(f"• Overfitting: {best.get('Overfitting')}")
                if "Best_Params" in best:
                    print(f"• Best Params: {best['Best_Params']}")

        # Persist CSV alongside Excel
        try:
            if logger.enabled:
                logger.flush()
                out_csv = Path(logger.path).with_suffix(".csv")
            else:
                out_csv = Path("./experiment_log.csv")
            pd.DataFrame(all_results).to_csv(out_csv, index=False)
            print(f"\n🗂  Summary CSV: {out_csv.resolve()}")
        except Exception:
            pass


        print(f"\n⏱️ Total elapsed: {total:.2f}s\n")


if __name__ == "__main__":
    main()