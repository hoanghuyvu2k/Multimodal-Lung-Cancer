"""Step 1 smoke — Stacked Late-Fusion BM1, 1 seed. Kiểm tra chạy + AUC hợp lý + trọng số meta."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data_setup import load_all      # noqa: E402
from common.evaluate import run_repeated_cv  # noqa: E402
from stacked_fusion.model_stack import train_stack  # noqa: E402


def main():
    ctx = load_all()
    res = run_repeated_cv(train_stack, "BM1", ctx, seeds=[42])
    print(f"\nSMOKE stacked BM1 seed42: AUC = {res['mean_auc']:.4f}")


if __name__ == "__main__":
    main()
