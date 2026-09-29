"""
Step 7 — Ablation KIẾN TRÚC HIỆN TẠI (đổi hướng so với plan gốc).

Plan gốc định ablation "phương pháp thắng cuộc", nhưng cả A và B đều bị loại nên không còn
đối tượng. Câu hỏi có giá trị hơn: trong chính kiến trúc đang dùng, thành phần nào đóng góp?

Bốn biến thể, tách dần cơ chế attention ra khỏi mô hình:

  original            : gate bật   -> total = Σ aᵢ·tanh(rᵢ),  aᵢ học được, Σaᵢ = 1
  original_gate_off   : gate tắt   -> total = Σ tanh(rᵢ)      (TỔNG, không chuẩn hoá)
  uniform_avg         : không gate -> total = Σ (maskᵢ/Σmask)·tanh(rᵢ)  (TRUNG BÌNH)
  original_cross      : gate bật + cross_modality_enabled=True

`original_gate_off` dùng cờ `attention_gate_enabled=False` có sẵn trong lung_helpers, nên
không cần code mô hình mới — chỉ khác đúng một dòng trong forward.

So sánh then chốt: original vs uniform_avg cô lập ĐÓNG GÓP CỦA ATTENTION, vì hai mô hình
dùng chung hoàn toàn phần risk head và chỉ khác cách tính aᵢ.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import benchmarks as B          # noqa: E402
from common.data_setup import load_all      # noqa: E402
from common.evaluate import (               # noqa: E402
    load_result, load_scores, paired_bootstrap, run_repeated_cv,
    save_result, save_scores, strip_internals,
)

from lung_helpers import train              # noqa: E402

VARIANTS = {
    "original_gate_off": {"attention_gate_enabled": False},
    "original_cross": {"cross_modality_enabled": True},
}


def main():
    ctx = load_all()
    payload = {"variants": VARIANTS, "methods": {}, "paired": {}}

    for bm_key in B.BENCHMARKS:
        print(f"\n{'=' * 70}\n{bm_key} — {B.BENCHMARKS[bm_key]['name']}\n{'=' * 70}",
              flush=True)

        for vname, extra in VARIANTS.items():
            print(f"  [{vname}]", flush=True)
            res = run_repeated_cv(train, bm_key, ctx, extra_params=extra)
            save_scores(vname, bm_key, res)
            payload["methods"].setdefault(vname, {})[bm_key] = strip_internals(res)
            print(f"  {bm_key} {vname:<18}: {res['mean_auc']:.4f} ± {res['sd_auc']:.4f}",
                  flush=True)

        # Ghép cặp mọi biến thể với original (đã lưu score từ Step 4)
        ref = load_scores("original", bm_key)
        for vname in list(VARIANTS) + ["uniform_avg"]:
            cur = load_scores(vname, bm_key)
            payload["paired"].setdefault(f"{vname}_vs_original", {})[bm_key] = dict(
                zip(("delta_auc", "ci_low", "ci_high", "p_value"),
                    paired_bootstrap(cur, ref)))

        save_result("ablation", payload)

    base = load_result("method_A")["methods"]
    prev = load_result("method_B")["methods"]

    print(f"\n\n{'=' * 104}\nABLATION KIẾN TRÚC (mean ± sd, 5 seeds × 10-fold)\n{'=' * 104}")
    print(f"{'BM':<5} {'original':>16} {'gate_off (tổng)':>18} {'uniform_avg (TB)':>18} "
          f"{'cross_modality':>18}")
    print("-" * 104)
    for bm_key in B.BENCHMARKS:
        row = [base["original"][bm_key],
               payload["methods"]["original_gate_off"][bm_key],
               prev["uniform_avg"][bm_key],
               payload["methods"]["original_cross"][bm_key]]
        print(f"{bm_key:<5} " + " ".join(
            f"{r['mean_auc']:>7.4f}±{r['sd_auc']:<8.4f}" for r in row))
    print("=" * 104)

    print("\nĐÓNG GÓP CỦA ATTENTION — uniform_avg (không attention) vs original (có):")
    for bm_key in B.BENCHMARKS:
        d = payload["paired"]["uniform_avg_vs_original"][bm_key]
        print(f"  {bm_key}: {d['delta_auc']:+.4f} "
              f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] p={d['p_value']:.3f}")

    print("\nTẮT GATE (tổng thay vì trung bình) vs original:")
    for bm_key in B.BENCHMARKS:
        d = payload["paired"]["original_gate_off_vs_original"][bm_key]
        print(f"  {bm_key}: {d['delta_auc']:+.4f} "
              f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] p={d['p_value']:.3f}")

    print("\nBẬT CROSS-MODALITY vs original:")
    for bm_key in B.BENCHMARKS:
        d = payload["paired"]["original_cross_vs_original"][bm_key]
        print(f"  {bm_key}: {d['delta_auc']:+.4f} "
              f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] p={d['p_value']:.3f}")

    save_result("ablation", payload)


if __name__ == "__main__":
    main()
