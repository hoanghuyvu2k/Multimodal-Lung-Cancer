"""21 tổ hợp bài báo × {base, +NLP} × OvO. Lưu tăng dần vào results/ovo_nlp_combos.json.

Đối chiếu với paper_combos.json["ovo"] (OvO base, không NLP) và nlp_combos.json (uniform_avg,
base/+Labs/+NLP) — trả lời câu hỏi OvO có tận dụng NLP-clinical tốt hơn/kém hơn uniform_avg không.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluate import load_result, save_result   # noqa: E402
from common.data_setup import load_all                  # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality     # noqa: E402
from nlp_verify.nlp_run import run_variant                # noqa: E402
from nlp_verify.run_nlp import PAPER                       # noqa: E402
from allcombo.run_ovo_singlerun import train_ovo_safe       # noqa: E402


def main():
    ctx = load_all()
    add_nlp_modality(ctx)
    try:
        payload = load_result("ovo_nlp_combos")
    except Exception:
        payload = {"combos": {}}
    done = payload["combos"]

    print(f"{'#':>3} {'combo':<26}{'OvO base':>16}{'OvO+NLP':>16}")
    for i, (label, mods) in enumerate(PAPER, 1):
        if label in done:
            e = done[label]
        else:
            base = [m for m in mods if m != "cnl_dem_labs"]
            variants = {"base": base, "nlp": base + ["cnl_nlp"]}
            e = {v: run_variant(m, ctx, train_fn=train_ovo_safe) for v, m in variants.items()}
            done[label] = e
            save_result("ovo_nlp_combos", payload)   # lưu tăng dần, an toàn nếu bị ngắt giữa chừng
        b, n = e["base"], e["nlp"]
        print(f"{i:>3} {label:<26}{b['mean']:.4f}±{b['sd']:.3f}".ljust(42) +
              f"{n['mean']:.4f}±{n['sd']:.3f}", flush=True)
    save_result("ovo_nlp_combos", payload)
    print("XONG OvO+NLP")


if __name__ == "__main__":
    main()
