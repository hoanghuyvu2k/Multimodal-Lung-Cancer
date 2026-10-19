"""#20/#21 (cấu hình có Labs) x {DyAM, OvO} trong cùng đường chạy run_variant với bảng +NLP, để bảng 21 tổ hợp dùng MỘT đường chạy.
Lưu results/dyam_ovo_labs21.json.   PYTHONIOENCODING=utf-8 python experiments/nlp_verify/run_labs_base.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data_setup import load_all                  # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality     # noqa: E402
from nlp_verify.nlp_run import run_variant                # noqa: E402
from nlp_verify.run_nlp import PAPER                       # noqa: E402
from allcombo.run_combo import train as train_dyam         # noqa: E402
from allcombo.run_ovo_singlerun import train_ovo_safe       # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "dyam_ovo_labs21.json"


def main():
    ctx = load_all()
    add_nlp_modality(ctx)
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for label, mods in PAPER:
        if "cnl_dem_labs" not in mods:
            continue
        for name, fn in (("dyam", train_dyam), ("ovo", train_ovo_safe)):
            key = f"{label}|{name}"
            if key in res:
                continue
            res[key] = run_variant(mods, ctx, train_fn=fn)
            OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
            print(f"[labs21] {key}: {res[key]['mean']:.4f}", flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
