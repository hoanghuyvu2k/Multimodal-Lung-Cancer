"""Pha 1 — 21 tổ hợp bài báo x {base, +NLP} x DyAM (đối chiếu OvO ở results/ovo_nlp_combos.json, cùng đường chạy run_variant).
Chạy song song theo phần: --part=i/n; lưu results/dyam_nlp_combos_part{i}.json.

    PYTHONIOENCODING=utf-8 python experiments/nlp_verify/run_dyam_nlp.py --part=0/2
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data_setup import load_all                  # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality     # noqa: E402
from nlp_verify.nlp_run import run_variant                # noqa: E402
from nlp_verify.run_nlp import PAPER                       # noqa: E402
from allcombo.run_combo import train as train_dyam         # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    part = next((a.split("=")[1] for a in sys.argv if a.startswith("--part=")), "0/1")
    i, n = (int(x) for x in part.split("/"))
    f = OUT / f"dyam_nlp_combos_part{i}.json"
    done = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    ctx = load_all()
    add_nlp_modality(ctx)
    for k, (label, mods) in enumerate(PAPER):
        if k % n != i or label in done:
            continue
        base = [m for m in mods if m != "cnl_dem_labs"]
        done[label] = {v: run_variant(m, ctx, train_fn=train_dyam) for v, m in {"base": base, "nlp": base + ["cnl_nlp"]}.items()}
        f.write_text(json.dumps(done, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        e = done[label]
        print(f"[dyam_nlp] {label}: base {e['base']['mean']:.4f} | +NLP {e['nlp']['mean']:.4f}", flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
