"""Chạy lại các thí nghiệm còn lại của bài trong MÔI TRƯỜNG THAM CHIẾU, ghi vào experiments/results/ref/ (không ghi đè kết quả Windows).

    run_ref.sh experiments/reference_env/run_paper_ref.py nlp20 dyam|ovo
    run_ref.sh experiments/reference_env/run_paper_ref.py miss [idx ...]
    run_ref.sh experiments/reference_env/run_paper_ref.py loso rad path gen pdl1 nlp
    run_ref.sh experiments/reference_env/run_paper_ref.py order t1|t2
    run_ref.sh experiments/reference_env/run_paper_ref.py surv      # verify_nlp_5seed + điểm nguy cơ cho tdAUC

Cấu hình và mã huấn luyện KHÔNG đổi so với script gốc; chỉ đổi thư mục ra. Bio_ClinicalBERT đọc từ cache parquet
(experiments/reference_env/make_bioclin_cache.py, tạo ở Windows vì môi trường tham chiếu không có transformers).
"""
import runpy
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
EXP, ROOT = _HERE.parents[1], _HERE.parents[2]
sys.path.insert(0, str(EXP)); sys.path.insert(0, str(ROOT))

REF = EXP / "results" / "ref"
(REF / "ovo_proof").mkdir(parents=True, exist_ok=True)
(REF / "survival_scores").mkdir(parents=True, exist_ok=True)
CACHE_BIO = Path("/mnt/d/repro_linux/cache/bioclin_embed_v2.parquet")


def main():
    kind, args = sys.argv[1], sys.argv[2:]
    # Cấu hình tham chiếu: bộ lọc radiomics GỐC ctx.rad_filters (không clean_rad_filters). Các script 20 seed gọi nlp_run.build_l1
    # (vốn làm sạch cột suy biến cho scipy>=1.14); môi trường tham chiếu (scipy 1.6.3) không cần, nên thay bằng bản không làm sạch.
    # Đặt TRƯỚC khi import các script. --clean giữ hành vi cũ (kết quả ở results/ref_cleanfilter).
    if "--clean" not in args:
        import nlp_verify.nlp_run as NR
        NR.build_l1 = lambda mods, ctx: {mods.index(n): ctx.rad_filters[NR.RAD[n]] for n in mods if n in NR.RAD}
    args = [a for a in args if a != "--clean"]
    if kind in ("loso", "miss", "order"):
        from archloop import harness as H
        H.OUT = REF / "ovo_proof"
    if kind == "nlp20":
        from nlp_verify import run_nlp20 as m
        m.OUT = REF / "ovo_proof"
        sys.argv = ["run_nlp20.py"] + args
        m.main()
    elif kind == "miss":
        sys.argv = ["run_miss_ovo.py"] + args
        runpy.run_path(str(EXP / "ovo_final" / "run_miss_ovo.py"), run_name="__main__")
    elif kind == "loso":
        sys.argv = ["run_loso_ovo.py"] + args
        runpy.run_path(str(EXP / "ovo_final" / "run_loso_ovo.py"), run_name="__main__")
    elif kind == "order":
        from validation import run_order_curve as OC
        OC.OUT = REF
        sys.argv = ["run_order_ovo.py"] + args
        runpy.run_path(str(EXP / "ovo_final" / "run_order_ovo.py"), run_name="__main__")
    elif kind == "surv":
        import pandas as pd
        from nlp_verify import verify_nlp_5seed as V
        from nlp_verify import verify_nlp_variants as VV
        from nlp_verify.verify_nlp_variants import slug

        def bio_from_cache(ctx, name="cnl_bioclin"):
            labs = ctx.modality_dict["cnl_dem_labs"]
            df = pd.read_parquet(CACHE_BIO)
            ctx.modality_MASK[name] = False
            ctx.modality_MASK.loc[df.index, name] = True
            ctx.modality_dict[name] = df

        V.add_bioclinbert = bio_from_cache
        order = [(a, v[0]) for a in V.ARMS for v in V.VARIANTS if (a, v[0]) in V.PAPER_AUC]
        orig, state = V.run_5seed, {"i": 0}

        def wrapped(ctx, clin, arm_mod, extra, no_scale_names):
            r = orig(ctx, clin, arm_mod, extra, no_scale_names)
            arm, name = order[state["i"]]; state["i"] += 1
            df = r["_scores"].copy(); df["label"] = r["_labels"]
            df.to_csv(REF / "survival_scores" / f"{arm}__{slug(name)}.csv")
            return r

        def save(name, payload):
            import json
            p = REF / f"{name}.json"
            p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print("   -> đã ghi", p)
            return p

        V.run_5seed, V.save_result = wrapped, save
        V.main()
    print("XONG")


if __name__ == "__main__":
    main()
