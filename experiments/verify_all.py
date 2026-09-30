"""Chạy nối tiếp toàn bộ các phase tính toán của kế hoạch kiểm chứng số liệu bài báo.

    document/2026-09-29_verify-paper-numbers/ke-hoach-kiem-chung.md

Thứ tự:  1a  run_paper_combos      (uniform + DyAM, 21 combo, 5-seed)
         1b  run_paper_singlerun   (cột single-run cho uniform + DyAM)
         1c  run_paper_ovo         (OvO 5-seed + single-run + paired bootstrap)
         2   run_ovo_nlp           (OvO + NLP, 21 combo)
         3   verify_nlp_variants   (Bảng 5 & 6 — 7 biến thể encoding lâm sàng)
         4   survival_analysis_nlp (Bảng 7, 8 + external validation)

Mỗi phase chạy trong try/except: một phase hỏng thì vẫn chạy tiếp các phase sau,
tổng kết ở cuối. Chạy từ code/:

    PYTHONIOENCODING=utf-8 python experiments/verify_all.py [ten_phase ...]
"""
import runpy
import sys
import time
import traceback
from pathlib import Path

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
for _p in (str(_ROOT), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- Vá scipy >= 1.14 --------------------------------------------------------
# Các lần chạy gốc dùng scipy < 1.14, bản đó không raise khi brent không bracket
# được trên cột radiomics suy biến. scipy 1.14 raise BracketError -> script chết
# giữa chừng. Khôi phục hành vi cũ: cột nào lỗi thì lambda = 1 (gần identity).
# Cùng cách vá đã dùng trong experiments/nlp_verify/nlp_run.py, KHÔNG sửa lung_helpers.
import sklearn.preprocessing._data as _skd                       # noqa: E402

_orig_yj = _skd.PowerTransformer._yeo_johnson_optimize


def _safe_yj(self, x):
    try:
        return _orig_yj(self, x)
    except Exception:
        return 1.0


_skd.PowerTransformer._yeo_johnson_optimize = _safe_yj

RESULTS = _ROOT / "experiments" / "results"
REF = RESULTS / "_ref_2026-09-29"


def _phase_1a():
    from allcombo import run_paper_combos
    run_paper_combos.main()


def _phase_1b():
    from allcombo import run_paper_singlerun
    run_paper_singlerun.main()


def _phase_1c():
    from allcombo import run_paper_ovo
    run_paper_ovo.main()


def _phase_2():
    # run_ovo_nlp bỏ qua combo đã có trong json -> phải dọn file cũ, nếu không
    # nó chỉ in lại số cũ chứ không train lại.
    live = RESULTS / "ovo_nlp_combos.json"
    if live.exists():
        REF.mkdir(parents=True, exist_ok=True)
        backup = REF / "ovo_nlp_combos.json"
        if not backup.exists():
            live.replace(backup)
        else:
            live.unlink()
        print(f"[phase 2] da don {live.name} de chay lai tu dau", flush=True)
    from nlp_verify import run_ovo_nlp
    run_ovo_nlp.main()


def _phase_3():
    from nlp_verify import verify_nlp_variants
    verify_nlp_variants.main()


def _phase_4():
    runpy.run_path(str(_ROOT / "survival_analysis_nlp.py"), run_name="__main__")


PHASES = [
    ("1a", "run_paper_combos      (uniform + DyAM, 21 combo, 5-seed)", _phase_1a),
    ("1b", "run_paper_singlerun   (single-run uniform + DyAM)",        _phase_1b),
    ("1c", "run_paper_ovo         (OvO 5-seed + single-run)",          _phase_1c),
    ("2",  "run_ovo_nlp           (OvO + NLP, 21 combo)",              _phase_2),
    ("3",  "verify_nlp_variants   (Bang 5 & 6)",                       _phase_3),
    ("4",  "survival_analysis_nlp (Bang 7, 8 + external)",             _phase_4),
]


def main(argv):
    wanted = set(argv[1:]) or {p[0] for p in PHASES}
    report = []
    for key, desc, fn in PHASES:
        if key not in wanted:
            continue
        print(f"\n{'=' * 90}\nPHASE {key} — {desc}\n{'=' * 90}", flush=True)
        t0 = time.time()
        try:
            fn()
            report.append((key, "OK", time.time() - t0, ""))
        except Exception as exc:                                  # noqa: BLE001
            traceback.print_exc()
            report.append((key, "LOI", time.time() - t0, f"{type(exc).__name__}: {exc}"))
        print(f"\n[phase {key}] {report[-1][1]} sau {report[-1][2] / 60:.1f} phut", flush=True)

    print(f"\n{'=' * 90}\nTONG KET\n{'=' * 90}")
    for key, status, secs, err in report:
        print(f"  phase {key:<3} {status:<4} {secs / 60:6.1f} phut  {err}")
    return 0 if all(r[1] == "OK" for r in report) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
