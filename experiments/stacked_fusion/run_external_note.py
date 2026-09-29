"""Step 5 — External note (trung thực về giới hạn ngoài mẫu).

Cohort ngoài đều ĐƠN-modality (rad_valid=radiomics-only, path_valid=pathology-only) nên stacked
rút gọn về base learner của modality đó — KHÔNG test được phần fusion đa-modality ngoài mẫu.

Ghép external base learner (validation.json) với trọng số meta (stacked_weights.json, Step 3):
meta ĐỀ CAO radiomics (fail ngoài mẫu ≤0.46) và HẠ pathology (generalize 0.767) -> trọng số meta
KHÔNG làm mô hình robust hơn với domain shift; nếu có, còn phản-robust. Ghi vào stacked.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluate import load_result, save_result           # noqa: E402


def main():
    payload = load_result("stacked")
    val = load_result("validation")
    w = load_result("stacked_weights")

    path_ext = val["pathology"]["lr"]["auc"]
    rad_ext = val["radiomics"]["lr"]["auc"]
    bm1_w = w["benchmarks"]["BM1"]["weights"]

    note = {
        "limitation": "Cohort ngoài đều đơn-modality (rad_valid=radiomics-only, "
                      "path_valid=pathology-only). Stacked rút gọn về base learner của modality đó; "
                      "phần fusion đa-modality (đóng góp chính của mô hình) KHÔNG kiểm chứng được ngoài mẫu.",
        "external_base_lr": {
            "pathology": round(path_ext, 4),
            "radiomics": round(rad_ext, 4),
        },
        "meta_weight_vs_generalization": {
            "path_ihc_glcm_mean_abs": round(bm1_w["path_ihc_glcm"]["mean_abs"], 4),
            "rad_lesion_ln_mean": round(bm1_w["rad_lesion_ln"]["mean"], 4),
            "verdict": "PHẢN-ROBUST: meta đề cao radiomics (ext ≤0.46) và hạ pathology (ext 0.767≈0 "
                       "trọng số). Trọng số học từ OOF in-sample không mã hoá được generalization; "
                       "stacked KHÔNG robust hơn uniform_avg với domain shift.",
        },
    }
    payload["external_note"] = note
    save_result("stacked", payload)

    print(f"\n{'=' * 78}\nSTEP 5 — External note\n{'=' * 78}")
    print(f"External base-LR:  pathology = {path_ext:.4f} | radiomics = {rad_ext:.4f}")
    print(f"Meta BM1 trọng số: path_ihc_glcm mean|coef| = {bm1_w['path_ihc_glcm']['mean_abs']:.4f} (≈0)"
          f" | rad_lesion_ln mean = {bm1_w['rad_lesion_ln']['mean']:.4f} (cao nhất)")
    print("\nKết luận: meta đề cao đúng modality FAIL ngoài mẫu (radiomics) và bỏ modality GENERALIZE "
          "(pathology)\n-> trọng số stacked phản-robust; giới hạn: fusion đa-modality không test được ngoài mẫu.")
    print("=" * 78)


if __name__ == "__main__":
    main()
