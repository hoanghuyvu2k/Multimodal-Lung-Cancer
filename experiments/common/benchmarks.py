"""
4 benchmark config cố định, dùng xuyên suốt mọi phương pháp để kết quả so sánh được với nhau.

Tham chiếu AUC cũ (1 seed, từ document/ovo-model/ovo-complete.md) chỉ để đối chiếu định tính —
KHÔNG dùng làm baseline. Baseline thật được khóa lại ở Step 3 với 5 seeds.
"""

RAD = ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln"]

MODEL_PARAMS = {
    "epochs": 125,
    "lr": 0.01,
    "alpha": 0.001,
    "beta": 0.0,
    "cross_modality_enabled": False,
}

SEEDS = [42, 7, 123, 2024, 31337]
FOLDS = 10

BENCHMARKS = {
    "BM1": {
        "name": "Rad+IHC-G+Gen+PDL1",
        "modalities": RAD + ["path_ihc_glcm", "gen_driver_mut_amp", "cnl_pdl1_score"],
        "use_rad_filters": True,
        "ref_auc_original": 0.7839,
        "ref_auc_ovo": 0.8003,
    },
    "BM2": {
        "name": "Rad+IHC-G+Gen+PDL1+Labs",
        "modalities": RAD + ["path_ihc_glcm", "gen_driver_mut_amp",
                             "cnl_pdl1_score", "cnl_dem_labs"],
        "use_rad_filters": True,
        "ref_auc_original": 0.7879,
        "ref_auc_ovo": 0.7834,
    },
    "BM3": {
        "name": "Rad+Gen",
        "modalities": RAD + ["gen_driver_mut_amp"],
        "use_rad_filters": True,
        "ref_auc_original": 0.7384,
        "ref_auc_ovo": 0.7548,
    },
    "BM4": {
        "name": "PDL1+Gen",
        "modalities": ["cnl_pdl1_score", "gen_driver_mut_amp"],
        "use_rad_filters": False,
        "ref_auc_original": 0.6931,
        "ref_auc_ovo": 0.7157,
    },
}

# Config chuẩn dùng cho chẩn đoán và cho paired bootstrap quyết định giữ/xóa
PRIMARY = "BM1"


def get_filters(bm_key, ctx):
    """Trả về l1_dfs_filter phù hợp cho benchmark."""
    return ctx.rad_filters if BENCHMARKS[bm_key]["use_rad_filters"] else {}
