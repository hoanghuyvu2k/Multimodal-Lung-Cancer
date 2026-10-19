"""Tạo sẵn embedding Bio_ClinicalBERT (768-d, prompt v2) ở môi trường Windows (doan-env, có transformers) và lưu parquet,
để môi trường tham chiếu (không có transformers) nạp lại. Suy luận thuần, không huấn luyện.

    C:/Users/Admin/miniconda3/envs/doan-env/python.exe experiments/reference_env/make_bioclin_cache.py
Kết quả: D:/repro_linux/cache/bioclin_embed_v2.parquet
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1])); sys.path.insert(0, str(_HERE.parents[2]))

from common.data_setup import load_all                               # noqa: E402
from nlp_verify.verify_nlp_variants import add_bioclinbert            # noqa: E402

OUT = Path("D:/repro_linux/cache/bioclin_embed_v2.parquet")

if __name__ == "__main__":
    ctx = load_all()
    add_bioclinbert(ctx, name="cnl_bioclin")
    df = ctx.modality_dict["cnl_bioclin"]
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(OUT)
    print("saved", OUT, df.shape)
