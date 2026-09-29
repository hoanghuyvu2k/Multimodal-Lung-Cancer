"""Build NLP-clinical embedding (13 biến lâm sàng -> câu chữ -> MiniLM 384-dim) + tích hợp modality."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DS = _ROOT.parent / "datasets"
CACHE = DS / "nlp_clinical_embed.parquet"
LABS = "cnl_dem_labs"


def build_nlp_cache(ctx):
    """Embed 13 biến cnl_dem_labs -> parquet (index=patient)."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    from clinical_nlp_embedding import ClinicalTextEmbedder
    labs = ctx.modality_dict[LABS]
    present = ctx.modality_MASK.loc[labs.index, LABS].values.astype(bool)
    labs = labs[present]
    emb = ClinicalTextEmbedder().embed_dataframe(labs)   # 384-dim, index=patient
    emb.to_parquet(CACHE)
    return emb


def add_nlp_modality(ctx, name="cnl_nlp"):
    emb = build_nlp_cache(ctx)
    ctx.modality_dict[name] = emb.reindex(ctx.df_outcomes.index)
    ctx.modality_MASK[name] = ctx.df_outcomes.index.isin(emb.index)
    return int(ctx.modality_MASK[name].sum())
