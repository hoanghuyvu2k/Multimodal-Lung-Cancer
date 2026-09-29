# -*- coding: utf-8 -*-
"""Sinh Table 1 (patient & cohort characteristics) TRỰC TIẾP từ dữ liệu.

Chạy:  python paper/make_table1.py            # in báo cáo + ghi đè file .tex
       python paper/make_table1.py --dry-run  # chỉ in, không ghi file

Vì sao cần script này: bản Table 1 trước đây được gõ tay, và phần nhân khẩu
học của cột Discovery bị chép nhầm từ thống kê của TOÀN BỘ file omnibus
(366 dòng) thay vì cohort discovery (247 bệnh nhân).

Cách join từng cohort — theo document/data/omnibus-inventory-analysis.md §1.1
(`dmp_pt_id` chỉ khớp discovery; 2 cohort validation phải join bằng cột khác):
    discovery  -> dmp_pt_id
    rad_valid  -> radiology_accession_number (fallback: did_acc)
    path_valid -> pdl1_image_id              (fallback: slide_id, pdl1_acc)

Lưu ý `pfs_censor`: code (lung_helpers.py, lifelines `event_observed=`) coi
1 = CÓ biến cố. Script này theo đúng quy ước của code.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

HERE = Path(__file__).resolve().parent          # paper/
CODE = HERE.parent                              # code/
DB = CODE.parent / 'datasets'
OMNI = DB / ('18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540'
             '-WITH-TB-and-SCANNER.csv')
OUT_TEX = HERE / 'tables' / 'table1_patients.tex'

JOIN_COLS = {
    'discovery':  ['dmp_pt_id'],
    'rad_valid':  ['radiology_accession_number', 'did_acc'],
    'path_valid': ['pdl1_image_id', 'slide_id', 'pdl1_acc'],
}


# ─────────────────────────────────────────────────────────── helpers ──
def r1(x):
    """Làm tròn 1 chữ số thập phân kiểu half-up.

    Không dùng f'{x:.1f}': biểu diễn nhị phân khiến 2.55 -> '2.5' (làm tròn
    xuống), lệch so với quy ước báo cáo y khoa.
    """
    return str(Decimal(str(float(x))).quantize(Decimal('0.1'), ROUND_HALF_UP))


def g(x):
    """Bỏ '.0' thừa cho số nguyên (tuổi 38.0 -> 38)."""
    f = float(x)
    return str(int(f)) if f == int(f) else r1(f)


def pct(k, n):
    return f'{k} ({r1(k / n * 100)}\\%)' if n else '---'


def load_cohorts():
    coh = pd.read_csv(DB / 'final_cohort_listing.csv')
    raw = pd.read_csv(OMNI, low_memory=False)
    out = {}
    for name, cols in JOIN_COLS.items():
        ids = set(coh.loc[coh.cohort == name, 'main_index'].astype(str))
        best, bestcol = None, None
        for c in cols:
            if c not in raw.columns:
                continue
            key = raw[c].astype(str).str.replace(r'\.0$', '', regex=True)
            sub = raw[key.isin(ids)]
            if best is None or len(sub) > len(best):
                best, bestcol = sub, c
        if best is None or len(best) != len(ids):
            raise SystemExit(
                f'[LOI] cohort {name}: join bang {bestcol} chi khop '
                f'{0 if best is None else len(best)}/{len(ids)} dong')
        out[name] = (best, bestcol, len(ids))
    return out, raw


def modality_counts_discovery():
    """Số bệnh nhân có từng modality, lấy từ chính mask của pipeline."""
    sys.path.insert(0, str(CODE / 'experiments'))
    from common.data_setup import load_all
    m = load_all().modality_MASK
    rad_any = m[['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln']].any(axis=1)
    return {
        'rad': int(rad_any.sum()),
        'path': int(m['path_ihc_glcm'].sum()),
        'gen': int(m['gen_driver_mut_amp'].sum()),
        'pdl1': int(m['cnl_pdl1_score'].sum()),
        'labs': int(m['cnl_dem_labs'].sum()),
        'n': len(m),
    }


def modality_counts_validation():
    """rad_valid / path_valid: bao nhiêu bệnh nhân thực sự có dữ liệu."""
    coh = pd.read_csv(DB / 'final_cohort_listing.csv')
    res = {}
    for name, fname in [
        ('rad_valid', 'lung_radiomics_spacing1.0_mirpon_window1350.250'
                      '_allimagetypes_bw20_validation.parquet'),
        ('path_valid', 'lung_pathology_pdl1_glcm_v3_validation.parquet'),
    ]:
        ids = set(coh.loc[coh.cohort == name, 'main_index'].astype(str))
        path = DB / fname
        if not path.exists():
            res[name] = (None, len(ids))
            continue
        df = pd.read_parquet(path)
        idx = df.index if df.index.name else df.iloc[:, 0]
        have = {str(x).replace('.0', '') for x in idx}
        res[name] = (len(ids & have), len(ids))
    return res


def stats(df):
    n = len(df)
    s = {'n': n}
    s['age'] = f'{r1(df.age.mean())} ({g(df.age.min())}--{g(df.age.max())})'
    s['male'] = pct(int((df.sex == 1).sum()), n)
    h = df.histo.value_counts()
    adeno, squa = int(h.get('Adenocarcinoma', 0)), int(h.get('Squamous', 0))
    s['adeno'], s['squa'] = pct(adeno, n), pct(squa, n)
    s['other'] = pct(n - adeno - squa, n)
    e = df.ecog.value_counts()
    e0, e1 = int(e.get(0, 0)), int(e.get(1, 0))
    s['ecog0'], s['ecog1'] = pct(e0, n), pct(e1, n)
    s['ecog2'] = pct(n - e0 - e1 - int(df.ecog.isna().sum()), n)
    for key, col in [('pd1', 'recieves_pd1_therapy'),
                     ('pdl1t', 'recieves_pdl1_therapy'),
                     ('combo', 'recieves_combo_therapy')]:
        s[key] = pct(int((df[col] == 1).sum()), n) if col in df and \
            df[col].notna().any() else '---\\textsuperscript{c}'
    resp = int((df.label == 0).sum())
    s['resp'], s['noresp'] = pct(resp, n), pct(n - resp, n)
    s['events'] = pct(int((df.pfs_censor == 1).sum()), n)
    s['pfs'] = (f'{r1(df.pfs.median())} '
                f'({r1(df.pfs.min())}--{r1(df.pfs.max())})')
    return s


# ──────────────────────────────────────────────────────────── main ──
def main():
    dry = '--dry-run' in sys.argv
    cohorts, raw = load_cohorts()
    st = {k: stats(v[0]) for k, v in cohorts.items()}
    mod_d = modality_counts_discovery()
    mod_v = modality_counts_validation()

    print(f'omnibus: {len(raw)} dòng')
    for k, (df, col, want) in cohorts.items():
        print(f'  {k:11s} join bằng "{col}" -> khớp {len(df)}/{want}')
    print()
    for k in ('discovery', 'rad_valid', 'path_valid'):
        s = st[k]
        print(f'--- {k} (n={s["n"]}) ---')
        for f in ('age', 'male', 'adeno', 'squa', 'other', 'ecog0', 'ecog1',
                  'ecog2', 'pd1', 'pdl1t', 'combo', 'resp', 'noresp',
                  'events', 'pfs'):
            print(f'    {f:8s}: {s[f].replace(chr(92) + "%", "%")}')
        print()
    print('modality (discovery):', mod_d)
    print('modality (validation):', mod_v)

    d, r, p = st['discovery'], st['rad_valid'], st['path_valid']

    def pair(field):
        return f'{r[field]}; {p[field]}'

    nd = d['n']
    rv_mod = f'{mod_v["rad_valid"][0]}/{mod_v["rad_valid"][1]}' \
        if mod_v['rad_valid'][0] is not None else '---'
    pv_mod = f'{mod_v["path_valid"][0]}/{mod_v["path_valid"][1]}' \
        if mod_v['path_valid'][0] is not None else '---'

    tex = f"""%% tables/table1_patients.tex
%% SINH TỰ ĐỘNG bởi paper/make_table1.py — ĐỪNG sửa tay.
%% Chạy lại: python paper/make_table1.py
\\begin{{table}}[H]
\\caption{{Patient and cohort characteristics.
  The discovery cohort was used for model development with 10-fold
  cross-validation; validation cohorts were held out entirely.
  Validation column reports radiomics-evaluable; pathology-evaluable cohort
  values, separated by a semicolon.
  \\textsuperscript{{a}}~Percentage of patients with the given modality
  available; \\textsuperscript{{b}}~ICI categories are recorded as independent
  flags and are not mutually exclusive (patients on combination therapy are
  counted in more than one row); \\textsuperscript{{c}}~not recorded in the
  source registry for the validation cohorts.
  PR/CR, partial/complete response; SD/PD, stable/progressive disease;
  ICI, immune checkpoint inhibitor; PD-L1, programmed death-ligand 1;
  TPS, tumour proportion score; PFS, progression-free survival.}}
\\label{{tab:patients}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Characteristic}} & \\textbf{{Discovery}} & \\textbf{{Validation}} \\\\
                        & \\textbf{{($n = {nd}$)}} & \\textbf{{(rad $n={r['n']}$; path $n={p['n']}$)}} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Demographics}}}} \\\\
\\quad Age, years, mean (range)  & {d['age']} & {pair('age')} \\\\
\\quad Male sex, $n$ (\\%)        & {d['male']} & {pair('male')} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Histology, $n$ (\\%)}}}} \\\\
\\quad Adenocarcinoma            & {d['adeno']} & {pair('adeno')} \\\\
\\quad Squamous cell             & {d['squa']} & {pair('squa')} \\\\
\\quad Other / NOS               & {d['other']} & {pair('other')} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Performance status (ECOG), $n$ (\\%)}}}} \\\\
\\quad 0                         & {d['ecog0']} & {pair('ecog0')} \\\\
\\quad 1                         & {d['ecog1']} & {pair('ecog1')} \\\\
\\quad $\\geq$2                   & {d['ecog2']} & {pair('ecog2')} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{ICI therapy\\textsuperscript{{b}}, $n$ (\\%)}}}} \\\\
\\quad Anti-PD-1                 & {d['pd1']} & {pair('pd1')} \\\\
\\quad Anti-PD-L1                & {d['pdl1t']} & {pair('pdl1t')} \\\\
\\quad Combination               & {d['combo']} & {pair('combo')} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Treatment outcome, $n$ (\\%)}}}} \\\\
\\quad Response (PR/CR), label=0 & {d['resp']} & {pair('resp')} \\\\
\\quad No response (SD/PD), label=1 & {d['noresp']} & {pair('noresp')} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Survival (PFS)}}}} \\\\
\\quad Events (progression/death), $n$ (\\%) & {d['events']} & {pair('events')} \\\\
\\quad Median PFS, months (range)            & {d['pfs']} & {pair('pfs')} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Modality availability\\textsuperscript{{a}}, $n$ (\\%)}}}} \\\\
\\quad CT radiomics (PC/PL/LN)  & {pct(mod_d['rad'], nd)}  & {rv_mod}; --- \\\\
\\quad Pathology IHC             & {pct(mod_d['path'], nd)}  & ---; {pv_mod} \\\\
\\quad Genomics (NGS)            & {pct(mod_d['gen'], nd)}  & --- \\\\
\\quad PD-L1 TPS score           & {pct(mod_d['pdl1'], nd)} & --- \\\\
\\quad Clinical labs (13 vars)   & {pct(mod_d['labs'], nd)} & --- \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    if dry:
        print('\n--- (dry-run) noi dung .tex ---\n')
        print(tex)
    else:
        OUT_TEX.write_text(tex, encoding='utf-8')
        print(f'\nDA GHI: {OUT_TEX}')


if __name__ == '__main__':
    main()
