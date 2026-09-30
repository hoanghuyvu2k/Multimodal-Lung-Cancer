# Kế hoạch kiểm chứng lại toàn bộ số liệu trong bài báo

> Ngày lập: 2026-09-29 · Đối tượng: `paper/main.tex` + `paper/supplementary/supplementary.tex`
> (Bảng 1–8, Hình 1–5, Hình S1–S5, Bảng S1–S3, và các con số nằm trong văn bản)

---

## 0. Nguyên tắc & ngưỡng chấp nhận

**Pipeline này là tất định.** `train()`/`train_ovo()` hardcode `KFold(random_state=0)` và
`torch.manual_seed(42)` trong `__init__`; harness 5-seed né bằng cách hoán vị hàng `outcomes` theo
seed. Hệ quả: **chạy lại phải ra đúng con số cũ**, không phải "gần đúng".

| Mức lệch | Kết luận |
|---|---|
| 0 (khớp tuyệt đối) | Đạt |
| ≤ 0.002 AUC | Chấp nhận — sai khác BLAS/threading giữa các lần chạy |
| 0.002–0.005 | Ghi chú lại, kiểm tra version thư viện |
| > 0.005 | **Điều tra** — nhiều khả năng dữ liệu đầu vào hoặc config đã đổi |

**Môi trường đã xác nhận có sẵn (2026-09-29):** torch 2.13.0+cpu, sklearn 1.6.1, pandas 2.3.3,
numpy 1.26.4, sentence-transformers 5.3.0, transformers 4.46.3, lifelines 0.30.3, sksurv 0.27.0,
scipy 1.14.1. Dữ liệu ở `../datasets/`. Tất cả lệnh chạy **từ thư mục `code/`**.

**Hai cái bẫy bắt buộc xử lý trước khi chạy:**

1. **Script ghi đè file kết quả cũ.** `run_paper_combos.py` khởi tạo `payload = {"combos": {}}` rồi
   `save_result("paper_combos", ...)` — chạy lại là mất bản gốc để đối chiếu. Sao lưu trước:
   ```bash
   mkdir -p experiments/results/_ref_2026-09-29
   cp experiments/results/*.json experiments/results/_ref_2026-09-29/
   ```
2. **Lỗi encoding trên Windows.** Script in tiếng Việt + ký tự `±`, `Δ` sẽ crash cp1252. Luôn chạy:
   ```bash
   PYTHONIOENCODING=utf-8 python <script>
   ```

**Thời gian tham chiếu (đo thật trên máy này 2026-09-29):** một lần train 10-fold × 125 epoch với
4 modality = **15,4 giây**; `load_all()` = 4,2 giây. Các ước tính bên dưới suy ra từ mốc này (tổ hợp
nhiều modality + có L1-filter radiomics sẽ chậm hơn, nên cộng thêm ~30%).

---

## 1. Bản đồ số liệu → nguồn

| Trong bài báo | Nguồn sinh số | File kết quả | Tái lập được? |
|---|---|---|---|
| **Bảng 1** (đặc điểm bệnh nhân) | `paper/make_table1.py` | `paper/tables/table1_patients.tex` | ✅ Script, chạy < 1 phút |
| **Bảng 2** (danh sách modality) | Viết tay từ code | — | ✅ Đối chiếu thủ công |
| **Bảng 3** (screen single-partition OvO) | `run_paper_singlerun.py` + `run_paper_ovo.py` | `paper_combos.json` | ✅ |
| **Bảng 4** (21 tổ hợp, 5-seed, 3 model) | `run_paper_combos.py` → `run_paper_singlerun.py` → `run_paper_ovo.py` → `run_ovo_nlp.py` | `paper_combos.json`, `ovo_nlp_combos.json` | ✅ |
| **Bảng 5, 6** (7 biến thể NLP + DeLong CI) | **Notebook `Figures-Finalized.ipynb`** | Chỉ còn `document/nlp-model/nlp-ket-qua.md` | ⚠️ **CHƯA có script — xem Phase 3** |
| **Bảng 7** (C-index/Cox/tdAUC/IBS) | `survival_analysis_nlp.py` | `excel/survival_analysis_summary.xlsx` | ✅ |
| **Bảng 8** (KM log-rank χ²) | `survival_analysis_nlp.py` | cùng file trên | ✅ |
| **Hình 1, 2, 3** (sơ đồ) | `paper/generate_fig{1,2,_attention}_demo.py` | `paper/figures/*_demo.pdf` | ✅ Không phụ thuộc train |
| **Hình 3a/3b** (bar AUC) | `paper/generate_fig3_demo.py` | — | ✅ Số hardcode từ Bảng 5/6 → kiểm cùng Phase 3 |
| **Hình 4** (KM curves) | `paper/drafts/make_fig4_km.py` | — | ✅ |
| **Hình 5** (C-index/tdAUC/Cox/IBS) | `paper/drafts/make_fig3_fig5.py` | — | ✅ |
| **Hình S1, S2** (PCA scree, t-SNE) | `paper/drafts/make_figS1_S2_nlp_embeddings.py` | Tính từ embedding thật | ✅ |
| **Hình S3** (bootstrap ΔAUC) | `paper/drafts/make_figS3_bootstrap.py` | — | ⚠️ Xem rủi ro R5 |
| **Hình S4** (scatter OvO) | `paper/drafts/make_figS4_ovo_scatter.py` | — | ✅ |
| **Bảng S3** (single-partition 21 tổ hợp) | `run_paper_singlerun.py`, `run_ovo_singlerun.py` | `paper_combos.json` | ✅ |
| **Note S1** (câu prompt, token) | `clinical_nlp_embedding.py::df_to_text_prompts` | — | ✅ Phase 0 |
| **Note S3** (power analysis) | Tính tay từ Hanley–McNeil | — | ✅ Phase 0 |
| **p-value bootstrap trong caption Bảng 4** | `run_paper_ovo.py` (`paired_bootstrap`, 2000 resamples) | `paper_combos.json` → `paired` | ✅ |

---

## Phase 0 — Tiền kiểm, không cần train (~30 phút)

Làm trước vì rẻ và bắt được lỗi dữ liệu sớm nhất.

### 0.1. Bảng 1
```bash
PYTHONIOENCODING=utf-8 python paper/make_table1.py --dry-run
```
Đối chiếu với `paper/tables/table1_patients.tex`. **Mốc chốt:** n=247; 62 đáp ứng (25,1%) / 185 không
(74,9%); tuổi TB 66,9 (38–93); nam 113 (45,7%); adeno 184 (74,5%); ECOG 1 = 194 (78,5%);
209 biến cố PFS (84,6%); radiomics 187 (75,7%); pathology 105 (42,5%); PD-L1 TPS 201 (81,4%).

### 0.2. Các con số trong Note S1 và Hình 2
```bash
PYTHONIOENCODING=utf-8 python - <<'PY'
import pandas as pd, sys
sys.path.insert(0,'.')
from clinical_nlp_embedding import df_to_text_prompts
# nạp df_labs đúng như pipeline rồi:
p = df_to_text_prompts(df_labs)
print(p[0])
print('words:', min(len(x.split()) for x in p), '-', max(len(x.split()) for x in p))
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
n = [len(tk(x)['input_ids']) for x in p]
print('tokens:', min(n), '-', max(n))
PY
```
**Kỳ vọng:** 65–68 từ, 129–135 token (Note S1); câu ví dụ trùng câu in trong Note S1 và Methods §2.3.

### 0.3. Phương sai PCA giữ lại (con số 94% xuất hiện 5 chỗ)
```bash
PYTHONIOENCODING=utf-8 python - <<'PY'
import pandas as pd
from sklearn.decomposition import PCA
emb = pd.read_parquet('../datasets/nlp_clinical_embed.parquet')
pca = PCA(n_components=16, random_state=42).fit(emb.values)
print('PC16 cumulative variance:', pca.explained_variance_ratio_.sum())
PY
```
**Kỳ vọng:** ≈0,94 (Bảng 2, Methods §2.4, caption Hình 2, chữ trong Hình 2, caption Hình S1).

### 0.4. Note S3 (power) — tính lại bằng tay
Kiểm 4 con số: nửa độ rộng CI ±0,07; SE(Δ) 0,014–0,016; n≈950–1.250 (Δ=0,02); n≈420–550 (Δ=0,03).
Công thức: Hanley–McNeil cho SE(AUC) với n₊=62, n₋=185; SE₀(Δ)=√(2(1−ρ))·SE(AUC) với ρ=0,9;
n ≥ n₀·((z_{α/2}+z_β)·SE₀(Δ)/Δ)², n₀=247.

### 0.5. Ba hình sơ đồ
```bash
python paper/check_figures.py     # phải exit 0
```

---

## Phase 1 — 21 tổ hợp: DyAM + OvO (Bảng 3, 4, S3) · ~3–4 giờ

**Thứ tự bắt buộc** — `run_paper_ovo.py` và `run_paper_singlerun.py` đều `load_result("paper_combos")`
rồi ghi thêm vào, nên phải chạy `run_paper_combos.py` đầu tiên:

```bash
# 1a. uniform_avg + DyAM, 5-seed x 10-fold, 21 combo  (~1,5-2h)
PYTHONIOENCODING=utf-8 python experiments/allcombo/run_paper_combos.py

# 1b. bổ sung cột single-run cho uniform + DyAM       (~25-35 phút)
PYTHONIOENCODING=utf-8 python experiments/allcombo/run_paper_singlerun.py

# 1c. OvO: 5-seed + single-run + paired bootstrap      (~45-60 phút)
PYTHONIOENCODING=utf-8 python experiments/allcombo/run_paper_ovo.py
```

**Đối chiếu:**
```bash
PYTHONIOENCODING=utf-8 python - <<'PY'
import json
new = json.load(open('experiments/results/paper_combos.json'))['combos']
ref = json.load(open('experiments/results/_ref_2026-09-29/paper_combos.json'))['combos']
for k in ref:
    for model in ('uniform','dyam','ovo'):
        if model in ref[k] and model in new.get(k,{}):
            a,b = ref[k][model]['mean_auc'], new[k][model]['mean_auc']
            flag = '' if abs(a-b)<=0.002 else '  <-- LECH'
            print(f"{k:<30}{model:<9}{a:.4f} -> {b:.4f}{flag}")
PY
```

**Mốc chốt (Bảng 4, cột DyAM / OvO):** #18 BM1 = 0,772 / 0,767; #11 BM3 = 0,704 / 0,704;
#8 BM4 = 0,708 / 0,718; #1 TMB = 0,616 cho cả hai (k=1 thì OvO trùng DyAM về mặt đại số — nếu
lệch nhau ở bất kỳ dòng k=1 nào là **có lỗi pipeline**).

**Mốc chốt (p-value bootstrap trong caption Bảng 4):** BM4 Δ=+0,011 p=0,172; BM3 Δ=+0,000 p=0,980;
BM1 Δ=+0,009 p=0,313; BM2 Δ=−0,004 p=0,536. Bootstrap có random — chấp nhận p lệch ±0,02.

---

## Phase 2 — OvO + NLP (cột thứ 3 của Bảng 4) · ~1,5–2 giờ

⚠️ `run_ovo_nlp.py` **bỏ qua combo đã có trong json** (`if label in done`). Phải dọn trước, nếu không
script chỉ in lại số cũ và bạn không kiểm chứng được gì:

```bash
mv experiments/results/ovo_nlp_combos.json experiments/results/_ref_2026-09-29/
PYTHONIOENCODING=utf-8 python experiments/nlp_verify/run_ovo_nlp.py
```

**Mốc chốt:** #18 BM1 = 0,7822 ± 0,017 (single-run 0,8123); #19 Rad+IHC-A+Gen+TMB+PDL1 single-run
**0,8133** — con số single-run cao nhất được nêu trong §3.2 bài báo; #8 BM4 = 0,7472.

Tuỳ chọn, nếu muốn kiểm luôn nhánh uniform_avg+NLP (§3.3 dùng để xác nhận chéo):
```bash
mv experiments/results/nlp_combos.json experiments/results/_ref_2026-09-29/
PYTHONIOENCODING=utf-8 python experiments/nlp_verify/run_nlp.py      # ~2h
```
**Mốc chốt:** BM1 +NLP = 0,7832 (single-run 0,8091); +NLP > +Labs ở 19/21 tổ hợp.

---

## Phase 3 — ⚠️ Lỗ hổng lớn nhất: Bảng 5 & 6 (~1 giờ viết script + 15 phút chạy)

**Vấn đề (cập nhật 2026-09-30 sau khi kiểm tra thật):** Bảng 5 và Bảng 6 — 7 biến thể encoding lâm
sàng kèm CI DeLong, tức là *trái tim của đóng góp NLP* — **không có nguồn tái lập được trong repo**.

`result/training-results.md` §2.0b ghi nguồn là "notebook gốc `Figures-Finalized*.ipynb`", nhưng tôi
đã quét cả 66 cell của notebook: **chỉ 1 cell nhắc tới "nlp"** (cell 63, kiểm định bootstrap), không
có cell nào gọi `clinical_nlp_embedding`, `MiniLM`, hay `sentence-transformers`. Notebook không sinh
ra các số này. Nguồn duy nhất còn lại là file ghi chép `document/nlp-model/nlp-ket-qua.md`.
Hình 3a/3b cũng hardcode lại chính các số đó.

**Mâu thuẫn thứ hai, cần giải quyết trong phase này:** cùng một model nhưng hai chỗ ghi hai số khác
nhau. §2.1 (từ `survival_analysis_nlp.py`, có script hẳn hoi) ghi cột "AUC binary": No Clinical
IHC-A = **0,776**, NLP-PCA16 IHC-A = **0,773**. Bảng 5 ghi **0,764** và **0,784**. Lệch ~0,01–0,02
và thậm chí *đảo chiều* hiệu ứng NLP. Phải xác định số nào đúng trước khi kết luận Bảng 5.

**Config để tái lập** (lấy từ cell 13 của notebook, trùng `survival_analysis_nlp.py`):
`model_params = {'epochs':125, 'lr':0.01, 'alpha':0.001, 'beta':0.0, 'cross_modality_enabled':False}`,
rad filter `l1_strength=0.1` cho PC/PL/LN, `folds=10`.

**Việc cần làm:** viết `experiments/nlp_verify/verify_nlp_variants.py` sinh lại đủ 7 biến thể cho cả
2 nhánh IHC:

| Biến thể | Modality thay đổi |
|---|---|
| No clinical (ref) | Rad(PC,PL,LN) + IHC + Gen + PDL1 |
| + Labs | thêm `cnl_dem_labs` |
| + NLP raw 384d | thêm `cnl_nlp_embedding`, `no_scale` |
| + NLP-PCA16 | thêm `cnl_nlp_pca16`, `no_scale` |
| + Labs + NLP | thêm cả hai |
| + Labs + NLP-PCA16 | thêm cả hai |
| + BioClinBERT-PCA16 | `prepare_clinical_bert_modality(prompt_version='v2')` |

Dùng `train()` (DyAM), `folds=10`, `MODEL_PARAMS` chuẩn, rồi `auc_roc_ci(y, score, 0.95)` để lấy CI
DeLong. Lưu `summary_df` của từng biến thể ra `experiments/results/scores/` — Phase 4 và Hình S3 cần
đúng các cột score này.

**Mốc chốt (Bảng 5, IHC-A):** 0,764 / 0,768 / 0,781 / **0,784** / 0,767 / 0,753 / 0,767.
**Mốc chốt (Bảng 6, IHC-G):** 0,784 / 0,788 / **0,813** / 0,812.
CI DeLong: kiểm ít nhất dòng NLP-PCA16 IHC-A = [0,719–0,848].

> Ghi chú: nếu số chạy lại lệch, **chưa chắc bài báo sai** — có thể notebook dùng config hơi khác
> (số epoch, l1_filter). Khi đó phải mở notebook đọc lại cell tương ứng rồi mới kết luận.

---

## Phase 4 — Survival: Bảng 7, 8 + Hình 4, 5 (~30–45 phút)

```bash
PYTHONIOENCODING=utf-8 python survival_analysis_nlp.py
```
Script tự train 8 biến thể DyAM (4 biến thể × 2 nhánh IHC), tính C-index Harrell (bootstrap 1.000),
Cox đa biến (adjust age/ECOG/albumin/dNLR/liver_mets, penalizer=0,1), tdAUC 6/12/18 tháng, IBS,
log-rank KM, và external validation. Kết quả ra `excel/survival_analysis_summary.xlsx` +
`excel/external_validation_cindex.xlsx`.

**Mốc chốt (Bảng 7, IHC-A):** C-index 0,623 / 0,626 / 0,625 / **0,628**; Cox HR 5,30 / 5,31 / 5,91 /
**6,07**; IBS 0,1745 / 0,1739 / 0,1723 / **0,1716**. Tất cả Cox p < 0,001.
**Mốc chốt (Bảng 8, KM χ²):** 28,17 / 23,74 / 20,07 / **28,92**.
**Mốc chốt (external, nêu trong Discussion):** IHC-G trên path_valid C-index 0,618, AUC 0,767.

---

## Phase 5 — Hình phụ lục & dựng lại bài (~30 phút)

```bash
cd paper
python drafts/make_figS1_S2_nlp_embeddings.py   # S1 scree (phải in PC16 = 93.9%), S2 t-SNE
python drafts/make_figS3_bootstrap.py           # xem rủi ro R5
python drafts/make_figS4_ovo_scatter.py
python drafts/make_figS5_heatmap.py
python drafts/make_fig4_km.py
python drafts/make_fig3_fig5.py
python generate_fig3_demo.py                    # Hình 3a/3b — số phải khớp Bảng 5, 6
python check_figures.py
latexmk -pdf main.tex && (cd supplementary && latexmk -pdf supplementary.tex)
python word_export/build_manuscript_docx.py
```

---

## Phase 6 — Đối chiếu & ghi nhận

1. Điền cột "chạy lại" vào checklist mục 7 bên dưới.
2. **Ghi kết quả vào `result/training-results.md`** theo Result logging rule của dự án — một mục mới
   `§2.10 Kiểm chứng lại số liệu bài báo (2026-09-29)`, ghi đủ config + verdict, **không ghi đè** mục cũ.
3. Số nào lệch > 0,005: sửa trong `paper/sections/*.tex` **và** `paper/word_export/build_manuscript_docx.py`
   (hai nơi phải luôn giống nhau), rồi compile lại cả PDF lẫn Word.
4. Xoá thư mục `_ref_2026-09-29/` chỉ sau khi đã đối chiếu xong.

---

## 6. Rủi ro đã biết — soi kỹ khi kiểm chứng

| # | Vấn đề | Ảnh hưởng | Nên làm gì |
|---|---|---|---|
| **R1** | **PCA rò rỉ dữ liệu.** `survival_analysis_nlp.py:105` fit PCA(16) trên **toàn bộ 247 bệnh nhân** rồi mới chia fold. Cùng cách làm ở notebook cho Bảng 5/6. | NLP-PCA16 (con số đẹp nhất bài: 0,784 / 0,812) có thể lạc quan hơn thực tế | Chạy thêm một biến thể fit PCA **trong từng fold train** rồi so. Nếu chênh đáng kể thì phải nêu trong Limitations |
| **R2** | `seed` của `train()` không có tác dụng (KFold hardcode `random_state=0`); harness né bằng hoán vị nhãn | 5-seed đo biến thiên khởi tạo + phân hoạch gián tiếp, không phải CV lặp chuẩn | Bài báo đã mô tả đúng chưa? Kiểm câu mô tả CV trong Methods |
| **R3** | KFold **không phân tầng** (không stratified), lớp thiểu số chỉ 62/247 | Một số fold có thể rất ít ca đáp ứng | Kiểm Methods có tự nhận là "stratified" không — nếu có là sai, phải sửa |
| **R4** | Median PFS trong Bảng 1 tính bằng `df.pfs.median()` thô, không phải ước lượng Kaplan–Meier | 2,7 vs 3,0 [2,1–4,7] ở path_valid | Quyết định: đổi sang KM (chuẩn hơn) hay chú thích rõ là median thô |
| **R5** | Bảng bootstrap trong `nlp-ket-qua.md` §4.3 tự mâu thuẫn: p≈0,5 nhưng CI không chứa 0 | Hình S3 dựng từ nguồn này | Tính lại bootstrap từ score thật (có sau Phase 3) |
| **R6** | BioClinBERT dùng prompt v2, MiniLM dùng v1 | So sánh lẫn lộn encoder với prompt | Note S2 đã nêu rõ; nếu muốn so sạch thì chạy thêm BioClinBERT với prompt v1 |
| **R7** | `build_l1` trong `nlp_run.py` chỉ map 3 cột radiomics chuẩn → combo #6 (Rad-LU) không có L1-filter | Đã tồn tại từ §2.8, giữ nguyên để so táo-với-táo | Không sửa, chỉ cần biết khi đọc số |
| **R8** | `experiments/results/scores/` vừa bị đưa vào .gitignore | Đây là input để tính lại DeLong/bootstrap mà không cần train lại | **Đừng xoá** thư mục này trước khi kiểm chứng xong |

---

## 7. Checklist

| Hạng mục | Lệnh/Phase | Kỳ vọng | Khớp? |
|---|---|---|---|
| Bảng 1 | P0.1 | n=247, 62/185, 209 biến cố | ☐ |
| Note S1 (câu, token) | P0.2 | 65–68 từ, 129–135 token | ☐ |
| 94% phương sai PCA | P0.3 | ≈0,94 | ☐ |
| Note S3 (power) | P0.4 | ±0,07; 950–1.250; 420–550 | ☐ |
| Hình 1, 2, 3 layout | P0.5 | exit 0 | ☐ |
| Bảng 4 cột DyAM | P1 | BM1 0,772 | ☐ |
| Bảng 4 cột OvO | P1 | BM1 0,767; k=1 trùng DyAM | ☐ |
| p-value caption Bảng 4 | P1 | 0,172 / 0,980 / 0,313 / 0,536 | ☐ |
| Bảng 3 + Bảng S3 (single-run) | P1 | — | ☐ |
| Bảng 4 cột OvO+NLP | P2 | BM1 0,7822; #19 1-run 0,8133 | ☐ |
| Bảng 5 (IHC-A, 7 biến thể) | P3 | 0,764→0,784 | ☐ |
| Bảng 6 (IHC-G) | P3 | 0,784→0,813 | ☐ |
| CI DeLong | P3 | NLP-PCA16 [0,719–0,848] | ☐ |
| Bảng 7 (survival) | P4 | C-index 0,628; HR 6,07 | ☐ |
| Bảng 8 (KM χ²) | P4 | 28,17/23,74/20,07/28,92 | ☐ |
| External validation | P4 | IHC-G path_valid 0,618 / AUC 0,767 | ☐ |
| Hình S1 scree | P5 | PC16 = 93,9% | ☐ |
| Hình 3a/3b khớp Bảng 5/6 | P5 | — | ☐ |
| Ghi vào training-results.md | P6 | §2.10 | ☐ |

---

## 8. Tổng thời gian

| Phase | Nội dung | Thời gian |
|---|---|---|
| 0 | Tiền kiểm không train | ~30 phút |
| 1 | 21 tổ hợp DyAM + OvO + single-run | ~3–4 giờ |
| 2 | OvO+NLP (+ tuỳ chọn uniform+NLP) | ~2 giờ (+2 giờ) |
| 3 | **Viết script mới** + chạy 7 biến thể NLP | ~1,5 giờ |
| 4 | Survival | ~45 phút |
| 5 | Hình + compile PDF/Word | ~30 phút |
| 6 | Đối chiếu, ghi nhận, sửa nếu lệch | ~1 giờ |
| | **Tổng** | **~9–12 giờ máy**, chia được thành nhiều buổi |

Các phase chạy độc lập được, trừ: P1 phải theo đúng thứ tự 1a→1b→1c, và P4 nên chạy sau P3 nếu muốn
dùng chung bộ score.
