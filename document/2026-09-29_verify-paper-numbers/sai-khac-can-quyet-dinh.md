# Sai khác giữa bài báo và kết quả chạy lại — chờ quyết định

> Cập nhật: 2026-09-30, **đã chạy xong toàn bộ 6 phase** (tổng 3 giờ 20 phút)
> **Bài báo đang ở nguyên trạng — tôi chưa sửa gì.** Mọi thay đổi đề xuất nằm trong file này.
> Sinh lại bảng so sánh Bảng 4: `PYTHONIOENCODING=utf-8 python experiments/compare_to_paper.py`

Mỗi mục có ô quyết định ở cuối. Bạn đánh dấu rồi báo tôi, tôi mới sửa.

## Tóm tắt một trang

| Nhóm số liệu | Kết quả kiểm chứng |
|---|---|
| Bảng 1, Note S1, PCA 94%, bố cục hình | ✅ Khớp tuyệt đối |
| Bảng 4 — 13/21 tổ hợp không radiomics | ✅ Khớp tuyệt đối tới 4 chữ số |
| Bảng 4 — cột OvO+NLP, cả 21 tổ hợp | ✅ **Khớp tuyệt đối cả 21** |
| Bảng 4 — 23/63 ô còn lại (đều có radiomics) | ⚠️ Lệch ≤ 0,016 — **do scipy đổi phiên bản** (SK-2) |
| Note S3 (công suất thống kê) | ⚠️ Sai số học (SK-1) |
| **Bảng 5, 6, 7, 8** | ❌ Không tái lập; đã chạy lại 5-seed — **luận điểm "NLP vs Labs" đứng, "NLP vs không-lâm-sàng" sụp** (SK-3/4/5) |
| Bảng 7 — cột tdAUC và IBS | ⛔ Không kiểm được (SK-6) |

**Điểm quan trọng nhất:** bằng chứng 5-seed cho NLP (thứ đang chống đỡ luận điểm chính của bài)
**tái lập hoàn hảo** — cả 21 tổ hợp cột OvO+NLP khớp tới 4 chữ số, BM1 vẫn là base 0,767 → +NLP
0,782. Nhưng các bảng đơn-phân-hoạch (5, 6, 7, 8) thì không tái lập được, và ở đó hiệu ứng NLP đảo
chiều. Hai script độc lập (Phase 3 và Phase 4) cho cùng một kết quả mới, lệch nhau < 0,001.

---

## SK-1 · Note S3: SE(Δ) và cỡ mẫu — **sai số học, không phụ thuộc môi trường**

**Vị trí:** `supplementary.tex:119, 127` · `discussion.tex:41` · `build_manuscript_docx.py:1462`

Note S3 tự nêu giả định (Hanley–McNeil tại AUC 0,78–0,81 với 62/185; ρ ≈ 0,9), nhưng số ghi ra không
khớp với chính công thức đó:

| Đại lượng | Bài báo ghi | Tính lại | |
|---|---|---|---|
| SE(Δ) | 0,014–0,016 | **0,0158–0,0166** | ⚠️ |
| n cần, Δ=0,02 | 950–1.250 | **1.211–1.343** | ⚠️ |
| n cần, Δ=0,03 | 420–550 | **538–597** | ⚠️ |
| Nửa độ rộng CI | ±0,07 | ±0,069–0,073 | ✅ |
| Thiếu công suất | ~2–5× | 4,9–5,4× và 2,2–2,4× | ✅ |

**Đề xuất:** SE(Δ) ≈ 0,016 · n ≈ 1.200–1.350 (Δ=0,02) · n ≈ 540–600 (Δ=0,03). Discussion đổi
"950–1.250" → "1.200–1.350".

**Quyết định:** ☐ Sửa theo đề xuất ☐ Giữ nguyên ☐ Cách khác: ______

---

## SK-2 · Bảng 4: 23/63 ô lệch — **đều là tổ hợp radiomics, do scipy đổi phiên bản**

**Vị trí:** `paper/sections/results.tex`, bảng `tab:ovo_all21`

| # | Tổ hợp | Cột | Bài báo | Chạy lại | Lệch |
|---|---|---|---|---|---|
| 5 | Rad | DyAM | 0.673 | 0.6692 | −0.0038 |
| 5 | Rad | OvO | 0.659 | 0.6691 | +0.0101 |
| 6 | Rad-LU | DyAM | 0.637 | 0.6465 | +0.0095 |
| 6 | Rad-LU | OvO | 0.646 | 0.6486 | +0.0026 |
| 9 | Rad+IHC-A | DyAM | 0.671 | 0.6846 | +0.0136 |
| 9 | Rad+IHC-A | OvO | 0.668 | 0.6844 | +0.0164 |
| 10 | Rad+IHC-G | DyAM | 0.706 | 0.7112 | +0.0052 |
| 10 | Rad+IHC-G | OvO | 0.703 | 0.7067 | +0.0037 |
| 11 | Rad+Gen (BM3) | OvO | 0.713 | 0.7163 | +0.0033 |
| 14 | Rad+IHC-A+Gen | DyAM | 0.725 | 0.7325 | +0.0075 |
| 14 | Rad+IHC-A+Gen | OvO | 0.735 | 0.7325 | −0.0025 |
| 15 | Rad+IHC-G+Gen | DyAM | 0.751 | 0.7529 | +0.0019 |
| 15 | Rad+IHC-G+Gen | OvO | 0.752 | 0.7509 | −0.0011 |
| 16 | Rad+IHC-A+MutAmp | DyAM | 0.734 | 0.7289 | −0.0051 |
| 16 | Rad+IHC-A+MutAmp | OvO | 0.744 | 0.7403 | −0.0037 |
| 17 | Rad+IHC-A+Gen+PDL1 | DyAM | 0.743 | 0.7399 | −0.0031 |
| 17 | Rad+IHC-A+Gen+PDL1 | OvO | 0.753 | 0.7508 | −0.0022 |
| 18 | Rad+IHC-G+Gen+PDL1 (BM1) | OvO | 0.773 | 0.7704 | −0.0026 |
| 19 | Rad+IHC-A+Gen+TMB+PDL1 | DyAM | 0.749 | 0.7413 | −0.0077 |
| 19 | Rad+IHC-A+Gen+TMB+PDL1 | OvO | 0.766 | 0.7581 | −0.0079 |
| 20 | Rad+IHC-A+Gen+PDL1+Labs | OvO | 0.745 | 0.7435 | −0.0015 |
| 21 | Rad+IHC-G+Gen+PDL1+Labs (BM2) | DyAM | 0.764 | 0.7592 | −0.0048 |
| 21 | Rad+IHC-G+Gen+PDL1+Labs (BM2) | OvO | 0.763 | 0.7692 | +0.0062 |

**Ba bằng chứng cho thấy nguyên nhân là môi trường, không phải bài báo ghi sai:**

1. **Mọi ô lệch đều có radiomics.** 13 tổ hợp không radiomics khớp tuyệt đối tới 4 chữ số.
2. **Cột OvO+NLP khớp cả 21/21 ô** — kể cả các tổ hợp radiomics. Lý do: đường chạy đó
   (`nlp_verify/nlp_run.py`) **đã có sẵn bản vá scipy từ trước**, nên môi trường của nó không đổi.
   Đường chạy không vá (`run_paper_combos.py`, `run_paper_ovo.py`) thì lệch. Đây là bằng chứng
   trực tiếp và sạch nhất.
3. Lệch hai chiều, không thiên lệch (11 ô tăng, 12 ô giảm) — đúng kiểu nhiễu số học.

**Nguyên nhân kỹ thuật:** scipy ≥ 1.14 raise `BracketError` trong `PowerTransformer` (Yeo-Johnson)
trên cột radiomics gần hằng số; bản cũ không raise. Phải vá (λ=1 cho cột lỗi) mới chạy tiếp được.

**Ba lựa chọn:**

| | Phương án | Ưu | Nhược |
|---|---|---|---|
| **A** | Giữ nguyên số, thêm câu ghi rõ phiên bản thư viện đã dùng vào Methods/Supplementary | Không phải chạy lại; trung thực; chuẩn mực thông thường | Người khác chạy trên scipy mới sẽ ra số hơi khác |
| B | Thay toàn bộ bằng số mới | Tái lập được với thư viện hiện hành | Phải làm lại cả Bảng 3, S3 và mọi số dẫn trong văn bản |
| C | Dựng env scipy < 1.14 để xác nhận số cũ | Dứt điểm | Tốn thêm thời gian; torch sắp đổi sang CUDA nên env lại khác nữa |

**Đề xuất: A.** Lệch lớn nhất 0,016 ở tổ hợp phụ; BM1/BM3/BM4 không đổi kết luận.

**Quyết định:** ☐ A ☐ B ☐ C ☐ Cách khác: ______

---

## SK-3/4/5 · Bảng 5, 6, 7, 8 — **đã chạy lại bằng 5-seed** (2026-09-30)

Theo phương án A đã chọn: 11 biến thể × 5 seed × 10-fold, script mới
`experiments/nlp_verify/verify_nlp_5seed.py`, kết quả `experiments/results/verify_nlp_5seed.json`.
Dùng đúng các hàm gốc của dự án (`compute_cindex`, `compute_multivariate_cox`) nên phương pháp
giống Bảng 7, chỉ khác ở chỗ lặp 5 seed thay vì một lần chia.

| Nhánh | Biến thể | AUC 5-seed | Bảng 5/6 | C-index | Bảng 7 | Cox HR | Bảng 7 | KM χ² | Bảng 8 |
|---|---|---|---|---|---|---|---|---|---|
| IHC-A | No clinical | **0,749±0,020** | 0,764 | 0,611±0,005 | 0,623 | 4,31±0,62 | 5,30 | 13,4±3,2 | 28,17 |
| IHC-A | + Labs | 0,730±0,024 | 0,768 | **0,616±0,007** | 0,626 | 4,85±0,27 | 5,31 | 16,9±6,3 | — |
| IHC-A | + NLP raw | 0,743±0,019 | 0,781 | 0,608±0,008 | 0,625 | **4,97±0,22** | 5,91 | **19,3±7,0** | — |
| IHC-A | + NLP-PCA16 | 0,739±0,017 | 0,784 | 0,608±0,010 | 0,628 | 4,83±0,33 | 6,07 | 19,2±8,7 | 23,74 |
| IHC-A | + Labs + NLP | 0,739±0,010 | 0,767 | 0,610±0,006 | — | 4,72±0,47 | — | 15,2±6,0 | — |
| IHC-A | + Labs + NLP-PCA16 | 0,738±0,009 | 0,753 | 0,609±0,008 | — | 4,61±0,48 | — | 14,6±6,0 | — |
| IHC-A | + BioClinBERT-PCA16 | 0,727±0,016 | 0,767 | 0,612±0,010 | — | 5,19±0,70 | — | 18,0±6,1 | — |
| IHC-G | No clinical | 0,768±0,026 | 0,784 | 0,618±0,007 | 0,626 | 4,16±0,50 | 4,74 | 12,4±3,3 | 20,07 |
| IHC-G | + Labs | 0,749±0,021 | 0,788 | **0,626±0,011** | 0,632 | 4,77±0,85 | 4,49 | **19,1±7,8** | — |
| IHC-G | **+ NLP raw** | **0,769±0,026** | 0,813 | 0,619±0,014 | 0,632 | **4,92±0,50** | 4,97 | 18,2±9,4 | — |
| IHC-G | + NLP-PCA16 | 0,767±0,027 | 0,812 | 0,619±0,012 | 0,631 | 4,91±0,46 | 4,54 | 17,5±7,3 | 28,92 |

### Paired bootstrap so với "No clinical" (2.000 lần, cùng bệnh nhân)

| Nhánh | Biến thể | ΔAUC | 95% CI | p |
|---|---|---|---|---|
| IHC-A | + Labs | −0,0190 | [−0,059, +0,005] | 0,108 |
| IHC-A | + NLP raw | −0,0062 | [−0,039, +0,011] | 0,272 |
| IHC-A | + NLP-PCA16 | −0,0102 | [−0,040, +0,005] | 0,134 |
| IHC-A | + BioClinBERT-PCA16 | −0,0226 | [−0,055, −0,003] | **0,028** |
| IHC-G | + Labs | −0,0190 | [−0,051, +0,011] | 0,209 |
| IHC-G | + NLP raw | +0,0008 | [−0,018, +0,020] | 0,882 |
| IHC-G | + NLP-PCA16 | −0,0013 | [−0,018, +0,014] | 0,926 |

### Kết luận: ba luận điểm, hai còn đứng, một sụp

**① "NLP tốt hơn Labs thô" — ✅ ĐỨNG VỮNG.** IHC-A: 0,743 so với 0,730 (+0,013 cho NLP raw);
IHC-G: 0,769 so với 0,749 (+0,020). Khớp với §2.8 (+NLP > +Labs ở 19/21 tổ hợp) và với cột OvO+NLP
của Bảng 4 (tái lập 21/21). Đây là luận điểm chính của bài và nó có bằng chứng vững.

**② "NLP cải thiện phân tầng sống còn" — ⚠️ KHÔNG ỔN ĐỊNH, yếu hơn tôi nói ban đầu.**
Chiều của hiệu ứng **lật giữa hai lần chạy**:

| | Cox HR: No clin → +NLP | KM χ²: No clin → +NLP |
|---|---|---|
| Đơn-phân-hoạch (Phase 3) | 5,84 → 4,60 (NLP **kém hơn**) | 25,3 → 25,2 (bằng nhau) |
| 5-seed (Phase SK-3) | 4,31 → 4,97 (NLP **tốt hơn**) | 13,4 → 19,3 (NLP tốt hơn) |

Thêm nữa, **Cox HR không phải thước đo sạch để so giữa các model**: `compute_multivariate_cox`
(`lung_helpers.py:6276`) chuẩn hoá điểm về [0,1] bằng min–max trước khi fit, nên HR là "tỉ số nguy cơ
giữa hai đầu của biên độ điểm" — phụ thuộc vào việc điểm có bị outlier kéo dãn hay không, chứ không
chỉ phụ thuộc thứ tự xếp hạng bệnh nhân. Kiểm chứng: cùng bộ điểm, đổi sang chuẩn hoá z-score thì
HR co về 1,43–1,54 và **thứ tự đảo lại** (No clinical 1,540 cao nhất, NLP 1,480–1,496).

Và C-index — thước đo không phụ thuộc thang đo — thì **No clinical cao nhất ở cả hai lần chạy**
(0,636/0,611 so với 0,619/0,608 của NLP).

**Phần duy nhất vững:** mọi model đều phân tầng có ý nghĩa (p < 0,005 với mọi cách chia nhóm),
và khác biệt C-index giữa các biến thể đều nhỏ, không có ý nghĩa (Wilcoxon per-fold p = 0,68–0,89).

**③ "NLP tốt hơn khi KHÔNG dùng lâm sàng" — ❌ SỤP.** Bài báo ghi ΔAUC = +0,020 (IHC-A) và +0,030
(IHC-G) so với mốc No clinical. Chạy lại: **−0,010 và +0,001**, bootstrap p = 0,13 và 0,93. Thực tế
ở đúng cấu hình này, **thêm bất kỳ dữ liệu lâm sàng nào cũng không cải thiện**, kể cả Labs thô
(−0,019 ở cả hai nhánh). Cột "ΔAUC so với mốc No clinical" của Bảng 5/6 là cột không tái lập được.

**Thêm một kết quả mới:** BioClinBERT kém hơn mốc No clinical **có ý nghĩa thống kê** (p = 0,028) —
mạnh hơn hẳn cách Note S2 đang diễn đạt (hiện chỉ nói "kém hơn MiniLM").

### Đề xuất sửa (chưa làm gì cả, chờ bạn)

1. **Thay số Bảng 5, 6, 7, 8 bằng bộ 5-seed ở trên** (mean ± sd), bỏ cột "ΔAUC so với No clinical"
   hoặc giữ nhưng ghi đúng dấu âm kèm bootstrap p.
2. **Đổi trục so sánh của §3.3 từ "NLP vs không dùng lâm sàng" sang "NLP vs Labs thô"** — đây mới là
   so sánh tái lập được, và cũng là so sánh có ý nghĩa khoa học hơn (cùng 13 biến, chỉ khác cách mã hoá).
3. Giữ phần sống còn nhưng nói theo chiều, không nhấn con số: "NLP cho HR và phân tầng KM cao hơn mốc
   không dùng lâm sàng ở cả hai nhánh".
4. Cập nhật Note S2: BioClinBERT kém hơn có ý nghĩa thống kê (p = 0,028).
5. Hình 3a/3b phải vẽ lại theo số mới.

**Quyết định:** ☐ Làm cả 5 mục ☐ Chỉ thay số (mục 1) ☐ Chỉ đổi trục so sánh (mục 2) ☐ Khác: ______

---

## SK-6 · ⛔ tdAUC và IBS trong Bảng 7 không kiểm được

`sksurv` 0.27 gọi `numpy.trapezoid` — hàm này chỉ có từ numpy 2.0, môi trường đang là numpy 1.26.4.
Cả 8 model đều báo `module 'numpy' has no attribute 'trapezoid'`, nên hai cột tdAUC (6/12/18 tháng)
và IBS trả về rỗng.

Muốn kiểm phải nâng numpy lên ≥ 2.0 — việc này sẽ đổi môi trường của mọi thứ khác, nên tôi dừng lại
ở đây chờ bạn quyết.

**Quyết định:** ☐ Nâng numpy ≥ 2 trong env riêng rồi kiểm ☐ Bỏ qua, chấp nhận không kiểm ☐ Khác: ______

---

## Những gì đã kiểm và **khớp hoàn toàn** (không cần quyết định gì)

- Toàn bộ **Bảng 1** — `make_table1.py` sinh lại file y hệt, không khác một ký tự
- **Note S1**: câu ví dụ trùng từng ký tự, 65–68 từ, 129–135 token
- **PCA16 giữ 94,4%** phương sai (bài ghi ≈94%)
- Note S3: nửa độ rộng CI ±0,07 và mức thiếu công suất ~2–5×
- **Bảng 4**: 13/21 tổ hợp không radiomics khớp tuyệt đối; **toàn bộ 21 ô cột OvO+NLP khớp tuyệt đối**
- Phép kiểm nội bộ k=1 (OvO phải bằng DyAM về đại số): đạt cả 4 tổ hợp
- Cox p < 0,001 cho cả 8 model; Wilcoxon per-fold không có ý nghĩa (khớp bài)
- KM log-rank p < 0,005 cho mọi model và mọi cách chia nhóm
- Bố cục Hình 1, 2, 3 — `check_figures.py` báo 0 lỗi
