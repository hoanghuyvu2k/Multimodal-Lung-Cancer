# Peer-Review Report — Manuscript `paper/main.tex`

**Tạp chí giả định:** MDPI *Cancers* (theo `\documentclass[cancers,article,...]{mdpi}`)
**Tiêu đề:** *NLP-Augmented Multimodal Attention Fusion for Immunotherapy Response Prediction in Non-Small Cell Lung Cancer*
**Ngày review:** 2026-09-09
**Người review:** Reviewer #1 (đóng vai giám khảo)

**Khuyến nghị: MAJOR REVISION.**
Ý tưởng NLP-encoding biến lâm sàng dạng bảng là đóng góp mới, có giá trị và được kiểm chứng
lặp lại tốt (19/21 tổ hợp, 2 backbone độc lập). Tuy nhiên bản thảo hiện tại **báo cáo thiếu một
baseline quan trọng đã chạy và cho kết quả tốt hơn cả hai mô hình đề xuất**, **không dùng đến 2
cohort external đã mô tả trong Table 1/Figure 1**, có **sai lệch giữa Methods và code thực thi**
(hyperparameter, công thức), và một loạt **mâu thuẫn số liệu nội tại** giữa Abstract / Results /
Tables / Supplementary. Đây đều là những điểm reviewer sẽ bắt và đủ để bị reject nếu không sửa.

---

## 0. Cách dùng file này

Mỗi mục có: **[Mức độ]** → *Bằng chứng* → **Yêu cầu sửa**. Tick `[x]` khi xong.
Mức độ: **BLOCKER** (không sửa thì reject) · **MAJOR** (bắt buộc sửa trong revision) ·
**MINOR** (sửa cho chỉn chu).

---

## 1. Điểm mạnh (ghi nhận)

- Câu hỏi nghiên cứu rõ, đóng góp phát biểu tường minh (2 contribution).
- Kỷ luật đánh giá lặp seed (5 seeds × 10-fold, 21 tổ hợp) tốt hơn mặt bằng chung của mảng radiomics fusion.
- Bản thảo **tự phê bình** kết quả single-partition của mình (§3.2) — hiếm và đáng khen.
- Feature selection radiomics thực sự nằm **trong fold** (`lung_helpers.py:2586-2595`, `l1_filter_features_list`
  loại `valid_px` trước khi chọn feature) → **không có leakage**. Điểm này đang bị giấu, phải nói rõ ra
  (xem M12) vì đó là điểm mạnh về phương pháp luận.
- Kết quả âm tính (BioClinBERT kém hơn, combined encoding kém hơn) được báo cáo trung thực.

---

## 2. BLOCKER — phải giải quyết trước khi bản thảo có thể được xét tiếp

### [ ] B1. Thiếu baseline không-attention (`uniform_avg`), trong khi baseline đó **bằng hoặc hơn cả DyAM lẫn OvO**

*Bằng chứng (từ log nội bộ của chính nhóm, `result/training-results.md` §0, §2.2, §2.5):*

| BM | uniform_avg | Original (DyAM) | OvO |
|---|---|---|---|
| BM1 | **0.7746 ± 0.019** | 0.7595 ± 0.022 | 0.7728 ± 0.020 |
| BM2 | **0.7665 ± 0.011** | 0.7638 ± 0.018 | 0.7632 ± 0.010 |
| BM3 | 0.7110 ± 0.015 | 0.7095 ± 0.018 | **0.7132 ± 0.015** |
| BM4 | 0.7191 ± 0.008 | 0.7072 ± 0.009 | 0.7183 ± 0.012 |

`uniform_avg` là **trung bình có mask, không có tham số học nào**. Log §2.5 còn kết luận thẳng:
*"OvO ≈ uniform_avg ở 5-seed … OvO về bản chất gần uniform, không phải cải tiến thật"*.

Bản thảo hiện **không nhắc một chữ nào** đến baseline này, trong khi toàn bộ §3.1 lại kết luận
*"Multimodal fusion is therefore the primary driver of predictive performance"* và §3.2/Discussion
bán câu chuyện "cơ chế attention". Một reviewer đọc code sẽ hỏi ngay: *nếu trung bình cộng không
tham số cũng ngang bằng, thì attention học được gì?* Việc có số liệu mà không báo cáo là vấn đề
**liêm chính báo cáo**, không chỉ là thiếu sót.

**Yêu cầu sửa:**
- [ ] Thêm cột `uniform_avg` vào Table 4 (bảng 21 tổ hợp) và vào 4 primary benchmark.
- [ ] Thêm 1 đoạn trong §3.2 nêu rõ: attention học được **không vượt** trung bình có mask ở n=247.
- [ ] Sửa lại framing của Discussion + Conclusion: đóng góp thật của bài là **NLP-encoding**
      (attention-agnostic), còn OvO nên hạ xuống mức "đối chứng kiến trúc, không mang lại lợi thế".
      Chính log §2.9 đã kết luận đúng như vậy — bản thảo chỉ cần trung thực theo nó.
- [ ] Thêm vào Discussion 1–2 câu diễn giải (đây lại là một **finding hay**, không phải điểm yếu):
      ở cỡ mẫu này, độ phức tạp của cơ chế fusion không phải là nút thắt; chất lượng biểu diễn
      đầu vào mới là.

---

### [ ] B2. Con số "0.760 → 0.783" trong §3.1 trộn **hai backbone khác nhau**

*Vị trí:* `sections/results.tex:20-24`.

> "Fusing four data sources … raised the AUC to $0.760 \pm 0.022$ … adding NLP-encoded clinical
> variables raised it further to $0.783 \pm 0.016$"

- 0.760 = **DyAM attention** (log §2.2, BM1 Original).
- 0.783 = **uniform_avg + NLP** (log §2.8: backbone `uniform_avg`, `nlp_combos.json`) — §3.3 có thừa
  nhận dùng "masked averaging" nhưng §3.1 thì không, và người đọc mặc định là cùng mô hình.

Phân rã đúng: 0.760 (DyAM) → 0.772–0.775 (đổi sang uniform_avg, **chưa có NLP**) → 0.783 (thêm NLP).
Tức **hơn nửa mức tăng +0.023 là do đổi cơ chế fusion, không phải do NLP**. Đây là so sánh
táo–với–cam và làm phóng đại đóng góp chính của bài.

**Yêu cầu sửa:**
- [ ] Chỉ so cùng backbone: `uniform_avg` base 0.772 vs `uniform_avg` +NLP 0.783 (Δ=+0.011),
      hoặc dùng cặp sạch nhất trong log §2.8: **+Labs 0.766 vs +NLP 0.783 (Δ=+0.017)**.
- [ ] Nêu rõ backbone ngay tại chỗ trong §3.1, không để người đọc phải suy ra từ §3.3.

---

### [ ] B3. Hai cohort external được mô tả trong Table 1 + Figure 1 nhưng **không có một kết quả nào** trong Results

*Vị trí:* `sections/methods.tex:22-28` ("both held out entirely for external evaluation"),
Table 1 (cột Validation), Figure 1 caption. Results/Discussion: **không có gì**.

Nhóm **đã chạy** external validation (log §2.1, §2.3):

| Modality | ext AUC neural [CI] | ext AUC LR | n |
|---|---|---|---|
| Pathology (IHC-G) → path_valid | **0.767** [0.62–0.89] | 0.765 | 52 |
| Radiomics → rad_valid | **0.425** [0.20–0.65] | 0.461 | 46 |

Và external cho NLP thì **không nhất quán**: ΔC = +0.011 (IHC-A) / −0.025 (IHC-G) / −0.009 (Rad);
IHC-G + NLP-PCA16 làm **tụt** AUC ngoài mẫu 0.767 → 0.705.

Mô tả cohort external rồi im lặng về kết quả là điều reviewer chắc chắn hỏi ("what happened to
the validation cohorts?"), và câu trả lời hiện có lại đi ngược claim của bài.

**Yêu cầu sửa (chọn 1, khuyến nghị (a)):**
- [ ] **(a)** Thêm mục §3.5 "External validation" báo cáo đầy đủ cả 3 con số trên, kể cả kết quả
      xấu; đưa vào Limitations rằng radiomics **không** generalize (AUC < 0.5) và NLP-effect chưa
      xác nhận được ngoài mẫu. Đây là bài báo mạnh hơn, không yếu hơn — pathology GLCM generalize
      tốt (0.767) là một finding đáng giá.
- [ ] **(b)** Nếu không báo cáo: **gỡ hẳn** cột Validation khỏi Table 1, gỡ cohort external khỏi
      Methods và khỏi Figure 1, và nói rõ trong Limitations là chưa có external validation.
      (Không được để tình trạng nửa vời như hiện nay.)

---

### [ ] B4. Hyperparameter trong Methods **sai so với code đã chạy**

*Vị trí:* `sections/methods.tex:299-301`.

| Bản thảo ghi | Thực tế đã chạy |
|---|---|
| `\ell_1` trên attention, **α = 1.0** | **α = 0.001** (`experiments/common/benchmarks.py`; notebook: `'alpha':0.001`) |
| `\ell_2` weight decay, **β = 1.0** | **β = 0.0** (không dùng weight decay) |
| lr 0.01, 125 epochs | đúng |

Sai 3 bậc độ lớn ở α và sai hoàn toàn ở β → ai đọc bài mà chạy lại theo Methods sẽ ra kết quả khác.
Ngoài ra Methods **không khai báo** các thiết lập quyết định kết quả:
`cross_modality_enabled=False`, `attention_gate_enabled=True`, `no_scale` cho NLP, `folds=10` với
`KFold(shuffle=True, random_state=0)` (**không phải** *stratified* như Methods đang ghi ở dòng 303 —
code dùng `KFold`, không phải `StratifiedKFold`).

**Yêu cầu sửa:**
- [ ] Sửa α = 0.001, β = 0 (hoặc ghi "no weight decay").
- [ ] Sửa "10-fold **stratified** cross-validation" → "10-fold cross-validation (`KFold`, shuffled)"
      — hoặc chạy lại bằng StratifiedKFold nếu muốn giữ chữ "stratified" (khuyến nghị: đổi chữ,
      đừng chạy lại).
- [ ] Bổ sung 1 bảng hyperparameter đầy đủ trong Supplementary (kể cả `cross_modality_enabled`,
      `attention_gate_enabled`, `no_scale`, seeds `[42, 7, 123, 2024, 31337]`).
- [ ] Nói rõ hyperparameter **không được tune** (lấy từ Vanguri 2022 / cấu hình chuẩn) — nếu có tune
      thì phải mô tả nested CV; nếu không tune thì nói thẳng, đó là câu trả lời hợp lệ.

---

### [ ] B5. Phương trình trong bản thảo **không khớp implementation**

| Bản thảo | Code (`lung_helpers.py`) | Sai lệch |
|---|---|---|
| Eq. (2): $s_i = \text{softplus}(W^\top x_i) / \lVert x_i\rVert_2$ | chia cho `l_feature_factor` = **số chiều $d_i$** (dòng 1022, 1087) | Chia cho **hằng số $d_i$**, không phải chuẩn L2 của vector |
| Eq. (5): $\mu_{-i} = \frac{1}{N-1}\sum_{j\ne i} s_j$ | chia cho `n_others = others_mask.sum()` = số modality **khả dụng** khác | Với bệnh nhân thiếu modality, mẫu số ≠ N−1 |
| Eq. (6): $a_i \propto \sigma(s_i-\mu_{-i})\cdot m_i$ | `attn_scores = mask * softplus(sigmoid(...))` (dòng ~1250) | Code có **thêm một softplus** sau sigmoid, bản thảo không có |
| Fig. 3 caption: "pairwise scores … column-summed into $s_j$" | với `cross_modality_enabled=False`, `matrix_mask` bị nhân với **identity** (dòng 1060-1069) → chỉ còn đường chéo | Không hề có tương tác pairwise trong cấu hình đã chạy |

**Yêu cầu sửa:**
- [ ] Viết lại Eq. (2), (5), (6) đúng theo code (hoặc sửa code cho khớp bài — nhưng khi đó phải chạy lại).
- [ ] Sửa caption Figure 3 và mô tả trong §2.4 để phản ánh cấu hình `cross_modality_enabled=False`.
- [ ] Nêu rõ trong Methods rằng mọi thí nghiệm chạy với `cross_modality_enabled=False`.

---

### [ ] B6. Claim "OvO đơn giản hơn: $N$ tham số vs $N\times N$" **không đúng với cấu hình đã đánh giá**

*Vị trí:* Abstract, `sections/results.tex` §3.2 (đoạn cuối), Discussion, Conclusion — lặp 4 lần.

Vì `cross_modality_enabled=False`, ma trận attention của mô hình cooperative bị **mask về đường chéo**:
$N^2-N$ lớp linear ngoài đường chéo vẫn được khởi tạo nhưng **luôn bị nhân 0**, không đóng góp gì vào
forward pass. Về mặt hàm số, cả hai mô hình đều dùng đúng $N$ lớp attention. Lợi thế "tiết kiệm tham số"
mà bài đang bán là **artefact của việc đếm tham số khởi tạo**, không phải tham số hữu dụng.

**Yêu cầu sửa (chọn 1):**
- [ ] **(a)** Bỏ hẳn claim parameter-efficiency (khuyến nghị — kết hợp với B1, OvO nên được hạ vai trò).
- [ ] **(b)** Giữ claim nhưng phải chạy lại với `cross_modality_enabled=True` để $N\times N$ thực sự
      được dùng, và báo cáo số tham số **hữu dụng** cùng thời gian train của cả hai.

---

## 3. MAJOR — bắt buộc sửa trong vòng revision

### [ ] M7. Sai lầm thống kê: dùng "hai CI 95% không chồng nhau" làm kiểm định

*Vị trí:* `sections/methods.tex:317-320`.

> "Two models were considered statistically significantly different when their 95\% CIs did not
> overlap, consistent with standard practice in oncology imaging studies."

Đây là kiểm định **quá bảo thủ và sai** cho hai AUC **tương quan** (cùng bệnh nhân, chung phần lớn
modality → ρ ≈ 0.9 như chính Supplementary Note S3 giả định). DeLong 1988 — bài mà nhóm đang cite —
tồn tại chính là để làm **paired test**, không phải để vẽ 2 CI rồi nhìn xem có chồng nhau không.
Hệ quả: toàn bộ kết luận "không significant" ở §3.3 có thể là **false negative do chọn sai test**.
Câu "consistent with standard practice" cũng không có citation.

**Yêu cầu sửa:**
- [ ] Chạy **DeLong paired test** (hoặc paired bootstrap trên cùng bệnh nhân, nhóm đã có hạ tầng ở
      `experiments/`) cho mọi cặp so sánh trong Table 5, Table 6, và báo cáo p-value từng cặp.
- [ ] Bỏ tiêu chí "CI không chồng nhau" khỏi Methods, hoặc giữ CI chỉ để mô tả độ chính xác.
- [ ] Điều chỉnh lại các câu "did not reach statistical significance" ở §3.3 và Discussion theo
      kết quả test đúng.

### [ ] M8. Trộn hai protocol đánh giá; con số headline lại là con số bài tự cảnh báo là không tin cậy

- Table 4 (21 tổ hợp) = **5-seed** (robust).
- Table 5, 6 (AUC + DeLong CI), Table 7, 8 (survival) = **single-partition** (`KFold(random_state=0)`).
- Abstract + Conclusion đặt **0.813** làm con số chủ đạo → đó là **single-partition**.

Trong khi đó §3.2 dành cả đoạn để lập luận rằng single-partition không đáng tin và rank có thể đảo.
Bài đang tự mâu thuẫn: dùng chuẩn khắt khe cho OvO (đóng góp yếu) nhưng chuẩn dễ dãi cho NLP (đóng góp
mạnh). Reviewer sẽ gọi đây là **double standard**. Log §2.8/§2.9 cũng ghi rõ: 5-seed cao nhất toàn dự án
chỉ ~0.783, **không mô hình nào vượt 0.80 trên chỉ số robust**.

**Yêu cầu sửa:**
- [ ] Abstract + Conclusion: đưa số **5-seed** lên làm số chính (0.783 ± 0.016), số single-partition
      0.813 chỉ nêu kèm chú thích "single-partition, not robust".
- [ ] Thêm 1 câu trong §3.3 nói rõ Tables 5–8 dùng protocol single-partition (để tương thích với
      Vanguri 2022) và vì sao.
- [ ] Bỏ/điều chỉnh câu "approaches the upper range of discrimination achievable" trong Discussion
      (dòng 17-19) — nó dựa trên 0.813.

### [ ] M9. Không hiệu chỉnh đa so sánh

21 tổ hợp × 3–4 mô hình × nhiều endpoint (AUC, C-index, tdAUC ×3, HR, IBS, log-rank), nhưng không có
một dòng nào về multiplicity. Các phát biểu kiểu "12 of 21", "19 of 21", "15 of 21" là **đếm mô tả**
— bài có nói điều này cho phần OvO (tốt) nhưng **không nói** cho phần NLP (19/21), nơi nó được dùng
như bằng chứng chính.

**Yêu cầu sửa:**
- [ ] Thêm câu tương tự cho §3.3: các đếm 19/21 là descriptive, test chính thức chỉ chạy ở BM.
- [ ] Hoặc: dùng một kiểm định phù hợp cho "nhiều tổ hợp cùng hướng" (ví dụ sign test / Wilcoxon trên
      21 Δ ghép cặp) — với 19/21 cùng dấu, sign test cho p ≈ 0.0004, **đây là con số nên đưa vào bài**.
      Rất đáng làm: nó biến kết quả "không significant" thành có bằng chứng thống kê thật.

### [ ] M10. Thiếu control experiment cho claim cốt lõi của NLP

Bài kết luận sentence embedding "bắt được tương tác phi tuyến giữa các biến lâm sàng"
(Discussion dòng 22-33). Nhưng chưa loại trừ được các giải thích cạnh tranh tầm thường:

1. Chỉ là **tăng số chiều** (13 → 384) giúp lớp linear + attention có thêm capacity.
2. Chỉ là một **phép biến đổi phi tuyến bất kỳ**, không cần "ngữ nghĩa y khoa".
3. Chỉ là hiệu ứng của việc **tắt RobustScaler** (`no_scale`) cho riêng modality đó.

Ngoài ra log §2.3 ghi **NLP đơn-modality 0.549 < Labs thô 0.573** — tức bản thân embedding chứa
*ít* thông tin phân biệt hơn 13 biến gốc, chỉ khi fusion mới có lợi. Đây là dữ kiện quan trọng và
bài **không báo cáo**.

**Yêu cầu sửa:**
- [ ] Thêm ít nhất 2 trong 4 control sau (đều rẻ, chạy trên harness sẵn có):
      (a) random projection 13 → 384 chiều; (b) MLP/RBF random features từ 13 biến;
      (c) câu văn với **giá trị bị xáo trộn giữa bệnh nhân** (giữ nguyên cú pháp, phá ngữ nghĩa);
      (d) transformer khởi tạo ngẫu nhiên (không pretrain).
      Nếu (a)/(d) cũng cải thiện tương đương → kết luận "ngữ nghĩa" phải rút lại.
- [ ] Thêm ablation `no_scale` bật/tắt để tách hiệu ứng scaling.
- [ ] Báo cáo AUC đơn-modality của NLP (0.549) vs Labs (0.573) và diễn giải: lợi ích là
      **hiệu ứng fusion**, không phải embedding mạnh hơn về mặt tuyệt đối. Điều này thực ra **củng cố**
      câu chuyện của bài (bổ trợ, không trùng lặp) nếu viết đúng.

### [ ] M11. Tuyên bố đạo đức / dữ liệu không phù hợp với bản chất dữ liệu

*Vị trí:* `sections/methods.tex:40-45`, back-matter `main.tex`.

Dữ liệu là MSK-MIND (Vanguri et al., Nat Cancer 2022), tức **phân tích thứ cấp trên bộ dữ liệu đã
công bố**. Nhưng bản thảo viết "approved by the IRB of [INSTITUTION] (Protocol No. [IRB-NUMBER])"
và "All patients provided written informed consent" — như thể nhóm tự thu thập. Với dữ liệu thứ cấp
đã de-identified, câu đúng thường là *IRB waiver / exempt* + trích dẫn nguồn dữ liệu gốc.

**Yêu cầu sửa:**
- [ ] Viết lại Ethics: nêu rõ đây là phân tích thứ cấp bộ dữ liệu công bố bởi Vanguri et al. 2022,
      nguồn truy cập (Synapse), và trạng thái IRB thực tế của nhóm.
- [ ] Điền toàn bộ placeholder: `[DEPARTMENT]`, `[UNIVERSITY]`, `[IRB-NUMBER]`, `[FUNDING]`,
      `[REPO]`, tên tác giả, email, ORCID.
- [ ] Data Availability phải trỏ tới nguồn dữ liệu gốc, không chỉ repo code.

### [ ] M12. Feature selection chưa được mô tả (đang bị "giấu" một bước có giám sát)

*Vị trí:* `sections/methods.tex:118-123` chỉ mô tả 2 bộ lọc ICC (<0.15) và outlier (z > 6).

Thực tế `l1_filter_features_list` → `select_radiomics_features_elastic` (`lung_helpers.py:745-766`)
còn chạy **elastic-net logistic regression có giám sát** (`l1_ratio=0.5`, `C=1.0`,
`class_weight='balanced'`, `solver='saga'`, sau `PowerTransformer`) để chọn feature radiomics.
Bước này **có dùng nhãn** — nếu người đọc không biết nó nằm trong fold, họ sẽ nghi ngờ leakage.

**Yêu cầu sửa:**
- [ ] Bổ sung mô tả đầy đủ pipeline chọn feature (3 bước, kèm siêu tham số).
- [ ] **Nói rõ** bước này được fit lại trong từng training fold, loại trừ hoàn toàn fold kiểm định
      (`lung_helpers.py:2595`) → không leakage. Đây là điểm cộng, nên viết ra.
- [ ] Báo cáo số feature còn lại sau lọc (trung bình ± sd qua 10 fold) — hiện Supplementary Table S2
      đang để `[TO VERIFY]` (xem C6).

### [ ] M13. Ngưỡng phân tầng KM "risk score = 0" chưa được mô tả đúng bản chất

*Vị trí:* `sections/methods.tex:338-343`.

Điểm rủi ro được chuẩn hoá bằng `response_zscore` (`lung_helpers.py:801-812`), trong đó
`mu = find_optimal_cutoff(target, input)` — tức **ngưỡng Youden tối ưu hoá trên nhãn đáp ứng nhị
phân của tập train**, còn `std` cũng từ train. Vậy "risk = 0" **không phải** một ngưỡng trung tính,
mà là ngưỡng đã được tối ưu; và vì pooled out-of-fold, **10 fold có 10 ngưỡng khác nhau** được gộp
chung vào một phân tầng KM.

Không có leakage (ngưỡng học trên train), nhưng người đọc **phải** được biết, nếu không họ sẽ hiểu
sai là ngưỡng tiên nghiệm.

**Yêu cầu sửa:**
- [ ] Mô tả đúng cơ chế z-score hoá và nguồn gốc ngưỡng 0 trong Methods.
- [ ] Thêm sensitivity analysis: phân tầng theo **median risk score** (ngưỡng không tối ưu hoá) và
      báo cáo lại χ²/log-rank. Nếu kết luận không đổi → luận điểm mạnh hơn hẳn.
- [ ] Giải thích luôn tiêu chí `χ² > 7.88 (p < 0.005)`: hiện đang viết "consistent with oncology
      reporting standards" mà **không có citation**. Hoặc cite, hoặc dùng α = 0.05 thông thường.

### [ ] M14. Yêu cầu bắt buộc của *Cancers* chưa đáp ứng

- [ ] **Thiếu "Simple Summary"** — *Cancers* **bắt buộc** có Simple Summary (tóm tắt phổ thông
      ≤ 200 từ) đặt trước Abstract. Hiện `main.tex` không có `\simplesumm{}`. Đây là lỗi
      **desk-reject** ở khâu pre-check.
- [ ] **Thiếu TRIPOD+AI checklist** — bài là một prediction-model study; *Cancers* và hầu hết tạp chí
      ung thư yêu cầu nộp kèm checklist TRIPOD (bản AI 2024). Nên thêm 1 câu trong Methods
      ("reported according to TRIPOD+AI; checklist in Supplementary") + file checklist.
- [ ] **Thiếu flow diagram bệnh nhân** — từ bao nhiêu bệnh nhân ban đầu, loại trừ bao nhiêu, vì sao,
      còn 247/50/71. Hiện Methods chỉ nêu tiêu chí nhận, không nêu số loại trừ.
- [ ] Abstract 192 từ (≤ 200 ✓), thân bài 4.606 từ ✓ — 2 mục này đạt.

---

## 4. MAJOR — mâu thuẫn số liệu (đã kiểm chứng lại từ dữ liệu gốc)

> Tôi đã tính lại Table 1 trực tiếp từ
> `../datasets/18193mskmindprojectm-omnibusinventory_data_2021-12-20_1540-with-tb-and-scanner.csv`
> join với `final_cohort_listing.csv` (n = 247, khớp chính xác).

### [ ] C1. Table 1 — Male sex "168 (45.9%)" là **bất khả thi về số học**

168/247 = **68.0%**, không phải 45.9%. Dữ liệu thật: `sex` có 113 vs 134 → **113 (45.7%)** hoặc
**134 (54.3%)** tuỳ quy ước mã hoá. Con số 45.9% gần đúng với nhóm 113, nên nhiều khả năng **n bị
gõ nhầm** (168 → 113). Phải xác định lại quy ước `sex=1/2` trước khi sửa.

### [ ] C2. Table 1 — các thống kê khác lệch so với dữ liệu gốc

| Mục | Bản thảo | Tính lại từ dữ liệu |
|---|---|---|
| Age, mean (range) | 66.8 (**30**–93) | 66.9 (**38**–93) |
| ECOG 0 / 1 / ≥2 | 12.5 / 78.7 / 8.8 % | 11.3 / 78.5 / **10.1** % |
| Adeno / Squamous / Other | 73.1 / 15.2 / 11.7 % | 74.5 / 14.6 / 10.9 % |
| Response label 0 / 1 | "≈62 (25.1%)" / "≈185" | **62 (25.1%) / 185 (74.9%)** — chính xác, bỏ dấu ≈ |
| Median PFS, events | 2.7 m, 209 (84.6%) | ✓ khớp |

- [ ] Chạy lại script sinh Table 1 từ dữ liệu và cập nhật toàn bộ ô.
- [ ] Bỏ ký hiệu `≈` cho các đại lượng đếm được chính xác.

### [ ] C3. Table 1 — tỷ lệ phác đồ ICI cộng lại **103.7%**, không có chú thích

78.8 + 20.0 + 4.9 = 103.7%. Dữ liệu thật: PD-1 mono 197 (79.8%), PD-L1 mono 50 (20.2%),
combo 12 (4.9%) — combo là **tập con** của hai nhóm trên, nên tổng > 100% là đúng về mặt dữ liệu
nhưng **phải chú thích**. Ngoài ra footnote `b` ("one patient may have multiple treatment lines")
được định nghĩa trong caption nhưng **không gắn vào ô nào** trong bảng.

- [ ] Thêm chú thích "combination therapy overlaps with the monotherapy categories".
- [ ] Gắn hoặc xoá footnote `b`. Xoá "IQR" khỏi danh sách viết tắt (bảng không dùng IQR).

### [ ] C4. Supplementary Table S1 **mâu thuẫn trực tiếp** với Table 1 — và S1 sai

| Modality | Table 1 (main) | Supp Table S1 | Đếm thật từ parquet |
|---|---|---|---|
| CT radiomics | 187 (75.7%) | **247 (100%)** ✗ | **187** ✓ (Table 1 đúng) |
| Pathology IHC | 105 (42.5%) | **71 (28.7%)** ✗ | **105** ✓ (Table 1 đúng) |
| PD-L1 TPS | 201 (81.4%) | 201 (81.4%) ✓ | 201 ✓ |
| Genomics / Labs | 247 (100%) | 247 (100%) ✓ | 247 ✓ |

Con số 71 trong S1 chính là **n của cohort path_valid** — lỗi copy nhầm. Các số PL 198 / LN 161
trong S1 cũng chưa được kiểm chứng.

- [ ] Sửa toàn bộ Supplementary Table S1 theo số thật; kiểm lại riêng PL/LN.

### [ ] C5. Supplementary Note S1 — template prompt có **15 biến**, main text nói **13 biến**

Template S1 chứa thêm `[SEX]`, `[BMI]`, `[TIME-SINCE-DX]` — không có trong danh sách 13 biến ở
`methods.tex:126-133`, và ví dụ trong main text (`methods.tex:171-181`) **cũng không có** 3 biến này.
Ngược lại, danh sách 13 biến ở main text lại **không có giới tính**, trong khi ví dụ S1 ghi "67-year-old
**male**". Người đọc không thể biết mô hình thực sự nhận vào gì.

- [ ] Đối chiếu với `clinical_nlp_embedding.py::df_to_text_prompts` và thống nhất **một** danh sách
      biến duy nhất giữa: main text §2.2, ví dụ §2.3, Supplementary S1, Figure 2.

### [ ] C6. Supplementary Table S2 còn **`[TO VERIFY]`** trong 4 ô

Placeholder chưa điền trong file nộp kèm → tự động bị pre-check trả về.

- [ ] Điền số feature radiomics và IHC-G sau lọc (trung bình qua fold).

### [ ] C7. Table 8 (KM) mâu thuẫn với §3.4 và Figure 5(D) về mô hình đạt χ² = 28.92

- Table 8: `DyAM + NLP-PCA16`, IHC-G, χ² = 28.92 ★
- `results.tex` §3.4: *"the highest χ² … for the IHC-G arm with **NLP raw**"*
- Figure 5(D) caption: *"IHC-G, + **NLP raw** (χ² = 28.92★)"*

Nguồn gốc (`paper/drafts/master_numbers.md` Table A5, `result/training-results.md` §2.0b) ghi rõ:
**NLP-PCA16**. → **Table 8 đúng, text và figure sai**.

- [ ] Sửa `results.tex` §3.4 và caption Figure 5(D) thành NLP-PCA16 (và sửa cả tên file/label subfig
      `fig4d_km_ihcg_nlpraw.pdf` nếu hình vẽ đúng là NLP-PCA16 — nếu hình vẽ đúng là NLP raw thì phải
      vẽ lại).

### [ ] C8. Table 7 (survival) — caption định nghĩa **tdAUC** nhưng bảng **không có cột tdAUC**

Caption: *"tdAUC: time-dependent AUC at 12 months"*. Các cột thực tế: C-index, CI, Cox HR, CI, IBS.

- [ ] Hoặc thêm cột tdAUC 6/12/18m (số đã có sẵn ở `master_numbers.md` Table A4), hoặc bỏ định nghĩa
      khỏi caption. Khuyến nghị **thêm cột** — vì tdAUC là luận cứ chính của §3.4 và Discussion.

### [ ] C9. §3.4 nói "all eight models" nhưng Table 8 chỉ có **4 dòng**

- [ ] Bổ sung 4 dòng còn thiếu vào Table 8, hoặc sửa câu thành "for the four representative models shown".

### [ ] C10. Abstract — hai phát biểu sai về số

1. *"placed OvO within **0.011** AUC of cooperative attention at every primary benchmark
   (AUC = 0.773 vs. 0.760…)"* → 0.773 − 0.760 = **0.013**, tự mâu thuẫn ngay trong một câu.
   (0.011 là Δ của **paired bootstrap** ở BM4, không phải hiệu của hai giá trị trung bình đang trích.)
2. *"NLP-encoded clinical features improved AUC **over raw numeric encoding** … (best AUC = 0.813,
   **Δ = +0.030**)"* → +0.030 là so với **no-clinical baseline** (0.784), không phải so với numeric
   labs (0.788 → Δ = **+0.025**). Trích sai mốc so sánh ngay ở câu bán hàng chính.
3. *"p ≥ 0.17"* ghép ngay sau cặp 0.773 vs 0.760, nhưng p của cặp đó là **0.313**; 0.172 là p ở BM4.

- [ ] Sửa cả 3, và thống nhất một định nghĩa Δ duy nhất xuyên suốt bài (khuyến nghị:
      **Δ so với +Labs**, vì đó mới là so sánh đúng câu hỏi nghiên cứu "NLP thay numeric").

### [ ] C11. §3.2 — Δ = +0.009 vs hiệu hai giá trị trung bình = +0.013, không giải thích

*Vị trí:* `results.tex:62-66` và caption Table 4. Hai đại lượng khác nhau (mean-of-seed-means vs
paired bootstrap trên điểm bệnh nhân) bị đặt cạnh nhau như thể cùng một thứ.

- [ ] Thêm một câu định nghĩa rõ hai đại lượng, hoặc chỉ báo cáo một loại.

### [ ] C12. §3.2 — "range −0.001 to **+0.018**" cho nhóm k ≥ 5

Từ Table 4, Δ lớn nhất trong nhóm k ≥ 5 là cấu hình #19: 0.766 − 0.749 = **+0.017**.

- [ ] Sửa thành +0.017 (hoặc kiểm lại từ số chưa làm tròn và ghi nhất quán).

### [ ] C13. Supplementary tham chiếu **sai số hiệu bảng** của main text

Đánh số thực tế trong `main.aux`: Table 1 = patients, **2 = modalities**, 3 = ovo screen,
4 = 21 combos, **5 = AUC IHC-A**, **6 = AUC IHC-G**, 7 = survival, 8 = KM.

| Supplementary ghi | Đúng phải là |
|---|---|
| "Tables 2–3 of the main text" (Note S3) | Tables **5–6** |
| "Table 3 of the main text" (Fig. S3 caption) | Table **6** |
| "Table 5 of the main text" (Table S3 caption) | Table **4** |

- [ ] Sửa cả 3 (và rà lại toàn bộ tham chiếu chéo còn lại trong supplementary).

### [ ] C14. Các lệch nhỏ khác giữa các phần

- [ ] Fig. S3 caption ghi ΔAUC = **0.026**; Table 6 cho 0.813 − 0.788 = **0.025**.
- [ ] Fig. S3 dùng **5.000** bootstrap resample, Table 4 dùng **2.000** → thống nhất hoặc giải thích.
- [ ] Fig. S4 nói "**20** tested modality configurations" trong khi phần chính đã chuyển sang **21**.
- [ ] Table 7 (survival) dựa trên `summary_df` của một run có AUC nhị phân 0.776 (IHC-A no-clin),
      còn Table 5 báo cáo 0.764 cho cùng cấu hình → **hai run khác nhau**. Phải nói rõ, nếu không
      người đọc sẽ thấy hai con số "cùng mô hình" mà lệch nhau.

---

## 5. MINOR — chỉnh sửa cho chỉn chu

### Trích dẫn và thuật ngữ

- [ ] **N1.** "One-vs-Others (OvO)": trong tài liệu ML, **OvO = one-vs-one**; cơ chế của bài là
      **one-vs-rest (OvR)**. Và `\cite{Hastie1998}` ("Classification by Pairwise Coupling") đúng là
      bài về OvO **thật sự** → citation không đỡ cho khái niệm đang dùng. Đề nghị: đổi tên thành
      *one-vs-rest competitive attention* (giữ nhãn OvO trong ngoặc cho khớp code), và thay/ bổ sung
      citation phù hợp (Rifkin & Klautau 2004, "In Defense of One-Vs-All Classification").
- [ ] **N2.** tdAUC cite `\cite{Harrell1996}` — sai; Harrell 1996 là C-index. Ước lượng
      cumulative/dynamic cần **Heagerty & Zheng 2005** hoặc **Uno et al. 2007**.
- [ ] **N3.** "AUC > 0.80 … commonly used in oncology decision support `\cite{Vanguri2022}`" —
      Vanguri 2022 không đặt ra ngưỡng này. Bỏ citation hoặc thay bằng nguồn đúng (Hosmer–Lemeshow).
- [ ] **N4.** `\cite{Vaswani2017}` ở §2.4 mang tính trang trí ("in the spirit of attention-based
      architectures broadly") — kiến trúc DyAM không liên quan self-attention của Transformer.
      Bỏ hoặc viết rõ điểm khác biệt.
- [ ] **N5.** Nhiều mục `.bib` dùng `and others` → với style MDPI cần liệt kê đủ tác giả (hoặc theo
      đúng quy định *et al.* của tạp chí). Kiểm lại số trang/volume của Wang2020MiniLM (NeurIPS
      không đánh số trang như vậy) và Vaswani2017 (thiếu pages).
- [ ] **N6.** Bib header còn dòng "PLACEHOLDER ENTRIES — replace with real data after Step F1" →
      xoá comment nội bộ khỏi file nộp.

### Diễn giải thống kê

- [ ] **N7.** IBS: *"values below 0.25 indicate better-than-random calibration"* — sai khái niệm.
      0.25 là Brier của dự báo hằng 0.5, không phải chuẩn hiệu chỉnh. Chuẩn đúng là so với
      **mô hình Kaplan–Meier null (reference)**; và Brier score đo cả discrimination lẫn calibration.
      → Sửa Methods dòng 335-336 và caption Fig. 6(D).
- [ ] **N8.** Supplementary Note S2: *"bootstrap p = 0.927, i.e., not significantly different from
      chance ordering"* — diễn giải sai p-value. p = 0.927 chỉ nghĩa là không có bằng chứng bác bỏ
      H₀; không nói gì về "chance ordering".
- [ ] **N9.** Discussion dòng 38-42 và Note S3 đưa ra cỡ mẫu cần ~1.200–1.500 nhưng công thức trong
      S3 viết dở dang (`Δ/SE_unit` không được định nghĩa). Viết lại chặt chẽ hoặc cite Obuchowski &
      McClish 1997 cho sample size của AUC.
- [ ] **N10.** Nên báo cáo **AUC trung bình ± sd theo fold** bên cạnh pooled out-of-fold AUC —
      pooled OOF với z-score theo từng fold là ước lượng không chuẩn, reviewer thống kê sẽ hỏi.
- [ ] **N11.** `0.616 ± 0.001` (TMB, 5-seed) — sd = 0.001 nhỏ bất thường; nên nêu lý do (TMB là
      1 biến, mô hình gần như tất định) để tránh bị nghi là lỗi.
- [ ] **N12.** Thiếu **decision curve analysis** và **calibration plot** cho endpoint nhị phân —
      *Cancers* rất hay yêu cầu bằng chứng clinical utility, không chỉ AUC. Cân nhắc thêm DCA.

### Trình bày / văn phong

- [ ] **N13.** `results.tex:57` — *"patient labels re-shuffled per seed"* đọc như **xáo trộn nhãn**
      (tức thí nghiệm null!). Thực tế là hoán vị **thứ tự hàng** để đổi cách chia fold
      (vì `KFold` hardcode `random_state=0`). Viết lại: *"the patient row order was permuted per
      seed to induce different fold assignments; the label–patient correspondence was preserved."*
      **Nếu không sửa, reviewer sẽ hiểu là bài shuffle nhãn.**
- [ ] **N14.** Nên nói thẳng trong Methods lý do phải hoán vị hàng (hàm `train()` không nhận seed
      hiệu quả) — minh bạch hơn là để người đọc tự phát hiện khi đọc code.
- [ ] **N15.** Discussion dòng 111-115: "PD-L1 TPS was scored using the **Sauter method**" — không
      có trong Methods và không phải tên một phương pháp chuẩn (Sauter là đồng tác giả của Vanguri
      2022). Kiểm lại và viết đúng assay (22C3/SP142/E1L3N…).
- [ ] **N16.** Conclusion (264 từ) lặp gần như nguyên văn Discussion. Rút còn ~120–150 từ,
      chỉ giữ 3 câu: đóng góp, phát hiện chính, việc cần làm tiếp.
- [ ] **N17.** Tiêu đề §3.2 quá dài (16 từ). Rút gọn: *"Competitive Attention Performs on Par with
      Cooperative Attention"*.
- [ ] **N18.** Hình vẽ hardcode số liệu trong `generate_fig*_demo.py` thay vì đọc từ
      `experiments/results/*.json` → rủi ro hình và bảng lệch nhau khi cập nhật. Nên đọc từ JSON.
- [ ] **N19.** Tên file hình `fig3a_*`, `fig4*`, `fig5*` lệch với số hiệu hình thực tế trong bài
      (Figure 4, 5, 6) → dễ nhầm khi nộp bản hình riêng. Đổi tên hoặc lập bảng mapping.
- [ ] **N20.** `main.tex` còn 2 file `manuscript_word.docx` / `_v2.docx` và các file build
      (`.aux/.fls/.fdb_latexmk`) trong repo — dọn trước khi đóng gói nộp.

---

## 6. Câu hỏi reviewer sẽ hỏi trong vòng phản biện (chuẩn bị sẵn câu trả lời)

1. **"Attention học được có hơn trung bình cộng không?"** → Hiện câu trả lời trung thực là *không*
   (B1). Chuẩn bị đoạn Discussion cho việc này.
2. **"Vì sao có 2 cohort external mà không báo cáo?"** → B3.
3. **"Gain của NLP có phải chỉ vì tăng chiều?"** → M10, cần control experiment.
4. **"Bài dùng single-partition hay repeated-seed để kết luận?"** → M8, phải nhất quán.
5. **"AUC 0.813 có tái lập được không?"** → Không (5-seed = 0.783). Phải nói trước, đừng để bị hỏi.
6. **"So với logistic regression đơn giản trên feature ghép thì sao?"** → Log §2.2 có sẵn
   `lr_concat` 0.7276 / `lr_late` 0.7615 ở BM1, và `lr_late` **thắng** ở BM4 (0.7515, p = 0.045 so với
   Original). Nên đưa vào bài một dòng baseline LR — nếu không, reviewer sẽ coi là né tránh.
   (Lưu ý: `master_numbers.md` có ghi chú nội bộ "không trích dẫn LR" — cần xem lại quyết định này,
   vì thiếu baseline tuyến tính là điểm trừ nặng ở tạp chí lâm sàng.)

---

## 7. Thứ tự đề xuất khi sửa

1. **Số liệu trước** (C1–C14) — rẻ, nhanh, và sửa xong thì các phần chữ mới ổn định.
2. **B4, B5, B6, M12, M13** — sửa Methods cho khớp code (không cần chạy lại gì).
3. **B1, B2, B3** — thêm bảng/mục mới từ kết quả **đã có sẵn** trong `result/training-results.md`
   (không cần train lại). Đây là 3 việc quan trọng nhất và đều là "viết ra cái đã có".
4. **M7, M9** — chạy lại phần test thống kê (paired DeLong + sign test 21 tổ hợp). Ít tốn kém,
   khả năng biến kết quả "không significant" thành có ý nghĩa.
5. **M10** — chạy control experiment cho NLP (2 trong 4 phương án). Đây là phần tốn thời gian nhất
   nhưng cũng là phần làm bài báo thực sự vững.
6. **M11, M14** — Ethics, Simple Summary, TRIPOD, flow diagram, điền placeholder.
7. **N1–N20** — dọn dẹp cuối cùng.

---

## 8. Đánh giá tổng thể

| Tiêu chí | Điểm (1–5) | Ghi chú |
|---|---|---|
| Tính mới | 3 | NLP-encoding biến bảng là mới và hữu ích; OvO gần như không mang lại gì |
| Chất lượng phương pháp | 2 | CV/feature-selection làm đúng, nhưng test thống kê sai và Methods lệch code |
| Chất lượng báo cáo | 2 | Thiếu baseline chủ chốt + external validation đã có sẵn; nhiều mâu thuẫn số |
| Ý nghĩa lâm sàng | 2 | Chưa external validation, chưa DCA, cỡ mẫu nhỏ |
| Trình bày | 3 | Viết mạch lạc, hình đẹp; nhưng còn placeholder và lỗi cross-reference |

**Nếu B1–B6 được xử lý trung thực** (báo cáo `uniform_avg`, tách hiệu ứng backbone, đưa external
validation vào, sửa Methods cho khớp code), bài sẽ chuyển từ "một claim cải tiến kiến trúc yếu"
sang **"một nghiên cứu ablation trung thực với một phát hiện dương rõ ràng về biểu diễn dữ liệu
lâm sàng"** — hướng đi này vừa đúng với dữ liệu đang có, vừa dễ được chấp nhận hơn nhiều.

---

*Nguồn đối chiếu: `paper/{main,sections/*,tables/*}.tex`, `paper/supplementary/supplementary.tex`,*
*`paper/drafts/master_numbers.md`, `result/training-results.md` §0/§2.0–§2.9, `lung_helpers.py`,*
*`../datasets/*` (tính lại trực tiếp Table 1 và độ phủ modality).*
