# Vòng lặp tìm kiến trúc — GIAO THỨC (cố định trước khi chạy)

> Lập 2026-10-04. Mỗi vòng lặp PHẢI đọc file này và `ledger.md` trước khi làm gì.
> Mục tiêu của người dùng: một kiến trúc mới cho AUC cao hơn rõ rệt ở các tổ hợp NHIỀU NGUỒN
> (luận điểm bài báo: kết hợp nhiều nguồn thì tốt hơn). Không cần thắng ở mọi tổ hợp.

## 1. Phạm vi đánh giá — KHÔNG đổi giữa chừng

Sáu tổ hợp k ≥ 5 nguồn (so sánh thuần kiến trúc: không NLP, không miss_pattern):

    Rad+IHC-A+Gen · Rad+IHC-G+Gen · Rad+IHC-A+MutAmp · Rad+IHC-A+Gen+PDL1
    Rad+IHC-G+Gen+PDL1 (BM1) · Rad+IHC-A+Gen+TMB+PDL1

Harness: MODEL_PARAMS chuẩn, 10-fold `KFold(shuffle=True, random_state=0)` trên chỉ số hoán vị theo seed.
20 seed = `B.SEEDS + [11,22,33,44,55,66,77,88,99,101,202,303,404,505,606]`.

**Mốc so sánh đã có sẵn, không phải chạy lại:** điểm từng bệnh nhân của `uniform_avg`, `ovo`, `dyam`
trên đúng 6 tổ hợp × 20 seed ở `experiments/results/ovo_proof/{model}__{slug}.csv`.

| Mốc (TB 6 tổ hợp) | 1 mô hình | ensemble 20 |
|---|---|---|
| DyAM | 0.7428 | 0.7633 |
| OvO | 0.7492 | 0.7655 |
| **uniform_avg (mốc phải vượt)** | **0.7507** | **0.7671** |

## 2. Quy trình cho mỗi ứng viên

**Bước A — Sàng lọc (rẻ, ~10 phút).** 3 tổ hợp (BM1, Rad+IHC-A+Gen+TMB+PDL1, Rad+IHC-G+Gen), 5 seed chuẩn.
So ghép cặp với điểm `uniform_avg` cùng seed. **Qua sàng lọc nếu Δ AUC một mô hình trung bình ≥ +0.005.**
Không qua → ghi ledger, XOÁ code + kết quả thô của ứng viên, sang ứng viên kế.

**Bước B — Xác nhận (đủ 6 tổ hợp × 20 seed, ~60 phút, chạy nền nhiều lượt vì giới hạn 30 phút/lượt).**
Lưu điểm từng bệnh nhân, script phải TIẾP TỤC được chỗ dở. Rồi chạy đúng 5 phép kiểm của
`experiments/ovo_proof/analyze.py` so với `uniform_avg`.

**Tiêu chí CHẤP NHẬN (tất cả phải đạt):**

1. Δ ensemble gộp 6 tổ hợp so với `uniform_avg` ≥ **+0.010**
2. Bootstrap gộp trên bệnh nhân: CI 95% không chứa 0 (p < 0.05)
3. Thắng ≥ 5/6 tổ hợp ở mức ensemble
4. Không tổ hợp nào kém `uniform_avg` có ý nghĩa (p DeLong < 0.05)

"Hơn hẳn" theo nghĩa người dùng muốn = đạt cả 4 điều trên với Δ gộp ≥ +0.020.
Đạt 1–3 nhưng p ≥ 0.05 → ghi là "xu hướng dương, chưa xác nhận", GIỮ code, KHÔNG tuyên bố thắng.

## 3. Hàng rào chống tự lừa mình

- **Đa so sánh:** ledger đếm mọi ứng viên đã thử. Khi báo cáo một ứng viên thắng, phải nêu kèm số
  ứng viên đã thử và ngưỡng Bonferroni (0.05 / số ứng viên vào Bước B).
- **Không dò tham số trên tập đánh giá:** mỗi ứng viên chạy với siêu tham số định trước (tối đa 2 giá
  trị cho một tham số, cả hai đều phải báo cáo). Bài học §2.16: +0.008 trên 3 seed thành −0.004 trên 5.
- **Kiểm chứng mã:** mọi ứng viên phải có chế độ "tắt" tái lập ĐÚNG baseline (seed 42 BM1 = 0.7927 với
  `uniform_avg`) trước khi tin số của nó.
- **Không đụng cache dùng chung:** không gọi `load_all(use_cache=False)` với dữ liệu đã biến đổi (§2.16).
- **Không sửa bài báo.** Mọi thứ chỉ ghi vào ledger + `result/training-results.md`.

## 4. Dọn dẹp (theo yêu cầu người dùng)

Ứng viên thất bại: xoá file code và kết quả thô của nó. **Luôn giữ lại**: một dòng trong `ledger.md`
và một mục ngắn trong `result/training-results.md` (quy tắc Result logging của CLAUDE.md — kết quả âm
phải được ghi để không chạy lại mù quáng).

## 5. Ngân sách và điều kiện dừng

- Mỗi đợt tối đa **10 ứng viên**. Hết 10 → tổng kết cho người dùng, hỏi có chạy đợt tiếp không.
- Dừng sớm nếu một ứng viên đạt tiêu chí chấp nhận → báo ngay.
- Dừng nếu hết ý tưởng khác bản chất với những gì đã thử (không lặp lại biến thể của ý cũ).

## 6. Hàng chờ ý tưởng (đợt 1) — xếp theo kỳ vọng

Quy luật từ 26 lần thử trước: thêm THAM SỐ ở tầng hợp nhất → thua; thêm THÔNG TIN phía đầu vào → thắng
(NLP, mẫu khuyết); giảm PHƯƠNG SAI → thắng (ensemble). Radiomics là nút thắt và đặc thù theo vị trí.

| # | Ý tưởng | Vì sao khác những gì đã thử |
|---|---|---|
| C1 | **Hợp nhất phân cấp cân bằng theo miền**: gộp 3 vị trí radiomics thành MỘT phiếu "radiology" rồi mới trung bình với pathology/genomics/PD-L1 | Hiện radiomics có tới 3 phiếu trong phép trung bình dù là nguồn nhiễu nhất và không khái quát; đổi cấu trúc cây hợp nhất, không thêm tham số |
| C2 | **Hợp nhất trên điểm chuẩn hoá**: chuẩn hoá z (hoặc hạng) từng chuyên gia trên fold train trước khi trung bình | Chưa từng chạy; các đầu tanh có thang khác nhau nên đầu nào phương sai lớn sẽ lấn át |
| C3 | **Hợp nhất bền vững**: trung vị / trung bình cắt đuôi thay cho trung bình | Không tham số; chống lại một chuyên gia nhiễu |
| C4 | **Chọn đặc trưng radiomics ổn định** (stability selection: bootstrap elastic-net, giữ đặc trưng được chọn ≥ 60%) | Nhắm đúng bước tạo ra sức mạnh của radiomics; ensemble cho thấy phương sai ở đó là lớn nhất |
| C5 | **Chuyên gia hoá tuần tự (boosting theo modality)**: nguồn mạnh nhất khớp trước, nguồn sau khớp phần dư | Thiên kiến quy nạp khác hẳn huấn luyện chung |
| C6 | **Mã hoá hợp nhất bằng văn bản**: đưa cả genomics + PD-L1 vào câu mô tả cùng lâm sàng rồi nhúng | Cùng họ với NLP — họ duy nhất từng thắng rõ |
| C7 | **Mixup trong không gian điểm chuyên gia** | Regularization không thêm tham số, chưa thử |
| C8 | **Chuyên gia phi tuyến nhỏ** (1 lớp ẩn 4 nút + dropout mạnh) | Mọi đầu hiện đều tuyến tính; chưa đo riêng |
| C9 | **Nội suy điểm chuyên gia cho modality thiếu** từ các nguồn còn lại | Xử lý khuyết ở tầng điểm thay vì bỏ qua |
| C10 | **Tổng thay cho trung bình** (cổng tắt) kết hợp chuẩn hoá theo số nguồn | Thay đổi cách số nguồn có sẵn ảnh hưởng độ tự tin |

Thứ tự có thể đổi nếu một kết quả gợi ý hướng mới, nhưng phải ghi lý do vào ledger.

## 7. Hàng chờ ý tưởng — ĐỢT 2 (2026-10-04): theo tài liệu "balanced multimodal learning"

Người dùng yêu cầu tiếp tục hướng KIẾN TRÚC, bám theo các bài báo. Dòng chưa từng thử trong dự án: mất cân
bằng modality khi huấn luyện chung (BalanceBenchmark 2025: OGM-GE và AGM đứng đầu; các phương pháp đơn giản
— giám sát phụ đơn-modality, huấn luyện luân phiên, điều biến gradient — cạnh tranh tốt, nhất là ở tập nhỏ).

**Chẩn đoán trước khi chạy** (BM1, 3 seed): AUC của từng chuyên gia TRONG mô hình chung so với khi huấn luyện
RIÊNG: rad PC 0.631/0.639 · PL 0.553/0.556 · LN 0.647/0.671 · **pathology 0.680/0.617 (+0.062)** ·
genomics 0.666/0.674 · PD-L1 0.714/0.715. → KHÔNG có mất cân bằng đáng kể; huấn luyện chung còn làm chuyên
gia pathology TỐT LÊN (điều chuẩn). Kỳ vọng cho nhóm này vì vậy thấp, nhưng vẫn sàng lọc cho có số liệu.

Ngân sách đợt 2: 10 ứng viên. Tiêu chí, phạm vi, harness: y như mục 1–5.

| # | Ý tưởng | Nguồn |
|---|---|---|
| D1 | Giám sát phụ từng chuyên gia: L = BCE(TB) + 0.5·TB_i BCE(r_i) | "extra supervision", arXiv 2304.00379; UMT |
| D2 | OGM-GE: giảm gradient chuyên gia lấn át, α = 0.3 | Peng et al., CVPR 2022 |
| D3 | Chuyên gia huấn luyện đơn-modality độc lập rồi hợp nhất | MLA, CVPR 2024 |
| D4 | Dropout 0.2 trên đặc trưng đầu vào của từng chuyên gia | regularization chuẩn, chưa thử |
| D5 | Boosting luân phiên theo modality (khớp phần dư) | ReconBoost, ICML 2024 |
| D6 | Nhiệm vụ phụ dự đoán chéo modality (từ pathology/radiomics dự đoán PD-L1, TMB) | "clinical prediction", arXiv 2304.00379 |
| D7 | Tự chưng cất từ thầy bagging trong fold | self-distillation |
