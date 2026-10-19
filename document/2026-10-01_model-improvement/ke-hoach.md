# Kế hoạch: bốn hướng cải thiện mô hình

> 2026-10-01 · Chạy nối tiếp bằng `experiments/modelimp/run_all.py`
> Mốc phải vượt: `uniform_avg` BM1 **0,7730** (1 seed) · **0,7850** (ensemble 5 seed) ·
> hiện tại tốt nhất dự án **0,7896** (uniform_avg + NLP + miss_pattern, §2.14)

## Bối cảnh

Bảy cơ chế hợp nhất và ba bộ học dạng bảng đều nằm trong dải 0,75–0,77. Ba phương pháp độc lập
(TabPFN, cooperative learning, XGBoost) cùng sụp ở tổ hợp có radiomics → **nút thắt nằm ở bộ mã hoá
radiomics, không phải tầng hợp nhất.** Bốn hướng dưới đây đều tác động ở tầng mô hình/huấn luyện,
không phải chế thêm đặc trưng.

---

## Hướng 1 — ComBat: hài hoà hoá radiomics theo máy chụp

**Vì sao:** đặc trưng radiomics phụ thuộc mạnh vào máy chụp và tham số tái tạo. Dự án chưa từng khử
hiệu ứng này. Đây có thể là lý do radiomics **không khái quát ra ngoài mẫu** (AUC `rad_valid`
0,44–0,48 ≈ ngẫu nhiên) dù lại là thứ quyết định hiệu năng nội bộ.

**Dữ liệu có sẵn:** `scanner_lightspeed` (59 bệnh nhân) và `scanner_revolution` (70), 129 dòng
thiếu → batch = {lightspeed, revolution, unknown}.

**Cách làm:** ComBat (`neurocombat`) trên ma trận đặc trưng radiomics trước khi dựng modality.
ComBat là hiệu chỉnh **không giám sát** (không dùng nhãn) nên không rò rỉ nhãn; vẫn ghi rõ là fit
trên toàn cohort, đúng thông lệ ngành.

**Đo:** BM1, BM3 (5 seed) + ngoại kiểm radiomics trên `rad_valid`. Thành công quan trọng nhất không
phải AUC nội bộ mà là ngoại kiểm nhích khỏi mức ngẫu nhiên.

---

## Hướng 2 — Ensemble theo seed, đưa vào định nghĩa mô hình

**Vì sao:** đã đo sẵn nhưng chưa dùng: BM1 một seed 0,7746 → ensemble 5 seed **0,7850** (+0,010),
lớn hơn phần lớn những gì đã thử.

**Cách làm:** trung bình điểm từng bệnh nhân qua 5 seed rồi mới tính AUC. Với `uniform_avg` thuần thì
tính trực tiếp từ điểm đã lưu (`experiments/results/scores/uni_*.csv`) — **không cần train lại**.
Với biến thể NLP+miss thì chạy lại 1 tổ hợp để lưu điểm.

**Đo:** cả 21 tổ hợp cho `uniform_avg`; BM1 cho NLP+miss.

---

## Hướng 3 — Loss tối ưu trực tiếp AUC + modality dropout

**Vì sao:** mô hình huấn luyện bằng BCE nhưng được chấm bằng AUC — hai thứ không trùng nhau.
Loss theo cặp biến 247 bệnh nhân thành 62 × 185 = **11.470 cặp**, thực chất là tăng cường dữ liệu.
Modality dropout: cohort có 23 mẫu khuyết khác nhau nhưng mô hình chưa bao giờ được huấn luyện có
chủ đích cho việc thiếu nhánh.

**Cách làm:** sao `model_uniform_avg.py` sang `experiments/modelimp/model_plus.py`, thêm hai tuỳ chọn:
- `loss='pairwise'`: `mean log(1 + exp(-(s⁺ − s⁻)))` trên các cặp trong batch
- `modality_dropout=p`: mỗi epoch, che ngẫu nhiên mask với xác suất p (giữ ít nhất 1 nhánh)

**Đo:** BM1 + BM4, lưới nhỏ: {BCE, pairwise} × {dropout 0, 0.2}, 5 seed.

---

## Hướng 4 — Dò siêu tham số

**Vì sao:** `epochs=125, lr=0.01, alpha=0.001, beta=0.0` cố định từ lâu, chưa từng dò hệ thống.

**Cách làm:** dò theo từng trục quanh mốc hiện tại (không dò lưới đầy để tiết kiệm):
`epochs ∈ {75, 125, 200}` · `lr ∈ {0.005, 0.01, 0.02}` · `alpha ∈ {0.0001, 0.001, 0.01}`.
9 cấu hình, 3 seed, trên BM1.

---

## Thứ tự chạy và thời gian ước tính

| | Hướng | Thời gian |
|---|---|---|
| 1 | Ensemble seed (rẻ nhất, từ điểm đã lưu) | ~5 phút |
| 2 | Dò siêu tham số (BM1, 3 seed) | ~25 phút |
| 3 | Ranking loss + modality dropout (BM1, BM4) | ~40 phút |
| 4 | ComBat radiomics (BM1, BM3 + ngoại kiểm) | ~40 phút |

Mỗi hướng bọc try/except, hỏng một hướng vẫn chạy tiếp, tổng kết ở cuối.
Kết quả thô: `experiments/results/modelimp.json`. Tổng hợp: `ket-qua.md` cùng thư mục này.

## Tiêu chí

Hướng nào cho ≥ +0,01 so với mốc tương ứng ở ít nhất 2 tổ hợp thì chạy tiếp đủ 21 tổ hợp
(Full evaluation rule). Không đạt thì ghi nhận âm và dừng.
