# PDL1 Score – Phân tích dữ liệu

**File:** `../datasets/pdl1_score.parquet`  
**Cột dữ liệu:** `Sauter PD-L1 Score`  
**Index:** `main_index` (dạng `P-XXXXXXX`)

---

## 0. Ý nghĩa y tế của PD-L1

### PD-L1 là gì?

**PD-L1** (Programmed Death-Ligand 1, hay CD274) là một protein bề mặt tế bào thuộc họ B7, được mã hóa bởi gene *CD274*. Trong điều kiện sinh lý bình thường, PD-L1 đóng vai trò kiểm soát miễn dịch, ngăn hệ miễn dịch tấn công nhầm vào mô lành.

Khi tế bào ung thư **biểu hiện PD-L1 cao**, chúng "giả vờ" là tế bào bình thường bằng cách gắn kết PD-L1 với thụ thể PD-1 trên tế bào T. Liên kết này **ức chế hoạt động của tế bào T**, giúp khối u thoát khỏi sự tiêu diệt của hệ miễn dịch (*immune evasion*).

```
Tế bào T ──(PD-1)───×───(PD-L1)── Tế bào ung thư
                  bị ức chế
                  → T cell "mệt mỏi", không tiêu diệt được khối u
```

### Vai trò trong điều trị NSCLC (ung thư phổi không tế bào nhỏ)

PD-L1 là **biomarker dự đoán đáp ứng với liệu pháp miễn dịch** (immune checkpoint inhibitors – ICI), đặc biệt là các thuốc nhóm anti-PD-1/PD-L1:

| Thuốc | Cơ chế | Hãng |
|---|---|---|
| Pembrolizumab (Keytruda) | Anti-PD-1 | Merck |
| Atezolizumab (Tecentriq) | Anti-PD-L1 | Roche |
| Durvalumab (Imfinzi) | Anti-PD-L1 | AstraZeneca |
| Nivolumab (Opdivo) | Anti-PD-1 | BMS |

Các thuốc này **chặn liên kết PD-1/PD-L1**, giải phóng tế bào T và phục hồi khả năng tiêu diệt khối u.

### Ý nghĩa của điểm số PD-L1 (TPS – Tumor Proportion Score)

Điểm **Sauter PD-L1 Score** trong dataset này là **TPS (Tumor Proportion Score)**: tỷ lệ phần trăm tế bào ung thư có nhuộm màu dương tính với PD-L1 trên tiêu bản giải phẫu bệnh (IHC – immunohistochemistry).

| Ngưỡng TPS | Phân loại | Ý nghĩa lâm sàng |
|---|---|---|
| < 1% | Am tinh | Ít có lợi từ ICI đơn trị; cân nhắc hóa trị |
| 1 – 49% | Biểu hiện thap | Có thể dùng ICI phối hợp hóa trị |
| >= 50% | Biểu hiện cao | **Ưu tiên dùng ICI đơn trị** (pembrolizumab first-line theo KEYNOTE-024) |

> **Lưu ý quan trọng:** PD-L1 dương tính **không đảm bảo** bệnh nhân sẽ đáp ứng ICI. Ngược lại, một số bệnh nhân PD-L1 âm tính vẫn có đáp ứng. Đây là lý do nghiên cứu này kết hợp PD-L1 với các modality khác (radiomics, genomics) để tăng độ chính xác dự đoán.

### Phương pháp nhuộm Sauter (22C3 pharmDx)

Điểm trong dataset được chấm theo phương pháp **Sauter** dùng kháng thể **22C3** (cùng kháng thể dùng để phê duyệt pembrolizumab của FDA). Bác sĩ giải phẫu bệnh quan sát tiêu bản và ước lượng tỷ lệ tế bào dương tính → cho ra điểm số thủ công (bán định lượng), thường là các giá trị tròn (0, 5, 10, 20, ..., 95%).

### Liên hệ với bài toán trong nghiên cứu này

- **Label = 0** (PR/CR): bệnh nhân có đáp ứng điều trị → kỳ vọng PD-L1 cao hơn (ICI hiệu quả)
- **Label = 1** (SD/POD): bệnh nhân không đáp ứng → PD-L1 có thể cao hoặc thấp (nhiều yếu tố khác can thiệp)

PD-L1 là một **tín hiệu đơn lẻ không đủ mạnh** để dự đoán đáp ứng. Mô hình multimodal (kết hợp radiomics + genomics + PD-L1 + lâm sàng) trong dự án này nhằm khắc phục hạn chế này.

---

## 1. Tổng quan

| Thuộc tính | Giá trị |
|---|---|
| Số bệnh nhân | 201 |
| Số cột | 1 |
| Giá trị null | 0 (không có missing) |
| Cohort | Discovery only (không có validation) |

---

## 2. Thống kê mô tả

| Thống kê | Giá trị |
|---|---|
| Mean | 28.59% |
| Median | 5.00% |
| Std | 36.40% |
| Min / Max | 0% / 95% |
| Q25 / Q75 | 0% / 60% |

Median thấp hơn nhiều so với mean → phân phối lệch phải (right-skewed), đặc trưng của dữ liệu PD-L1 trong NSCLC.

---

## 3. Phân phối giá trị

Điểm là số nguyên rời rạc (đánh giá thủ công của bác sĩ giải phẫu bệnh), chỉ xuất hiện 13 giá trị phân biệt:

| Điểm (%) | Số BN |
|---|---|
| 0 | 91 |
| 1 | 6 |
| 5 | 14 |
| 10 | 11 |
| 20 | 3 |
| 30 | 6 |
| 50 | 11 |
| 60 | 9 |
| 70 | 6 |
| 75 | 1 |
| 80 | 20 |
| 90 | 10 |
| 95 | 13 |

---

## 4. Phân nhóm lâm sàng

Theo ngưỡng chuẩn trong lâm sàng (NSCLC / immunotherapy):

| Nhóm | Ngưỡng | N | % |
|---|---|---|---|
| Am tinh | < 1% | 91 | 45.3% |
| Thap | 1 – 49% | 40 | 19.9% |
| Cao | >= 50% | 70 | 34.8% |

Phân nhóm chi tiết hơn:

| Nhom | N |
|---|---|
| 0% | 91 |
| 1–24% | 34 |
| 25–49% | 6 |
| 50–74% | 26 |
| 75–100% | 44 |

---

## 5. Nhận xét

- **Phân phối bimodal**: cụm đầu tại 0% (âm tính, ~45%), cụm thứ hai tại 75–95% (>20%). Đây là đặc điểm điển hình của PD-L1 expression trong NSCLC.
- **Chỉ trong discovery cohort**: toàn bộ 201 bệnh nhân đều thuộc discovery set → dùng làm feature huấn luyện, không dùng cho validation.
- **Điểm rời rạc**: là kết quả đánh giá bán định lượng (semiquantitative) của bác sĩ, không phải đo liên tục.

---

## 6. Cách tích hợp vào pipeline

Dùng `prepare_other_modalities()` để thêm vào `modality_dict`:

```python
df_pdl1 = pd.read_parquet(f"{BASE_DB_DIR}/pdl1_score.parquet")
prepare_other_modalities(modality_dict, df_pdl1, modality_MASK, name='pdl1_score')
```

**Lưu ý:** Không cần `no_scale` – điểm PD-L1 là giá trị số thô, RobustScaler có thể áp dụng bình thường. Nếu muốn binarize:

```python
df_pdl1['pdl1_high'] = (df_pdl1['Sauter PD-L1 Score'] >= 50).astype(int)
```
