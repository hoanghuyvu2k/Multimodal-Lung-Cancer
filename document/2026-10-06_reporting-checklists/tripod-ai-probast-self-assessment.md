# Tự đánh giá TRIPOD+AI và PROBAST+AI cho bản thảo

> Lập 2026-10-06. File này để DUYỆT — bài báo (`paper/`) KHÔNG bị sửa.
> **Nguồn danh mục:** các mục 1–27 của TRIPOD+AI lấy từ bảng 2 của bài gốc (BMJ 2024, bản mở trên Europe PMC PMC11019967), tôi đã đọc trực tiếp văn bản.
> **PROBAST+AI:** tôi KHÔNG đọc được toàn văn danh sách câu hỏi tín hiệu (chỉ biết cấu trúc: hai phần phát triển / đánh giá, mỗi phần bốn miền: người tham gia và nguồn dữ liệu, biến dự đoán, kết cục, phân tích). Mục 2 vì vậy chỉ đánh giá theo MIỀN, không theo từng câu hỏi; cần điền mẫu chính thức.
> **Căn cứ trạng thái:** văn bản trong `paper/main.tex`, `paper/sections/*.tex` (đối chiếu bằng tìm kiếm từ khoá và đọc đoạn liên quan), và các kết quả trong `result/training-results.md`. Trạng thái "Chưa rõ" nghĩa là tôi chưa tìm thấy trong bản thảo, không có nghĩa là tác giả chưa làm.

## 1. TRIPOD+AI (27 mục chính)

| Mục | Nội dung chính | Trạng thái | Ghi chú / việc cần làm |
|---|---|---|---|
| 1 Tiêu đề | Nêu đây là phát triển / đánh giá mô hình dự đoán, quần thể, kết cục | Cần kiểm | Đối chiếu tiêu đề hiện tại với yêu cầu này |
| 2 Tóm tắt | Theo TRIPOD+AI for Abstracts | Cần kiểm | Tóm tắt phải nêu cỡ mẫu, kết cục, kiểm chứng và giới hạn |
| 3a Bối cảnh y khoa | Lý do, tham chiếu các mô hình hiện có | Có | Nên thêm so sánh với Chowell 2022 (mô hình tabular lớn, 1.479 bệnh nhân) |
| 3b Quần thể, mục đích, người dùng | Vị trí trong đường chăm sóc | **Một phần** | Chưa rõ mô hình sẽ dùng khi nào (trước điều trị?), ai dùng |
| 3c Bất bình đẳng sức khoẻ | | **Chưa rõ** | Không tìm thấy; ít nhất nêu rõ không phân tích theo nhóm |
| 4 Mục tiêu | Phát triển hay kiểm chứng | Có | |
| 5a Nguồn dữ liệu | Nguồn, lý do, tính đại diện | Một phần | Cohort MSK MIND; thiếu bàn luận tính đại diện |
| 5b Thời gian thu nhận | Ngày bắt đầu/kết thúc | Có (2017–2021) | Thiếu ngày kết thúc theo dõi |
| 6a Bối cảnh, số trung tâm | | Có | Đơn trung tâm — nêu rõ |
| 6b Tiêu chí chọn | | Có | IIIB/IV, ICI dòng 1–2, có CT nền |
| 6c Điều trị đã nhận | Cách xử lý | **Một phần** | Biến đơn trị/phối hợp có trong 13 biến; chưa mô tả việc xử lý |
| 7 Tiền xử lý, kiểm tra chất lượng | | Một phần | Có bộ lọc radiomics; chưa mô tả kiểm tra chất lượng GLCM, vùng u |
| 8a Định nghĩa kết cục | Khoảng thời gian, cách đánh giá, lý do chọn | Có | BOR → nhị phân. **Nên thêm:** 29/48 bệnh nhân SD có lợi ích bền vững bị xếp "không đáp ứng" (§2.35) |
| 8b Người đánh giá kết cục | | **Chưa rõ** | |
| 8c Làm mù đánh giá kết cục | | **Chưa rõ** | |
| 9a Chọn biến dự đoán ban đầu | | Một phần | Cần nêu rõ danh sách biến lâm sàng đã thử và lý do dùng 13 biến |
| 9b Định nghĩa biến, đo khi nào | | Một phần | Điểm PD-L1: bài viết "lấy từ báo cáo lâm sàng" nhưng kênh dùng điểm chấm lại (201/247) — nêu rõ nguồn và việc lấp bằng điểm lâm sàng (C18) |
| 9c Người đánh giá biến | | **Chưa rõ** | Người chấm lại PD-L1 / khoanh vùng u |
| 10 Cỡ mẫu | Giải thích, tính toán | Một phần | Có phân tích công suất (Note S3); **các số trong Note S3 sai** (SK-3), phải sửa trước khi nộp |
| 11 Dữ liệu khuyết | | Có | Mặt nạ nhị phân. **Nên thêm:** vector mẫu khuyết là tín hiệu quy trình chăm sóc (§2.14, §2.39), ~2/3 giá trị của pathology là "có tiêu bản" |
| 12a Cách dùng dữ liệu, chia | | Có | 10-fold lặp; nêu rõ KHÔNG có tập kiểm tra giữ ngoài cho mô hình hợp nhất |
| 12b Xử lý biến | | Có | RobustScaler, `no_scale` cho embedding |
| 12c Loại mô hình, mọi bước, chỉnh siêu tham số, kiểm chứng nội bộ | | **Một phần — quan trọng** | **Phải khai báo ~52 ứng viên đã thử trên cùng 247 bệnh nhân** và việc chọn thành phần bằng chính CV đó (§2.40 H1) |
| 12d Dị biệt giữa cụm | | Không áp dụng / nêu | Đơn trung tâm |
| 12e Thước đo đánh giá | Phân biệt, hiệu chuẩn, lợi ích lâm sàng | **Một phần** | Có AUC, IBS. **Không có đường cong hiệu chuẩn và phân tích lợi ích lâm sàng**; điểm là z-score theo fold, không phải xác suất → nêu rõ mô hình chỉ xếp hạng |
| 12f Cập nhật mô hình | | Không áp dụng | |
| 12g Cách tính dự đoán khi đánh giá | | Có | Mã nguồn |
| 13 Mất cân bằng lớp | | Có | `pos_weight` (tỉ lệ 1 : 2,98) |
| 14 Công bằng | | **Chưa rõ** | Chưa phân tích theo giới, tuổi, mô học; nêu là hạn chế |
| 15 Đầu ra, ngưỡng | | Một phần | Chưa nói rõ không có ngưỡng lâm sàng; ngưỡng `find_optimal_cutoff` chỉ dùng để chuẩn hoá |
| 16 Khác biệt phát triển và đánh giá | | Một phần | Cohort ngoài đơn nguồn, thiếu 6/13 biến lâm sàng, thiếu genomics (§2.36) — cần nêu |
| 17 Phê duyệt đạo đức | | **CHƯA ĐIỀN** | `main.tex` và `methods.tex` còn `[INSTITUTION]`, `[IRB-NUMBER]`. Tuyên bố "All patients provided written informed consent" cần tác giả XÁC NHẬN (nghiên cứu hồi cứu thường dùng miễn đồng ý) — tôi không kiểm chứng được |
| 18a Tài trợ | | **CHƯA ĐIỀN** | `[FUNDING — ...]` |
| 18b Xung đột lợi ích | | Có | |
| 18c Đề cương | | **Chưa rõ** | Nêu "không có đề cương đăng ký trước" |
| 18d Đăng ký | | **Chưa rõ** | Nêu "không đăng ký" |
| 18e Chia sẻ dữ liệu | | Có (theo yêu cầu) | |
| 18f Chia sẻ mã | | **CHƯA ĐIỀN** | `https://github.com/[REPO]`; nhớ kho mã có nhiều tệp thực nghiệm đã xoá — chốt phiên bản |
| 19 Bệnh nhân và công chúng tham gia | | **Chưa rõ** | Nêu "không có" |
| 20a Dòng bệnh nhân | | Một phần | Có Table 1; chưa có sơ đồ dòng, số bệnh nhân có/không có từng nguồn |
| 20b Đặc điểm, lượng dữ liệu khuyết | | Có | Table 1 sinh bằng `make_table1.py` |
| 20c So phân bố phát triển và đánh giá | | **Chưa rõ** | Cohort ngoài lệch phân bố GLCM (§2.26) |
| 21 Số bệnh nhân và sự kiện ở từng phân tích | | Một phần | Ghi rõ n = 247 (62 / 185), 52/71 (pathology) và 46/50 (radiomics) có đặc trưng ở cohort ngoài |
| 22 Đặc tả mô hình đầy đủ | | Có (qua mã) | |
| 23a Hiệu năng kèm CI, nhóm con | | **Một phần — cần bổ sung** | Dùng CI cấp bệnh nhân của §2.37; chưa có nhóm con |
| 23b Dị biệt giữa cụm | | Không áp dụng | |
| 24 Cập nhật mô hình | | Không áp dụng | |
| 25 Diễn giải | | Có | Cần chỉnh theo kết quả đã kiểm chứng (OvO: ổn định, không chính xác hơn) |
| 26 Hạn chế | | **Một phần — cần mở rộng** | Xem mục 3 |
| 27a Xử lý dữ liệu đầu vào thiếu khi triển khai | | **Chưa rõ** | Vector mẫu khuyết khó dùng khi triển khai (§2.40 H2) |
| 27b Người dùng, chuyên môn | | **Chưa rõ** | |
| 27c Hướng nghiên cứu tiếp | | Có | Nên nêu cohort đa phương thức độc lập |

**Tóm tắt:** mục hiện không đạt mà tôi chắc chắn: 17, 18a, 18f (còn chỗ giữ chỗ), 12c (thiếu khai báo số lần thử), 12e (thiếu hiệu chuẩn), 10 (số liệu công suất sai). Các mục "Chưa rõ" phần lớn chỉ cần thêm một câu "không áp dụng / không thực hiện".

## 2. PROBAST+AI — đánh giá theo miền (KHÔNG theo câu hỏi tín hiệu)

| Miền | Nguy cơ thiên lệch (tự đánh giá) | Lý do |
|---|---|---|
| Người tham gia và nguồn dữ liệu | Trung bình | Hồi cứu, đơn trung tâm; cohort ngoài chọn theo từng nguồn, không đại diện cho quần thể dùng mô hình hợp nhất |
| Biến dự đoán | **Cao với vector mẫu khuyết và biến "có tiêu bản"** | Đo thông tin về quy trình chăm sóc, có thể không khả dụng/ổn định khi triển khai; điểm PD-L1 chấm lại ở 201/247, phần còn lại lấp bằng điểm lâm sàng |
| Kết cục | Trung bình | BOR nhị phân có nhiễu: 29/48 bệnh nhân SD có lợi ích bền vững bị xếp không đáp ứng; chưa rõ làm mù người đánh giá |
| Phân tích | **Cao** | n = 247 (62 sự kiện đáp ứng), khoảng 52 ứng viên thử trên cùng dữ liệu, thành phần chọn bằng chính CV; không có tập kiểm tra giữ ngoài cho mô hình hợp nhất; chưa hiệu chuẩn; kiểm chứng ngoài chỉ từng nguồn |
| Khả năng áp dụng | Trung bình đến cao | Khác biệt giữa phát triển và đánh giá ở mục 16 |

Ý nghĩa: bài nên tự nhận là **nghiên cứu phát triển có nguy cơ thiên lệch cao ở miền phân tích** thay vì một mô hình sẵn sàng triển khai; điều này khớp hướng viết "đánh giá lại chặt chẽ" đã đề xuất.

## 3. Các câu nên thêm vào Limitations (đã có số liệu)

1. Mô hình hợp nhất chưa được kiểm chứng ngoài; chỉ PD-L1 (AUC 0,739) và pathology (0,755) khái quát ở cohort ngoài (§2.36).
2. Khoảng tin cậy cấp bệnh nhân của AUC ensemble rộng ≈ 0,12 và chồng lên nhau giữa mô hình chốt và DyAM (§2.37).
3. Lợi thế không chuyển sang PFS (C-index 0,62 cho mọi mô hình) và hầu như không sang điểm cuối DCB (§2.37).
4. ~52 ứng viên thử trên cùng 247 bệnh nhân; mức tăng của mô hình chốt so với trọng số đều (+0,020) nằm trong dải kỳ vọng của việc chọn (+0,011 đến +0,024) (§2.40).
5. Khoảng 2/3 giá trị của pathology là tín hiệu "có tiêu bản" (quy trình chăm sóc) (§2.39).
6. Genomics không xác nhận được ngoài (C-index sống còn 0,54) (§2.37).
