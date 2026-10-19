# Các quyết định cần chốt trước khi viết lại bài báo

> Lập 2026-10-07. **Bản thảo trong `paper/` chưa bị sửa.** Bạn đánh dấu `[x]` vào ô chọn, tôi mới sửa.
> Mỗi mục có: (1) nó là gì, (2) nó thay đổi gì trong bài, (3) bằng chứng, (4) các lựa chọn và hệ quả, (5) đề xuất của tôi (đánh dấu ⭐).
> Muốn nhanh: ghi "theo đề xuất" cho cả nhóm, tôi sẽ lấy các ô ⭐.
> Mọi con số có tham chiếu mục trong `result/training-results.md`. Chi tiết bản nháp văn bản tiếng Anh: `document/2026-10-04_arch-search-loop/draft-architecture-ablation.md`.

## Bức tranh chung (đọc trước)

Sau khoảng 59 ứng viên thử trên cùng 247 bệnh nhân, kết luận trung thực nhất của nghiên cứu là:

1. **Mô hình đầy đủ (trọng số đều + mã hoá lâm sàng bằng câu + PD-L1 lâm sàng) hơn DyAM gốc ở mọi tổ hợp nhiều nguồn** và ở mọi phân hoạch (+0,035 mỗi seed), nhưng **ở mức bệnh nhân chưa có ý nghĩa thống kê** (p = 0,16) vì cỡ mẫu nhỏ.
2. **Cơ chế attention học được không hơn trung bình đều.** Lợi thế của OvO là ổn định, không phải chính xác.
3. **AUC còn tăng gần tuyến tính theo cỡ mẫu** trong khoảng đã quan sát (89 → 222 bệnh nhân): nút thắt là dữ liệu, không phải mô hình.
4. Hai nguồn **khái quát tốt ra bệnh nhân ngoài** (PD-L1, pathology), còn radiomics thì không. Mô hình hợp nhất đầy đủ **chưa có kiểm chứng ngoài**.

Vì vậy tôi đề xuất bài viết theo hướng **"đánh giá lại chặt chẽ + mã hoá lâm sàng bằng ngôn ngữ"**, thay vì "kiến trúc mới thắng DyAM".

> **ĐÍNH CHÍNH QUAN TRỌNG (2026-10-07, mới hơn ghi chú bên dưới):** đặc trưng pathology thủ công (GLCM, IHC-A) chỉ tồn tại cho **105 bệnh nhân PD-L1 DƯƠNG** (điểm chấm lại > 0, TPS ≥ 1%): 105/105 người có đặc trưng đều dương, 0 người âm có đặc trưng
> (110 người dương, 5 người dương không có đặc trưng). Vì vậy bit "có pathology" trong vector mẫu khuyết **gần như chính là biến "PD-L1 dương tính"** — một dấu ấn SINH HỌC đã biết (tỉ lệ không đáp ứng: dương 0,60, âm 0,90), không phải quy trình chăm sóc.
> Bài gốc (Vanguri 2022) cũng ghi: 201/247 bệnh nhân có tiêu bản đạt kiểm soát chất lượng (≥ 100 tế bào u), trong đó 105 (52%) dương tính (TPS ≥ 1%), và mô hình che (mask) nguồn pathology ở khối u PD-L1 âm tính.
> Điều này **giải thích** kết quả ở Q2: vector chỉ giúp ở tổ hợp KHÔNG có PD-L1 (nó mang sẵn thông tin dương/âm tính), còn khi chuyên gia PD-L1 có mặt thì vector thừa. Các câu dưới đây gọi bit này là "quy trình chăm sóc" cần đọc theo nghĩa mới này; phần số nguồn còn lại (CT khả phân đoạn, điểm PD-L1 có hay không) vẫn là "mẫu có mặt, cơ chế chưa xác định".
> Ghi chú đính chính thuật ngữ trước đó (về biến "có tiêu bản" 163 người) vẫn đúng với biến đó.

> **ĐÍNH CHÍNH THUẬT NGỮ (2026-10-07):** trong file này (và trong các giải thích trước đó của tôi) cụm "quy trình chăm sóc" chỉ là GIẢ THUYẾT, chưa được kiểm chứng. Đã kiểm lại trên 247 bệnh nhân:
> có tiêu bản thì tỉ lệ không đáp ứng 0,70, không có tiêu bản 0,85 (p = 0,013), nhưng 13 biến lâm sàng KHÔNG dự đoán được "có tiêu bản" (AUC 0,48) và các chỉ số mức độ nặng (ECOG, albumin, tổng khối u, dòng điều trị, di căn) không khác có ý nghĩa giữa hai nhóm.
> Cách gọi đúng hơn: **"mẫu có mặt của các nguồn dữ liệu — cơ chế chưa xác định"** (có thể là quy trình chăm sóc không đo được, cách lắp ráp dữ liệu của nghiên cứu, yếu tố thời kỳ, hoặc một phần sinh học). Khi đọc "quy trình chăm sóc" bên dưới, hãy hiểu theo nghĩa này.
> Các đề xuất ⭐ không đổi (rủi ro không khái quát và không kiểm chứng ngoài được vẫn còn nguyên).

---

> **QUYẾT ĐỊNH CỦA NGƯỜI DÙNG (2026-10-10), Q1 = phương án B:** bài tập trung vào OvO; trọng số đều KHÔNG đưa vào nội dung bài. Hệ quả cần nhớ khi viết: (i) so sánh chính chỉ còn OvO với DyAM, phát biểu được là "OvO ổn định hơn và nhỉnh hơn DyAM ở tổ hợp nhiều nguồn",
> không viết được "OvO tốt nhất trong mọi cách gộp"; (ii) các số đã tính trên nền trọng số đều (bảng ablation 20 seed, C18, Phikon, bỏ-từng-nguồn, 21 tổ hợp mô hình đầy đủ) cần được tính lại trên nền OvO nếu muốn đưa vào — thang cộng dần trên nền OvO hiện chỉ có ở 5 seed (§2.23: DyAM 0,7599 → OvO 0,7704 → +NLP 0,7822 → +miss 0,7874 → ensemble 0,7933 ở BM1);
> (iii) người phản biện có thể hỏi về mốc trung bình đơn giản; số liệu vẫn nằm trong sổ (§2.24) nếu cần trả lời.

> **QUYẾT ĐỊNH CỦA NGƯỜI DÙNG (2026-10-10) về thống kê:** bỏ trọng tâm "có ý nghĩa thống kê" khỏi bài vì cỡ mẫu nhỏ. Thực hiện: trình bày hướng và độ lớn của khác biệt kèm khoảng tin cậy dạng ước lượng, không dùng p-value làm tiêu chí kết luận, không dùng chữ "có ý nghĩa" cho OvO vs DyAM; vẫn giữ MỘT câu trong Limitations về cỡ mẫu nhỏ và khoảng tin cậy rộng.

> **ĐÃ ÁP DỤNG VÀO BÀI (2026-10-12):** Q1 B, Q2 A (bài không dùng miss_pattern nên không có thay đổi), Q3 A (bỏ khỏi bài vì nền trọng số đều), Q4 A (số tiêu đề thay bằng OvO+NLP 0,780 / ensemble 0,791; số giữ-ngoài lấy nhánh DyAM; số kiểm chứng ngoài 0,689 bỏ vì nền trọng số đều), Q5 A (phần lớn mục phụ lục không thêm vì nền trọng số đều), SK-1, SK-2 B, SK-3/4/5 mục 1–5, H1 (xoá experiments/gbdt). SK-6 CHƯA làm. Chi tiết trong báo cáo cuối phiên.

# NHÓM 1 — Hướng của bài (4 quyết định, quan trọng nhất)

## Q1. Backbone nào làm mô hình chính?

**Là gì:** quy tắc gộp điểm của các nguồn. Có ba loại đã đo: DyAM (attention cộng tác, bài gốc), OvO (attention cạnh tranh, đóng góp của bài hiện tại) và trọng số đều (trung bình điểm các nguồn có mặt).

**Thay đổi trong bài:** tên mô hình trong tiêu đề, tóm tắt, Hình 1, Hình attention, toàn bộ Bảng 3–4 và phần Discussion. Đây là quyết định lớn nhất về cách kể câu chuyện.

**Bằng chứng (20 seed, 6 tổ hợp từ 5 nguồn trở lên, §2.24):**

- OvO − DyAM: +0,0064 mỗi seed (OvO thắng 14/20, p = 0,008), nhưng **ở mức bệnh nhân chỉ +0,0022, p = 0,57**.
- OvO − trọng số đều: −0,0015 mỗi seed (OvO chỉ thắng 4/20 seed).
- Trọng số đều − DyAM: +0,0079 mỗi seed (16/20).
- Độ ổn định giữa các seed: OvO 0,0157, DyAM 0,0177.
- Hơn 40 biến thể attention và kiến trúc khác đều không vượt trọng số đều.

**Lựa chọn:**

- [ ] **A. Trọng số đều làm mốc, OvO và DyAM là các nhánh so sánh.** *(đề xuất ban đầu của tôi; người dùng đã KHÔNG chọn)*  Bài nói: cơ chế hợp nhất không phải thứ quyết định, OvO ổn định hơn DyAM. *Hệ quả:* câu chuyện trung thực nhất, nhưng mất vai trò "đề xuất OvO" ở bài hiện tại; phải viết lại tóm tắt và Discussion.
- [X] ⭐ **B. Giữ OvO làm mô hình chính — NGƯỜI DÙNG ĐÃ CHỌN (2026-10-10): bỏ trọng số đều khỏi các nội dung đưa vào bài, tập trung OvO.** *Hệ quả:* vẫn dùng được số liệu §2.23 (OvO + NLP + mẫu khuyết), nhưng phải nêu rõ trọng số đều ngang OvO và OvO chỉ hơn về ổn định. **Không viết được** "OvO cho AUC cao hơn".
- [ ] **C. Giữ nguyên cách kể hiện tại** (OvO chính xác hơn DyAM). *Hệ quả:* **không có số liệu ủng hộ** ở mức bệnh nhân; rủi ro bị phản biện. Tôi không khuyến nghị.

---

## Q2. Vector mẫu khuyết (`miss_pattern`) giữ hay bỏ?

**Là gì:** một nguồn nhị phân cho mô hình biết bệnh nhân **có** nguồn dữ liệu nào (radiomics, pathology, genomics, PD-L1) và tổng số nguồn. Trong đó bit "có pathology" thực chất là **PD-L1 dương tính** (dấu ấn sinh học đã biết); các bit còn lại (CT khả phân đoạn, có điểm PD-L1 chấm lại) mang thông tin về việc mẫu/ảnh có đủ điều kiện hay không, cơ chế chưa xác định.

**Thay đổi trong bài:** định nghĩa mô hình trong Methods, bảng ablation, Limitations (phải giải thích đây là tín hiệu quy trình chăm sóc, khó khái quát sang cơ sở khác).

**Bằng chứng:**

- BM1 (có PD-L1): bỏ vector chỉ mất 0,001 (p = 0,47) (§2.39, §2.41).
- Khi 50% bệnh nhân bị ẩn pathology ở tập kiểm tra: mô hình **có** vector mất thêm 0,006 so với mô hình **không** có (p = 0,019) (§2.40) — bền kém hơn.
- **Nhưng** ở hai tổ hợp 5 nguồn **không có PD-L1** (Rad+IHC-A+Gen, Rad+IHC-G+Gen), bỏ vector mất **0,017 và 0,018** (thắng 20/20 và 19/20 seed) (§2.41). Quy tắc định trước cho kết quả "giữ".
- Giả thuyết (chưa kiểm chứng): khi PD-L1 có ở gần mọi bệnh nhân, thông tin sẵn-có đã được mang sẵn qua phép trung bình nên vector thừa.

**Lựa chọn:**

- [X] ⭐ **A. Mô hình chính BM1 không có vector; ở tổ hợp thiếu PD-L1 báo cả hai phiên bản kèm lưu ý.** *Hệ quả:* mô hình chính đơn giản, bền; vẫn trung thực về chỗ vector giúp. Hơi phức tạp để trình bày.
- [ ] **B. Giữ vector ở mọi nơi.** *Hệ quả:* AUC cao nhất mọi tổ hợp, nhưng cả bài phụ thuộc vào tín hiệu quy trình chăm sóc, dễ bị phản biện.
- [ ] **C. Bỏ vector ở mọi nơi.** *Hệ quả:* sạch và bền nhất, nhưng mất 0,017–0,018 ở tổ hợp thiếu PD-L1 (nơi nó là thứ duy nhất mang tín hiệu sẵn-có).

---

## Q3. Biến "có tiêu bản" có đưa vào không?

**Là gì:** biến cho mô hình biết bệnh nhân có tiêu bản pathology được quét hay không (163 người, nhiều hơn 105 người có đặc trưng pathology thủ công). Thêm một chuyên gia pathology từ mô hình nền (Phikon) nâng BM1 lên 0,809 (ensemble 0,818) — nhưng đối chứng cho thấy **khoảng ba phần tư mức tăng tái lập được bằng một chuyên gia không có ảnh nào**, chỉ biết "có tiêu bản" (BM1 0,8087). Phần do nội dung ảnh chỉ +0,004 (§2.33, §2.34).

**Thay đổi trong bài:** có thêm một dòng ablation hay không; con số tiêu đề có là 0,796 hay 0,809.

**Lựa chọn:**

- [X] ⭐ **A. Chỉ là phân tích bổ sung có khai báo** ("mở rộng vector sẵn-có thêm biến có-tiêu-bản"), không đưa vào mô hình chính. *Hệ quả:* con số tiêu đề không phụ thuộc tín hiệu khó bảo vệ nhất.
- [ ] **B. Không nhắc đến.** *Hệ quả:* bài gọn hơn nhưng bỏ mất phát hiện hữu ích (đặc trưng ảnh từ mô hình nền không mang thêm thông tin ngoài PD-L1).
- [ ] **C. Đưa vào mô hình chính và nêu số 0,809.** *Hệ quả:* con số đẹp hơn nhưng phải giải thích đây là tín hiệu quy trình chăm sóc, không phải đặc trưng học sâu từ ảnh.

---

## Q4. Con số tiêu đề của bài là gì?

**Là gì:** con số hiệu năng nêu ở tóm tắt và Discussion. Có nhiều cách đo, cho kết quả khác nhau:

| Cách đo (BM1)                                                   | AUC                       |
| ----------------------------------------------------------------- | ------------------------- |
| DyAM gốc, trung bình 20 phân hoạch                            | 0,760                     |
| Mô hình chốt, trung bình 20 phân hoạch                      | **0,796 ± 0,011**  |
| Mô hình chốt, ensemble 20 seed                                 | 0,804 [0,740; 0,861]      |
| Giữ-ngoài theo thứ tự số chụp CT (proxy)                    | khoảng 0,70 [0,58; 0,82] |
| Kiểm chứng ngoài PD-L1 + genomics (133 bệnh nhân, nhãn DCB) | 0,689 [0,594; 0,784]      |

**Cần khai báo kèm:**

- Khoảng tin cậy cấp bệnh nhân của mô hình chốt và DyAM **chồng nhau gần hoàn toàn** (§2.37).
- Mức tăng so với trọng số đều (+0,020) nằm trong dải thổi phồng kỳ vọng do chọn trong ~59 ứng viên (0,011–0,024) (§2.40).
- Bài gốc công bố 0,80; mô hình chốt ngang mức đó, nên điểm mới của bài là so sánh **cùng quy trình**, không phải con số tuyệt đối.

**Lựa chọn:**

- [X] ⭐ **A. Nêu 0,796 (một mô hình, 20 phân hoạch) làm con số chính, kèm CI bệnh nhân, ensemble 0,804, và ghi rõ hai con số bảo thủ (≈0,70 giữ-ngoài, 0,689 kiểm chứng ngoài).** *Hệ quả:* trung thực, phản biện khó bắt bẻ, nhưng con số không "đẹp".
- [ ] **B. Chỉ nêu 0,796 và ensemble.** *Hệ quả:* ngắn gọn nhưng bỏ qua cảnh báo thiên lệch; rủi ro nếu người đọc so với con số công bố 0,80.
- [ ] **C. Nêu con số bảo thủ (≈0,70) làm tiêu đề.** *Hệ quả:* an toàn nhất, nhưng thấp hơn con số công bố và ước lượng dựa trên 21 bệnh nhân đáp ứng.

---

## Q5. Phân tích nào vào bài chính, phân tích nào vào phụ lục?

**Là gì:** bạn đã có nhiều phân tích mới. Bài chính dài hạn chế; phần còn lại vào Supplementary.

| Phân tích                                                                                        | Nên ở                                   | Công dụng                                                              |
| -------------------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------ |
| So sánh quy tắc hợp nhất (DyAM, OvO, trọng số đều), Bảng A                                | Bài chính                               | Luận điểm trung tâm                                                  |
| Ablation của mô hình chốt (Bảng C) và 21 tổ hợp (Bảng E)                                  | Bài chính                               | Bằng chứng đóng góp từng bước, và "chỉ tốt khi nhiều nguồn" |
| Khoảng tin cậy cấp bệnh nhân                                                                  | Bài chính                               | Khai báo độ bất định                                               |
| Huấn luyện chung giúp từng nguồn (Bảng B)                                                    | Bài chính hoặc phụ lục               | Luận điểm "kết hợp nguồn giúp từng nguồn"                       |
| Bỏ từng nguồn một (LOSO)                                                                       | Phụ lục                                 | Giải thích đóng góp theo nguồn                                     |
| PFS và DCB, kiểm chứng ngoài PD-L1 và genomics, đường cong học, giữ-ngoài theo thứ tự | Phụ lục (một vài câu ở Limitations) | Độ bền, hạn chế                                                     |
| Hơn 50 ứng viên thất bại (kết quả âm)                                                      | Phụ lục                                 | Minh bạch, tránh người khác lặp lại                               |
| Danh mục TRIPOD+AI và PROBAST+AI                                                                 | Phụ lục                                 | Yêu cầu của nhiều tạp chí                                          |

**Lựa chọn:**

- [X] ⭐ **A. Theo bảng trên.**
- [ ] **B. Đưa nhiều hơn vào bài chính** (ghi rõ phân tích nào: ____________________).
- [ ] **C. Chỉ giữ tối thiểu**, bỏ phần lớn phân tích mới.

---

# NHÓM 2 — Các sai khác từ lần kiểm chứng số liệu (chi tiết ở `document/2026-09-29_verify-paper-numbers/sai-khac-can-quyet-dinh.md`)

## SK-1. Note S3 (phân tích công suất): số liệu sai số học

**Là gì:** Note S3 nói cần bao nhiêu bệnh nhân để phát hiện chênh lệch AUC 0,02 hoặc 0,03. Số ghi trong bài không khớp với chính công thức nó nêu.

**Thay đổi:** cỡ mẫu cần cho Δ = 0,02 từ "950–1.250" thành **1.211–1.343**; cho Δ = 0,03 từ "420–550" thành **538–597**; SE(Δ) từ 0,014–0,016 thành 0,0158–0,0166. Đổi cả câu tương ứng trong Discussion.

- [X] ⭐ **Sửa theo số tính lại.**
- [ ] Giữ nguyên số cũ.
- [ ] Cách khác: ____________

## SK-2. Bảng 4: 23/63 ô lệch tới 0,016 do thư viện scipy đổi phiên bản

**Là gì:** khi chạy lại, 23 ô (đều có radiomics) lệch ≤ 0,016 so với bài. Nguyên nhân là phiên bản scipy mới xử lý một bước chuẩn hoá khác đi; 13 tổ hợp không radiomics và toàn bộ cột OvO+NLP khớp tuyệt đối.

- [ ] ⭐ **A. Giữ nguyên số trong bài, thêm câu ghi phiên bản thư viện đã dùng vào Methods/Supplementary.** Không cần chạy lại; chuẩn mực thông thường.
- [X] **B. Thay toàn bộ bằng số mới.** Tái lập được với thư viện hiện hành nhưng phải làm lại nhiều bảng.
- [ ] **C. Dựng môi trường scipy cũ để xác nhận số cũ.** Dứt điểm nhưng tốn thời gian.

## SK-3/4/5. Bảng 5–8: không tái lập; một luận điểm sụp

**Là gì:** các bảng về NLP so với Labs thô so với không dùng lâm sàng (Bảng 5–8) đến từ một lần chia dữ liệu duy nhất, không tái lập được. Chạy lại với 5 seed:

- ✅ **"NLP tốt hơn Labs thô" đứng vững** (+0,013 và +0,020).
- ❌ **"NLP tốt hơn khi không dùng lâm sàng" sụp** (bài ghi +0,020 và +0,030; chạy lại −0,010 và +0,001).
- ⚠️ "NLP cải thiện phân tầng sống còn" không ổn định, chiều đổi giữa hai lần chạy.
- BioClinBERT kém hơn mốc không lâm sàng có ý nghĩa thống kê (p = 0,028).

**Năm mục cần làm (đánh dấu từng mục bạn đồng ý):**

- [X] ⭐ 1. Thay số Bảng 5, 6, 7, 8 bằng bộ 5 seed (trung bình ± độ lệch chuẩn).
- [X] ⭐ 2. Đổi trục so sánh của mục 3.3 từ "NLP so với không lâm sàng" sang "**NLP so với Labs thô**" (so sánh tái lập được và có ý nghĩa khoa học hơn).
- [X] ⭐ 3. Nói về sống còn theo chiều, không nhấn con số ("NLP cho HR và phân tầng KM cao hơn mốc không lâm sàng ở cả hai nhánh").
- [X] ⭐ 4. Cập nhật Note S2: BioClinBERT kém hơn có ý nghĩa (p = 0,028).
- [X] ⭐ 5. Vẽ lại Hình 3a, 3b theo số mới.

Ghi chú mới (ảnh hưởng mục 2): trong mô hình chốt, bỏ NLP chỉ mất 0,003 (p = 0,027), nên tránh viết "NLP là thành phần quan trọng nhất"; nên viết theo mức tăng khi thêm vào mô hình trọng số đều trần (+0,015).

## SK-6. Cột tdAUC và IBS của Bảng 7: không kiểm được

**Là gì:** thư viện tính hai chỉ số này cần numpy ≥ 2, môi trường hiện là numpy 1.26 nên không chạy được.

- [ ] ⭐ **Bỏ qua**, ghi trong bài rằng hai cột này không kiểm chứng lại được (hoặc bỏ hai cột).
- [X] Nâng numpy trong một môi trường riêng rồi kiểm. (Tốn thời gian; không đổi kết luận chính.)
- [ ] Cách khác: ____________

---

# NHÓM 3 — Thông tin chỉ bạn cung cấp được (đang là chỗ giữ chỗ trong bản thảo)

Điền vào chỗ trống; tôi không đoán thay. Bản thảo hiện còn các chỗ `[ ... ]` này trong `paper/main.tex` và `paper/sections/methods.tex`.

- Tên cơ quan phê duyệt đạo đức: ______________________________
- Số đề cương: ______________________________
- Nguồn tài trợ: ______________________________
- Địa chỉ kho mã (GitHub): ______________________________
- Chữ viết tắt tên tác giả trong phần đóng góp: ______________________________
- **Xác nhận câu "Tất cả bệnh nhân đồng ý bằng văn bản"** (nghiên cứu hồi cứu thường dùng miễn đồng ý; tôi không kiểm chứng được):
  - [ ] Đúng, giữ nguyên
  - [ ] Sửa thành: ______________________________ (ví dụ "hội đồng đạo đức miễn yêu cầu đồng ý do nghiên cứu hồi cứu")

---

# NHÓM 4 — Dọn dẹp và kho mã

## H1. Dọn tệp không cần thiết

**Là gì:** số tệp thay đổi trong kho mã rất nhiều; ở đầu phiên làm việc tôi đã liệt kê các nhóm tệp có thể xoá (A, B, C), bạn chưa chọn. Thư mục `experiments/gbdt/` (XGBoost, kết quả âm) cũng chưa quyết.

- [ ] ⭐ Tôi liệt kê lại danh sách hiện tại (vì kho mã đã thay đổi nhiều) rồi bạn chọn nhóm xoá.
- [ ] Giữ nguyên, không xoá gì.
- [X] Xoá `experiments/gbdt/` (giữ kết quả đã ghi trong sổ).

## H2. Đẩy mã lên GitHub

**Là gì:** khoá SSH đã tạo nhưng chưa thêm vào tài khoản GitHub, nên chưa đẩy được.

Các bước (bạn làm bước 1, tôi làm bước 2):

1. Thêm khoá công khai sau vào GitHub (Settings → SSH and GPG keys → New SSH key):
   `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIL/ydYIMg9t7Vdkls9+87fF8o/Lk4DRwa3N7drhUkfhe hoanghuyvu2k@gmail.com`
2. Tôi đổi địa chỉ remote sang `git@github.com:hoanghuyvu2k/Multimodal-Lung-Cancer.git` và đẩy.

- [ ] Đã thêm khoá, hãy đẩy
- [X] Chưa làm, để sau

**Lưu ý:** mỗi lần commit tôi sẽ hỏi lại trước khi thực hiện; cần bạn đồng ý rõ ràng.

---

## Sau khi bạn quyết định, tôi sẽ làm gì

1. Viết lại các mục trong `paper/sections/*.tex` và `paper/supplementary/` theo lựa chọn; **không sửa gì khi chưa có đánh dấu**.
2. Chạy `python paper/check_figures.py` và vẽ lại các hình bị ảnh hưởng (Hình 3a/3b nếu chọn SK-3/4/5 mục 5; hình mô hình nếu đổi backbone).
3. Biên dịch PDF, **tự động xuất lại bản Word** (`paper/manuscript_word.docx`) như quy định của dự án, và nhập các con số mới vào `build_manuscript_docx.py` cho khớp.
4. Chạy lại `python paper/make_table1.py` nếu đổi số liệu bệnh nhân.
5. Báo cáo lại danh sách thay đổi để bạn rà soát trước khi coi là xong.

**Cách trả lời nhanh:** "theo đề xuất" (lấy mọi ô ⭐), hoặc "theo đề xuất, trừ Q2 chọn C" và những thay đổi riêng.
