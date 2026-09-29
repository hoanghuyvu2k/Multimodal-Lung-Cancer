# Kịch bản thuyết trình

**Bài báo:** *NLP-Augmented Multimodal Attention Fusion for Immunotherapy Response Prediction in Non-Small Cell Lung Cancer*
(Tích hợp đa phương thức dựa trên cơ chế attention tăng cường NLP để dự đoán đáp ứng miễn dịch trong ung thư phổi không tế bào nhỏ)

**Thời lượng đề xuất:** 15–18 phút trình bày + 5 phút hỏi đáp
**Số slide gợi ý:** 16 slide

> **Cách dùng tài liệu này:** Mỗi mục gồm 3 phần:
> - 🖼️ **Slide** — nội dung hiển thị (bullet ngắn gọn, đưa lên slide)
> - 🎙️ **Lời thuyết trình** — đọc/diễn đạt theo ý này (không cần học thuộc)
> - ⏱️ thời lượng gợi ý cho slide đó
>
> Thuật ngữ chuyên môn (AUC, attention, embedding…) giữ nguyên tiếng Anh như chuẩn trình bày học thuật ở VN.

---

## Slide 1 — Tiêu đề (⏱️ 45 giây)

🖼️ **Slide**
- Tên đề tài (tiếng Việt + tiếng Anh)
- Tên bạn, đơn vị, người hướng dẫn
- "Phát triển trên nền tảng DyAM (Vanguri et al., *Nature Cancer* 2022)"

🎙️ **Lời thuyết trình**
> "Kính chào thầy cô và các bạn. Em xin trình bày đề tài: *Tích hợp đa phương thức dựa trên cơ chế attention tăng cường NLP để dự đoán đáp ứng miễn dịch trong ung thư phổi không tế bào nhỏ*. Đây là công trình em phát triển dựa trên mô hình nền DyAM được công bố trên Nature Cancer năm 2022, với hai đóng góp mới của riêng em mà em sẽ trình bày ở phần sau."

---

## Slide 2 — Đặt vấn đề lâm sàng (⏱️ 1,5 phút)

🖼️ **Slide**
- Ung thư phổi: nguyên nhân tử vong do ung thư hàng đầu; NSCLC chiếm ~85%
- Liệu pháp miễn dịch (ICI – ức chế PD-1/PD-L1) tạo đột phá điều trị
- **Nhưng chỉ 20–30% bệnh nhân đáp ứng** → phần lớn kháng thuốc
- Nhu cầu: **dự đoán trước điều trị** ai sẽ đáp ứng

🎙️ **Lời thuyết trình**
> "Ung thư phổi hiện là nguyên nhân gây tử vong do ung thư cao nhất thế giới, trong đó NSCLC chiếm khoảng 85% số ca. Sự ra đời của liệu pháp miễn dịch ICI — các thuốc ức chế trục PD-1/PD-L1 — đã thay đổi căn bản việc điều trị, mang lại đáp ứng bền vững cho một nhóm bệnh nhân.
>
> Tuy nhiên, vấn đề là chỉ khoảng 20 đến 30% bệnh nhân thực sự đáp ứng với điều trị, số còn lại bị kháng thuốc nguyên phát hoặc mắc phải. Mỗi đợt điều trị ICI rất tốn kém và có thể gây tác dụng phụ. Vì vậy, nếu dự đoán được TRƯỚC điều trị bệnh nhân nào có khả năng đáp ứng, bác sĩ có thể ưu tiên đúng người, đồng thời cân nhắc phác đồ thay thế cho những người ít có khả năng đáp ứng. Đây chính là bài toán lâm sàng mà nghiên cứu của em hướng tới."

---

## Slide 3 — Hạn chế của biomarker hiện tại & hướng đa phương thức (⏱️ 1,5 phút)

🖼️ **Slide**
- Biomarker phổ biến: **PD-L1 TPS** và **TMB** → khả năng phân biệt còn hạn chế
  - PD-L1 không đồng nhất về không gian, phụ thuộc xét nghiệm
  - TMB đơn lẻ chỉ ở mức trung bình
- → Xu hướng **tích hợp đa phương thức**: radiomics (CT) + bệnh học (mô) + gen + lâm sàng
- Thách thức: trọng số hóa các phương thức khác độ tin cậy + xử lý dữ liệu thiếu

🎙️ **Lời thuyết trình**
> "Hiện nay hai biomarker được dùng nhiều nhất là điểm PD-L1 TPS và gánh nặng đột biến khối u TMB. Nhưng cả hai đều có hạn chế đã được ghi nhận rõ: PD-L1 phân bố không đồng nhất trong khối u và kết quả phụ thuộc loại xét nghiệm; còn TMB dùng một mình thì khả năng phân biệt chỉ ở mức khiêm tốn.
>
> Điều này thúc đẩy hướng tiếp cận đa phương thức — tức là kết hợp nhiều nguồn dữ liệu cùng lúc: đặc trưng hình ảnh radiomics từ CT, đặc trưng từ ảnh mô bệnh học, các biến đổi gen, và dữ liệu lâm sàng. Lý do là mỗi nguồn dữ liệu phản ánh một khía cạnh khác nhau của sinh học khối u và thể trạng bệnh nhân, bổ sung cho nhau. Tuy nhiên cách làm này cũng đặt ra thách thức mới: làm sao gán trọng số cho các phương thức có độ tin cậy khác nhau, và làm sao xử lý khi bệnh nhân thiếu một vài phương thức mà không phải loại bỏ họ."

---

## Slide 4 — Mô hình nền DyAM & khoảng trống nghiên cứu (⏱️ 1,5 phút)

🖼️ **Slide**
- **DyAM** (Vanguri 2022): cơ chế attention học theo từng bệnh nhân, tự gán trọng số mỗi phương thức, xử lý dữ liệu thiếu bằng masking
- 2 khoảng trống chưa giải quyết:
  1. **Biểu diễn biến lâm sàng**: DyAM dùng số thô → bỏ mất quan hệ giữa các biến
  2. **Dạng cơ chế attention**: DyAM chỉ dùng attention "hợp tác" (cooperative) → giả định mọi phương thức bổ sung nhau

🎙️ **Lời thuyết trình**
> "Mô hình nền DyAM giải quyết các thách thức trên bằng một cơ chế attention được học riêng cho từng bệnh nhân: nó tự động gán trọng số cho mỗi phương thức có sẵn, và xử lý dữ liệu thiếu một cách tự nhiên thông qua masking.
>
> Tuy nhiên DyAM gốc còn hai điểm em thấy có thể cải tiến. Thứ nhất, về cách biểu diễn biến lâm sàng: DyAM đưa thẳng các con số thô vào mô hình, khiến mô hình phải tự học các mối quan hệ giữa các biến — ví dụ ý nghĩa tiên lượng khi một bệnh nhân vừa lớn tuổi, vừa albumin thấp, vừa thể trạng kém — từ một tập dữ liệu khá nhỏ. Thứ hai, về cơ chế attention: DyAM chỉ dùng attention kiểu 'hợp tác', tức ngầm giả định rằng mọi phương thức đều đóng góp thông tin bổ sung và cộng gộp. Đây chính là hai khoảng trống mà nghiên cứu của em nhắm vào."

---

## Slide 5 — Mục tiêu & đóng góp (⏱️ 1 phút)

🖼️ **Slide**
- **Đóng góp 1 — OvO competitive attention:** cơ chế attention "cạnh tranh" (one-vs-others), so sánh trên 20 tổ hợp phương thức
- **Đóng góp 2 — NLP encoding:** chuyển 13 biến lâm sàng số → câu tiếng Anh → embedding bằng sentence-transformer (MiniLM-L6-v2)
- Đánh giá: phân loại nhị phân (AUC) **và** phân tích sống còn (Cox, C-index, KM)
- Cohort: 247 bệnh nhân NSCLC điều trị ICI

🎙️ **Lời thuyết trình**
> "Từ đó, em đề xuất hai đóng góp. Đóng góp thứ nhất là một biến thể attention 'cạnh tranh' theo kiểu một-đối-phần-còn-lại, gọi là OvO, và em so sánh nó với attention hợp tác gốc trên 20 tổ hợp phương thức khác nhau. Đóng góp thứ hai là chuyển 13 biến lâm sàng dạng số thành câu ngôn ngữ tự nhiên rồi mã hóa bằng một mô hình sentence-transformer đã huấn luyện sẵn.
>
> Điểm đáng chú ý là em đánh giá trên cả hai khía cạnh: vừa phân loại nhị phân bằng AUC, vừa phân tích sống còn bằng hồi quy Cox, C-index và Kaplan-Meier. Toàn bộ thực hiện trên cohort khám phá gồm 247 bệnh nhân NSCLC giai đoạn tiến xa được điều trị ICI."

---

## Slide 6 — Tổng quan thiết kế nghiên cứu (Hình 1) (⏱️ 1,5 phút)

🖼️ **Slide**: chèn **Figure 1 (fig1_overview)**
- (A) Nguồn dữ liệu: CT radiomics, bệnh học IHC, genomics, PD-L1, lâm sàng (NLP)
- (B) Kiến trúc DyAM: risk score + attention (cooperative vs OvO)
- (C) Đánh giá: AUC-ROC, C-index, Cox, Kaplan-Meier

🎙️ **Lời thuyết trình**
> "Hình này tóm tắt toàn bộ thiết kế nghiên cứu. Bên trái — phần A — là các nguồn dữ liệu đầu vào: đặc trưng radiomics từ CT, đặc trưng từ ảnh mô bệnh học PD-L1, dữ liệu gen, điểm PD-L1, và các biến lâm sàng được mã hóa bằng NLP. Phần giữa — phần B — là mô hình DyAM: mỗi phương thức cho ra một điểm rủi ro, rồi được kết hợp qua cơ chế attention, ở đây em so sánh hai loại là hợp tác và cạnh tranh OvO. Phần bên phải — phần C — là cách đánh giá: vừa phân loại đáp ứng bằng AUC-ROC, vừa phân tích sống còn bằng C-index, hồi quy Cox và đường Kaplan-Meier.
>
> Cohort gồm 247 bệnh nhân để phát triển mô hình với kiểm định chéo 10-fold, cùng hai tập validation độc lập 50 và 71 bệnh nhân."

---

## Slide 7 — Kiến trúc DyAM & hai cơ chế attention (Hình kiến trúc) (⏱️ 2,5 phút) ⭐

> ⭐ Slide trọng tâm kỹ thuật — nên dành thời gian nhiều nhất.

🖼️ **Slide**: chèn **Figure kiến trúc attention (fig_attention_arch)**
- Cả hai: risk score `rᵢ = tanh(Wᵣ·xᵢ)`; output `ŷ = Σ rᵢ·aᵢ`
- **(A) Cooperative (AttentionMatrix):** ma trận N×N + softplus → L1-norm → các phương thức **chia sẻ ngân sách attention cố định** (tổng = 1)
- **(B) Competitive OvO (AttentionMatrixOvO):** mỗi phương thức so điểm `sᵢ` với **trung bình các phương thức còn lại** `μ₋ᵢ` qua sigmoid: `σ(sᵢ − μ₋ᵢ)` → trọng số **độc lập**, "best-of"

🎙️ **Lời thuyết trình**
> "Đây là phần kỹ thuật cốt lõi. Cả hai cơ chế đều có chung hai thành phần: mỗi phương thức i cho ra một điểm rủi ro rᵢ bằng hàm tanh, và dự đoán cuối cùng ŷ là tổng có trọng số của các điểm rủi ro, với trọng số là attention aᵢ. Điểm khác biệt duy nhất nằm ở cách tính trọng số attention aᵢ.
>
> Bên trái — cơ chế hợp tác, chính là AttentionMatrix gốc. Nó dùng một ma trận N×N các lớp linear với hàm softplus, để tính điểm chú ý giữa từng cặp phương thức, sau đó cộng theo cột và chuẩn hóa L1. Kết quả là các phương thức 'chia sẻ một ngân sách attention cố định' — tổng các trọng số luôn bằng 1. Tức là khi một phương thức được tăng trọng số thì phương thức khác phải giảm.
>
> Bên phải — đóng góp của em, cơ chế cạnh tranh OvO. Ở đây mỗi phương thức cho ra một điểm sᵢ, rồi điểm này được so sánh với TRUNG BÌNH điểm của tất cả các phương thức CÒN LẠI, ký hiệu μ trừ i, thông qua hàm sigmoid. Ý nghĩa là: nếu phương thức i 'vượt trội' so với phần còn lại thì nó nhận trọng số cao. Khác biệt mấu chốt là mỗi trọng số được tính ĐỘC LẬP, không bị ràng buộc tổng bằng 1. Cơ chế này giống như một phép chọn 'best-of' — phù hợp khi các phương thức có phần trùng lặp thông tin và ta muốn phương thức tốt nhất nổi trội lên."

💡 *Mẹo: chỉ tay vào ma trận N×N (bên trái) và nút μ₋ᵢ (bên phải) khi nói, để khán giả bám theo.*

---

## Slide 8 — Đóng góp 2: NLP encoding cho biến lâm sàng (Hình 2) (⏱️ 2 phút) ⭐

🖼️ **Slide**: chèn **Figure 2 (fig2_nlp_pipeline)**
- 13 biến lâm sàng số → `df_to_text_prompts()` → **câu tiếng Anh** mô tả bệnh nhân
- → sentence-transformer **MiniLM-L6-v2** → embedding **384 chiều** ("NLP raw")
- → tùy chọn PCA 16 chiều ("NLP-PCA16")
- ⚠️ Quan trọng: `no_scale=True` — KHÔNG dùng RobustScaler để giữ hình học cosine của embedding

🎙️ **Lời thuyết trình**
> "Đóng góp thứ hai là cách mã hóa biến lâm sàng. Thay vì đưa 13 con số thô vào mô hình, em chuyển hồ sơ lâm sàng của mỗi bệnh nhân thành một câu tiếng Anh hoàn chỉnh. Ví dụ: 'Bệnh nhân 69 tuổi, tiền sử hút thuốc 48 bao-năm, ECOG 1, albumin 4.0, dNLR 2.4… không có di căn gan.' Câu này được đưa qua một mô hình sentence-transformer tên là MiniLM-L6-v2, cho ra một vector embedding 384 chiều mà em gọi là 'NLP raw'. Em cũng thử nén xuống 16 chiều bằng PCA, gọi là 'NLP-PCA16'.
>
> Có một chi tiết kỹ thuật quan trọng: các embedding này đã được chuẩn hóa cosine, nên em phải tắt bước RobustScaler — đặt no_scale bằng True — nếu không sẽ làm hỏng cấu trúc hình học của không gian embedding.
>
> Trực giác ở đây là: một câu mô tả sẽ nắm bắt được mối quan hệ tổng thể giữa các biến, để những bệnh nhân có hồ sơ lâm sàng tương tự nhau được ánh xạ về các vector gần nhau — điều mà cách biểu diễn số thô khó làm được. Và một ưu điểm thực tế: phương pháp này không cần huấn luyện lại, mã hóa một bệnh nhân chỉ mất chưa tới 1 giây trên CPU."

---

## Slide 9 — Dữ liệu & phương pháp đánh giá (⏱️ 1 phút)

🖼️ **Slide**
- Cohort khám phá: **247 BN** NSCLC; nhãn: đáp ứng (PR/CR) vs không đáp ứng (SD/PD), mất cân bằng ~1:3
- Huấn luyện: BCE loss + cân bằng lớp tự động; Adam, 10-fold CV
- Đánh giá nhị phân: **AUC-ROC** + khoảng tin cậy DeLong 95%
- Đánh giá sống còn (PFS): **Cox HR, time-dependent AUC, C-index, IBS, Kaplan-Meier log-rank**

🎙️ **Lời thuyết trình**
> "Về dữ liệu và đánh giá: cohort gồm 247 bệnh nhân, nhãn là đáp ứng — tức đáp ứng một phần hoặc hoàn toàn — so với không đáp ứng. Dữ liệu mất cân bằng khoảng 1 đối 3, nên em dùng hàm mất mát BCE có tự động cân bằng lớp. Mô hình được kiểm định chéo 10-fold.
>
> Điểm em muốn nhấn mạnh là em đánh giá trên HAI trục. Trục thứ nhất là phân loại nhị phân, dùng AUC-ROC kèm khoảng tin cậy DeLong 95%. Trục thứ hai là phân tích sống còn theo thời gian sống không tiến triển PFS, gồm tỷ số nguy cơ Cox, AUC phụ thuộc thời gian, chỉ số C của Harrell, điểm Brier tích hợp, và phân tầng Kaplan-Meier với kiểm định log-rank. Việc đánh giá cả hai trục này về sau sẽ cho một phát hiện thú vị."

---

## Slide 10 — Kết quả 1: Đa phương thức vượt đơn phương thức (⏱️ 1 phút)

🖼️ **Slide**
- Đơn phương thức (logistic regression): AUC 0.570 (lâm sàng) · 0.640 (radiomics) · 0.650 (gen) · 0.729 (PD-L1)
- **DyAM tích hợp 4 phương thức: AUC = 0.784** [0.717–0.850]
- → Tích hợp đa phương thức là động lực chính của hiệu năng

🎙️ **Lời thuyết trình**
> "Kết quả đầu tiên khẳng định giá trị của tích hợp đa phương thức. Khi dùng từng phương thức riêng lẻ với hồi quy logistic, AUC chỉ dao động từ 0.57 cho lâm sàng đến 0.73 cho PD-L1. Nhưng khi DyAM tích hợp cả bốn phương thức lại, AUC đạt 0.784. Như vậy bản thân việc hợp nhất nhiều nguồn dữ liệu đã là động lực chính tạo nên hiệu năng dự đoán, vượt xa bất kỳ phương thức đơn lẻ nào."

---

## Slide 11 — Kết quả 2: OvO attention — lợi ích phụ thuộc ngữ cảnh (⏱️ 1,5 phút)

🖼️ **Slide** (có thể chèn Table 1 — so sánh OvO vs Original)
- 20 tổ hợp: OvO thắng 7/20, Original thắng 9/20, hòa 4
- **2 phương thức:** OvO tốt hơn ở 3/6 (PDL1+Gen **+3,27%**, Rad+Gen +2,23%)
- **≥3 phương thức:** Original tốt hơn ở 6/10 (nhất là khi có IHC-G/lâm sàng)
- **Cấu hình tốt nhất toàn bộ: OvO Rad+IHC-G+Gen+PDL1, AUC = 0,800** [0,739–0,862]

🎙️ **Lời thuyết trình**
> "Kết quả thứ hai về cơ chế OvO. Trên 20 tổ hợp phương thức, bức tranh khá cân bằng: OvO thắng 7 lần, mô hình gốc thắng 9 lần. Nhưng điều thú vị nằm ở việc phân tích theo SỐ LƯỢNG phương thức.
>
> Khi chỉ có HAI phương thức, OvO tốt hơn ở một nửa số trường hợp, với mức cải thiện lớn nhất là tổ hợp PD-L1 cộng gen, tăng 3,27% AUC. Ngược lại, khi có TỪ BA phương thức trở lên — đặc biệt khi có IHC-G hoặc biến lâm sàng — thì mô hình hợp tác gốc lại tốt hơn ở đa số trường hợp.
>
> Đáng chú ý, cấu hình tốt nhất trong toàn bộ nghiên cứu — radiomics cộng IHC-G cộng gen cộng PD-L1 dùng attention OvO — đạt AUC 0,800. Bài học rút ra: OvO phù hợp khi các phương thức trùng lặp thông tin và ta cần chọn cái tốt nhất; còn attention hợp tác phù hợp khi có nhiều phương thức thực sự bổ sung cho nhau."

---

## Slide 12 — Kết quả 3: NLP cải thiện AUC ổn định (Hình 3) (⏱️ 1,5 phút)

🖼️ **Slide**: chèn **Figure 3 (fig3a/b — AUC comparison)**
- IHC-A: **NLP-PCA16** tốt nhất (AUC 0,784; Δ = +0,020 so với không lâm sàng) vs +0,004 cho số thô
- IHC-G: **NLP raw** tốt nhất (AUC 0,813; Δ = +0,030)
- Cải thiện **nhất quán** ở cả hai nhánh, nhưng CI DeLong còn chồng lấn (chưa đạt ý nghĩa thống kê)
- BioClinBERT (domain-specific) lại **kém hơn** MiniLM tổng quát

🎙️ **Lời thuyết trình**
> "Kết quả thứ ba về NLP. Ở cả hai nhánh bệnh học, mã hóa NLP đều cải thiện AUC so với số thô. Cụ thể, nhánh IHC-A thì NLP-PCA16 tốt nhất với mức tăng 0,020, trong khi số thô chỉ tăng 0,004. Nhánh IHC-G thì NLP raw 384 chiều đạt AUC 0,813, tăng 0,030. Sự cải thiện này NHẤT QUÁN ở cả hai nhánh và cả hai biến thể NLP.
>
> Tuy nhiên em phải trung thực: các khoảng tin cậy DeLong vẫn chồng lấn nhau, nên xét về mặt thống kê thì chưa đạt ý nghĩa. Em sẽ giải thích lý do ở phần bàn luận. Một phát hiện phụ thú vị: mô hình BioClinBERT chuyên cho văn bản y khoa lại KÉM hơn MiniLM tổng quát — vì BioClinBERT được huấn luyện cho masked language modeling chứ không phải cho độ tương đồng ngữ nghĩa giữa các câu."

---

## Slide 13 — Vì sao thực hiện phân tích sống còn? (⏱️ 1,5 phút) ⭐

🖼️ **Slide** (3 khối)
- **1. Bài toán nhị phân bị đặt sai (lâm sàng):** phân loại chỉ hỏi *"có đáp ứng không?"* — nhưng **SD ở tháng 3 ≠ SD ở tháng 12**. Câu hỏi thật là *"đáp ứng được **BAO LÂU**?"* → thử nghiệm ung thư dùng **PFS/OS** làm endpoint chính, không phải tỉ lệ đáp ứng.
- **2. Giả thuyết trung tâm:** embedding đặt bệnh nhân hồ sơ giống nhau (già, albumin thấp, di căn gan…) gần nhau — nhóm **sống ngắn hơn** nhưng chưa chắc SD/PD nhiều hơn ở n=247 → NLP mã hóa thông tin **"bao lâu"** (time-to-event) tốt hơn **"có/không"** (binary) → đó là lý do **AUC nhị phân cải thiện ít**.
- **3. Lợi thế thống kê:** AUC nhị phân: mỗi BN = 1 nhãn → **247** đơn vị thông tin. C-index: so trên **mọi cặp** ≈ 209×38 ≈ **7.900 cặp** → **power cao hơn nhiều** trên cùng cohort.

🎙️ **Lời thuyết trình**
> "Trước khi xem kết quả sống còn, em xin giải thích **vì sao** lại thực hiện phân tích này — đây là một quyết định thiết kế có chủ đích, không phải làm thêm cho đủ.
>
> Thứ nhất, về mặt lâm sàng, bài toán phân loại nhị phân thực ra đặt sai câu hỏi. Nó chỉ hỏi 'bệnh nhân có đáp ứng hay không', nhưng một bệnh nhân ổn định ở tháng thứ 3 khác hoàn toàn một bệnh nhân ổn định ở tháng thứ 12. Câu hỏi mà bác sĩ thật sự cần là 'đáp ứng được bao lâu' — và đó chính là lý do mọi thử nghiệm ung thư đều dùng PFS hoặc OS làm tiêu chí chính, chứ không phải tỉ lệ đáp ứng.
>
> Thứ hai, đây là giả thuyết trung tâm của em: khi mã hóa hồ sơ lâm sàng thành câu rồi thành embedding, mô hình đặt những bệnh nhân có hồ sơ tương tự — cao tuổi, albumin thấp, di căn gan — gần nhau. Nhóm này sống ngắn hơn, nhưng trong tập 247 người chưa chắc có tỉ lệ không đáp ứng cao hơn. Nghĩa là NLP mã hóa thông tin 'bao lâu' tốt hơn thông tin 'có hay không' — và điều đó giải thích vì sao AUC nhị phân chỉ cải thiện chút ít.
>
> Thứ ba, về thống kê: với cùng 247 bệnh nhân, AUC nhị phân chỉ có 247 nhãn để học, trong khi C-index so sánh trên gần 8 nghìn cặp bệnh nhân. Tức là phân tích sống còn có power thống kê cao hơn nhiều — đúng công cụ cho cỡ mẫu hạn chế của chúng em."

> 📌 *Chi tiết động lực, khoảng trống y văn và narrative arc: xem file `document/2026-06-08_pathology-pdl1-glcm/survival-analysis-plan.md`.*

---

## Slide 14 — Kết quả 4: Phân tích sống còn — tín hiệu mạnh nhất (Hình 5) (⏱️ 3 phút) ⭐⭐

🖼️ **Slide**: chèn **Figure 5** — bên trái **forest plot Cox HR**, bên phải **time-dependent AUC**
- Risk score DyAM là yếu tố tiên lượng **độc lập** cho PFS ở **cả 8 biến thể** (Cox, p < 0,001)
- **Cox HR:** NLP-PCA16 cao nhất ở IHC-A (HR **6,07** vs 5,30); NLP raw cao nhất ở IHC-G (4,97 vs 4,74). **Số thô (+Labs) KHÔNG cải thiện HR**
- **Time-dependent AUC:** NLP lợi thế ở **12–18 tháng** (+0,024 / +0,020), gần như không ở 6 tháng
- **Kaplan-Meier:** mọi phân tầng đạt p < 0,005; χ² cao nhất = 28,92 (IHC-G + NLP raw)

> 💡 **Vì sao cần phân tích sống còn (không chỉ AUC)?** AUC nhị phân chỉ trả lời "có đáp ứng hay không" tại 1 thời điểm. Phân tích sống còn trả lời **"bệnh tiến triển KHI NÀO"** — tận dụng cả **trục thời gian** và cả những bệnh nhân **bị kiểm duyệt** (censored, chưa có event đến lúc cắt dữ liệu). Đây là góc nhìn lâm sàng quan trọng hơn cho tiên lượng.

### 📊 Cách đọc 2 biểu đồ trên slide

**① Forest plot — Cox Hazard Ratio (bên trái):**
- Mỗi **hàng** = 1 mô hình. 8 hàng = 4 cách mã hóa lâm sàng (No Clinical / +Labs / +NLP raw / +NLP-PCA16) × 2 nhánh bệnh học (IHC-A màu xanh, IHC-G màu cam).
- **Chấm tròn** = giá trị HR ước lượng; **đường ngang** = khoảng tin cậy 95%. Trục X là **HR theo thang log**.
- **Đường tham chiếu tại HR = 1** = "không ảnh hưởng". Chấm **càng nằm xa về bên phải của 1** → risk score càng phân tầng nguy cơ mạnh. Nếu **cả đường CI nằm bên phải 1** → có ý nghĩa thống kê.
- 👉 Đọc ra gì: **mọi chấm đều ở khoảng HR 4–6, toàn bộ CI nằm bên phải 1** → risk score là yếu tố tiên lượng mạnh và có ý nghĩa ở mọi mô hình. Các hàng **+NLP** có chấm **dịch sang phải hơn** so với No Clinical/+Labs → NLP nâng HR.

**② Time-dependent AUC (bên phải):**
- 2 panel: IHC-A (trái) và IHC-G (phải). Trục X = thời gian (**6 / 12 / 18 tháng**); trục Y = **AUC tại từng mốc** (khả năng phân biệt còn sống / đã tiến triển ở thời điểm đó).
- 4 đường = 4 cách mã hóa: ● No clinical (xanh), ■ +Labs (cam), ▲ +NLP raw (lá), ◆ +NLP-PCA16 (đỏ).
- 👉 Đọc ra gì (panel IHC-A rõ nhất): **tại 6 tháng** 4 đường gần như chồng nhau (~0,72) — ngắn hạn không khác biệt. **Tại 12–18 tháng**, hai đường NLP (lá & đỏ) **vọt lên trên** (~0,74), trong khi **+Labs (cam) tụt xuống thấp nhất** (~0,66 ở 18 tháng), thậm chí **dưới cả No clinical**. → Số thô **thêm nhiễu**, còn NLP **thêm tín hiệu**, và tín hiệu đó là **dài hạn**.

🎙️ **Lời thuyết trình**
> "Kết quả thứ tư, theo em là quan trọng nhất, đến từ phân tích sống còn. Khác với AUC nhị phân chỉ hỏi 'có đáp ứng hay không', phân tích sống còn hỏi 'bệnh tiến triển khi nào' và tận dụng được cả những bệnh nhân chưa có biến cố. Em đánh giá trên thời gian sống không tiến triển PFS.
>
> Xin nhìn biểu đồ bên trái — đây là forest plot của hồi quy Cox. Mỗi hàng là một mô hình, chấm tròn là tỷ số nguy cơ HR, đường ngang là khoảng tin cậy 95%, và đường thẳng đứng tại HR bằng 1 là mốc 'không ảnh hưởng'. Điểm đầu tiên cần thấy: tất cả các chấm đều nằm trong khoảng HR từ 4 đến 6, và toàn bộ khoảng tin cậy nằm bên phải số 1 — nghĩa là risk score của DyAM là yếu tố tiên lượng độc lập, mạnh và có ý nghĩa ở mọi mô hình, ngay cả sau khi đã hiệu chỉnh cho tuổi, ECOG, albumin, dNLR và di căn gan. Điểm thứ hai: các hàng có NLP, chấm dịch sang phải hơn — HR cao nhất 6,07 ở IHC-A — trong khi thêm số thô thì HR gần như không nhúc nhích.
>
> Bây giờ nhìn biểu đồ bên phải — AUC phụ thuộc thời gian, tại 6, 12 và 18 tháng. Hãy nhìn panel IHC-A: ở mốc 6 tháng bốn đường gần như trùng nhau, nhưng đến 12 và 18 tháng, hai đường NLP — màu lá và màu đỏ — tách lên trên rõ rệt, còn đường số thô màu cam lại tụt xuống thấp nhất, thậm chí thấp hơn cả khi không dùng lâm sàng. Điều này nói lên hai ý: lợi thế của NLP là dài hạn, không phải ngắn hạn; và số thô thực ra thêm nhiễu chứ không thêm thông tin.
>
> Tổng hợp lại — và đây là thông điệp cốt lõi: mặc dù AUC nhị phân của NLP cải thiện nhưng chưa đạt ý nghĩa do thiếu cỡ mẫu, thì phân tích sống còn lại cho tín hiệu mạnh và nhất quán nhất. Nó gợi ý rằng mã hóa NLP của biến lâm sàng chủ yếu mang thông tin TIÊN LƯỢNG dài hạn — bệnh tiến triển khi nào — hơn là thông tin DỰ ĐOÁN đáp ứng nhị phân ngắn hạn. Đây là một phân biệt quan trọng cho cách đánh giá các phương pháp mã hóa lâm sàng trong tương lai."

> 📌 *Công thức và cách tính chi tiết của Cox HR, td-AUC, C-index, IBS, KM/log-rank: xem **Phụ lục B***.

---

## Slide 15 — Bảng kết quả phân tích sống còn (8 mô hình) (⏱️ 2 phút) ⭐

🖼️ **Slide**: bảng Table 4 (bài báo) + cột tdAUC 12m — 8 mô hình × 4 chỉ số, theo 2 nhánh IHC-A / IHC-G; in đậm = tốt nhất mỗi cột/nhánh.

| Mô hình | C-index | Cox HR | tdAUC 12m | IBS↓ |
|---|---|---|---|---|
| *Nhánh IHC-A* | | | | |
| No clinical | 0,623 | 5,30 | 0,718 | 0,1745 |
| + Labs | 0,626 | 5,31 | 0,707 | 0,1739 |
| + NLP raw | 0,625 | 5,91 | 0,739 | 0,1723 |
| **+ NLP-PCA16** | **0,628** | **6,07** | **0,742** | **0,1716** |
| *Nhánh IHC-G* | | | | |
| No clinical | 0,626 | 4,74 | 0,691 | 0,1748 |
| + Labs | **0,632** | 4,49 | 0,686 | 0,1746 |
| **+ NLP raw** | **0,632** | **4,97** | **0,699** | **0,1736** |
| + NLP-PCA16 | 0,631 | 4,69 | 0,694 | 0,1736 |

🎙️ **Lời thuyết trình** (đi qua từng cột/metric)
> "Đây là bảng tổng hợp toàn bộ kết quả sống còn của 8 mô hình. Em xin đi qua **từng chỉ số** để thầy cô thấy mỗi cột nói lên điều gì.
>
> **Cột C-index** — khả năng xếp hạng nguy cơ. Tất cả mô hình đều quanh 0,62–0,63, đều cao hơn hẳn mức ngẫu nhiên 0,5, cho thấy mô hình nền gồm hình ảnh và gen vốn đã phân biệt tốt. Mã hóa NLP chỉ nhỉnh hơn rất nhẹ — cao nhất 0,628 ở nhánh IHC-A — chênh lệch chỉ khoảng 0,005 và chưa đạt ý nghĩa thống kê do phương sai giữa các fold lớn.
>
> **Cột Cox HR** — đây là kết quả mạnh nhất của bảng. Mọi tỷ số nguy cơ đều nằm trong khoảng 4,5 đến 6,1 với p nhỏ hơn 0,001, nghĩa là risk score là yếu tố tiên lượng độc lập ở mọi mô hình. Và điểm mấu chốt: mã hóa NLP cho HR cao nhất — 6,07 ở IHC-A và 4,97 ở IHC-G — trong khi thêm số thô, cột Labs, lại KHÔNG làm tăng HR, thậm chí còn giảm ở nhánh IHC-G, từ 4,74 xuống 4,49. Điều này chứng tỏ NLP mang thêm thông tin tiên lượng thật, còn số thô thì không.
>
> **Cột tdAUC 12 tháng** — khả năng phân biệt tại mốc 12 tháng. NLP tốt nhất ở cả hai nhánh, 0,742 và 0,699, còn Labs thấp nhất. Nhất quán với thông điệp: NLP mạnh hơn ở dài hạn, số thô thêm nhiễu.
>
> **Cột IBS** — độ hiệu chuẩn, càng thấp càng tốt. Mọi mô hình đều quanh 0,17, thấp hơn nhiều ngưỡng 0,25, tức tất cả đều hiệu chuẩn tốt; NLP-PCA16 thấp nhất với 0,1716.
>
> Tóm lại, nếu nhìn ngang cả bảng: các biến thể NLP **luôn nằm ở hoặc gần đỉnh mỗi cột**, rõ nhất là Cox HR — trong khi số thô thì không. Đây là bằng chứng nhất quán rằng mã hóa NLP đóng góp tín hiệu tiên lượng thật, dù khác biệt từng chỉ số còn nhỏ ở cỡ mẫu này."

> 📌 *Ý nghĩa & cách tính từng chỉ số: Phụ lục B. Ví dụ số minh họa: Phụ lục C.*

---

## Slide 16 — Bàn luận: Vì sao NLP tốt & khi nào dùng OvO (⏱️ 1,5 phút)

🖼️ **Slide**
- **Vì sao NLP > số thô?** Sentence embedding nắm bắt tương tác **phi tuyến** giữa các biến (tuổi cao + albumin thấp + ECOG kém); BN hồ sơ giống nhau → vector gần nhau
- **Vì sao chưa đạt ý nghĩa thống kê?** Hạn chế về cỡ mẫu: n=247 → CI rộng ~±0,07; cần ~1.200–1.500 BN để phát hiện ΔAUC≈0,02 ở power 80%
- **Khi nào dùng OvO?** Phương thức trùng lặp (ít phương thức) → OvO; nhiều phương thức bổ sung → cooperative

🎙️ **Lời thuyết trình**
> "Phần bàn luận trả lời ba câu hỏi. Thứ nhất, vì sao NLP tốt hơn số thô? Em cho rằng sentence embedding nắm bắt được các tương tác phi tuyến giữa các biến lâm sàng — ví dụ ý nghĩa tiên lượng kết hợp của tuổi cao, albumin thấp và thể trạng kém — những thứ không được biểu diễn tường minh trong một vector số 13 chiều. Không gian embedding đặt những bệnh nhân có hồ sơ tương tự ở gần nhau.
>
> Thứ hai, vì sao cải thiện chưa đạt ý nghĩa thống kê? Đây là vấn đề về power chứ không phải không có hiệu ứng. Với 247 bệnh nhân, khoảng tin cậy AUC rộng khoảng cộng trừ 0,07; để phát hiện một mức tăng thật khoảng 0,02 với power 80%, cần tới 1.200 đến 1.500 bệnh nhân. Việc cải thiện nhất quán theo cùng một hướng ở cả hai nhánh và cả hai biến thể NLP ủng hộ giả thuyết đây là hiệu ứng thật nhưng đang thiếu cỡ mẫu.
>
> Thứ ba, khi nào nên dùng OvO? Khi các phương thức trùng lặp thông tin thì OvO có lợi; khi có nhiều phương thức thực sự bổ sung thì attention hợp tác khai thác tốt hơn."

---

## Slide 17 — Hạn chế (⏱️ 45 giây)

🖼️ **Slide**
- Cohort **đơn trung tâm, hồi cứu** → có thể sai lệch chọn mẫu, hạn chế khái quát hóa
- Cỡ mẫu nhỏ (n=247) → cải thiện AUC chưa đạt ý nghĩa thống kê
- Sentence encoder dùng **zero-shot**, chưa fine-tune cho thuật ngữ NSCLC
- PD-L1 chấm theo phương pháp Sauter → có thể khác assay SP142/22C3 ở nơi khác

🎙️ **Lời thuyết trình**
> "Em cũng xin nêu rõ các hạn chế. Đây là cohort đơn trung tâm và hồi cứu, nên có thể có sai lệch chọn mẫu và hạn chế tính khái quát. Cỡ mẫu 247 là nhỏ, khiến cải thiện AUC chưa đạt ý nghĩa thống kê. Sentence encoder được dùng theo kiểu zero-shot, chưa fine-tune cho thuật ngữ chuyên biệt của NSCLC. Và điểm PD-L1 được chấm theo phương pháp Sauter, có thể khác với các assay được dùng ở cơ sở khác. Những hạn chế này định hướng cho công việc tiếp theo là validation trên cohort lớn, đa trung tâm."

---

## Slide 18 — Kết luận (⏱️ 1 phút)

🖼️ **Slide**
- Mở rộng DyAM với **2 đóng góp: OvO attention + NLP encoding**
- OvO: tốt nhất ở cấu hình ít phương thức; cho mô hình tốt nhất toàn bộ (AUC 0,800)
- NLP: cải thiện AUC ổn định (tới +0,030) và **giá trị tiên lượng độc lập mạnh nhất** trong phân tích sống còn
- Thực tiễn: **không cần fine-tune**, áp dụng được cho mọi trường EHR có cấu trúc
- → Cần validation đa trung tâm, cỡ mẫu lớn hơn

🎙️ **Lời thuyết trình**
> "Tóm lại, em đã mở rộng mô hình DyAM với hai đóng góp: cơ chế attention cạnh tranh OvO và cách mã hóa biến lâm sàng bằng NLP. OvO cho hiệu năng tốt nhất ở các cấu hình ít phương thức và tạo ra mô hình tốt nhất toàn bộ với AUC 0,800. Còn mã hóa NLP cải thiện AUC một cách ổn định tới 0,030, và quan trọng hơn, mang lại giá trị tiên lượng độc lập mạnh nhất trong phân tích sống còn.
>
> Về mặt thực tiễn, phương pháp NLP này không cần fine-tune, áp dụng được cho bất kỳ trường dữ liệu EHR có cấu trúc nào có thể diễn đạt thành câu. Hướng tiếp theo là kiểm chứng trên các cohort lớn, đa trung tâm để khẳng định và đưa vào ứng dụng lâm sàng cho việc lựa chọn điều trị. Em xin cảm ơn thầy cô và các bạn đã lắng nghe, và rất mong nhận được câu hỏi."

---

## Phụ lục — Dự kiến câu hỏi & gợi ý trả lời (Q&A)

**Q1. Tại sao cải thiện AUC không có ý nghĩa thống kê mà vẫn coi là đóng góp?**
> Vì đây là giới hạn power do cỡ mẫu (n=247), không phải không có hiệu ứng. Tính toán cho thấy cần ~1.200–1.500 BN để phát hiện ΔAUC≈0,02. Quan trọng hơn, hiệu ứng nhất quán về HƯỚNG ở cả 2 nhánh + 2 biến thể NLP, và trong phân tích sống còn (HR, time-dependent AUC) thì NLP cho tín hiệu rõ rệt và mạnh nhất.

**Q2. Vì sao chuyển số thành câu rồi mã hóa lại tốt hơn để nguyên số?**
> Vì sentence embedding nắm bắt tương tác phi tuyến giữa các biến và đặt các bệnh nhân có hồ sơ tương tự gần nhau trong không gian embedding. Số thô buộc mô hình tự học các tương tác này từ tập nhỏ → khó hơn.

**Q3. Vì sao BioClinBERT (chuyên y khoa) lại kém hơn MiniLM tổng quát?**
> Vì BioClinBERT được pretrain cho masked language modeling trên ghi chú lâm sàng, không tối ưu cho độ tương đồng NGỮ NGHĨA giữa các câu. MiniLM được huấn luyện đúng cho mục tiêu sentence similarity — phù hợp với cách dùng của chúng tôi.

**Q4. OvO khác attention gốc ở đâu, một câu?**
> Attention gốc chia một ngân sách cố định (tổng trọng số = 1, các phương thức ràng buộc nhau); OvO tính trọng số ĐỘC LẬP qua so sánh mỗi phương thức với trung bình các phương thức còn lại — kiểu "best-of", phù hợp khi phương thức trùng lặp.

**Q5. Phân biệt "predictive" và "prognostic" nghĩa là gì?**
> Predictive = dự đoán đáp ứng nhị phân với điều trị (PR/CR vs SD/PD). Prognostic = tiên lượng thời gian sống/tiến triển bất kể điều trị. Kết quả cho thấy NLP encoding mạnh ở khía cạnh prognostic (sống còn) hơn là predictive (AUC nhị phân).

**Q6. Tại sao kết hợp (số thô + NLP) lại tệ hơn?**
> Vì tăng số phương thức lên 8 trong khi n=247 → tỷ lệ n/N≈31 quá thấp, khiến việc học attention bất ổn. Do đó khuyến nghị NLP là PHƯƠNG ÁN THAY THẾ, không phải bổ sung, cho biến lâm sàng số.

**Q7. Mô hình có chạy được khi bệnh nhân thiếu một số phương thức không?**
> Có. DyAM xử lý dữ liệu thiếu tự nhiên bằng masking — phương thức thiếu được gán mask = 0 và không tham gia tính attention, nên không phải loại bỏ bệnh nhân.

**Q8. "NLP-PCA16" là gì? Khác "NLP raw" thế nào?**
> NLP raw là embedding **384 chiều** lấy thẳng từ sentence-transformer MiniLM-L6-v2. NLP-PCA16 là dùng **PCA (phân tích thành phần chính) nén 384 → 16 chiều**, giữ lại 16 thành phần quan trọng nhất (≈ 80% phương sai). Lý do nén: 384 đặc trưng cho **một** phương thức là quá nhiều so với n = 247 → dễ overfit và **lấn át** các phương thức khác (genomics, PD-L1 chỉ vài–vài chục chiều); nén về 16 giúp cân bằng và ổn định việc học attention. Bài thử cả hai: PCA16 tốt hơn ở nhánh IHC-A, raw tốt hơn ở IHC-G.

**Q9. Vì sao tách IHC-A và IHC-G? Có phải hai mẫu mô khác nhau không?**
> **Không** — cùng một lát cắt nhuộm PD-L1 IHC, nhưng **hai cách trích xuất đặc trưng khác nhau**: **IHC-A** (Aggregation) tổng hợp **cường độ điểm ảnh** PD-L1, tức "bao nhiêu" PD-L1 (18 đặc trưng); **IHC-G** (GLCM) lấy **đặc trưng kết cấu** Gray-Level Co-occurrence Matrix, tức "sắp xếp/họa tiết như thế nào" (≈150 đặc trưng). Chạy **riêng từng nhánh** vì đây là hai **biểu diễn thay thế nhau** của cùng modality bệnh học — mục tiêu là so sánh biểu diễn nào tốt hơn; gộp cả hai sẽ làm số phương thức/số chiều phình quá mức (n/N thấp → attention bất ổn, đúng như trường hợp gộp 8 phương thức ở Q6).

**Q10. Phân tích sống còn có nghĩa là model dự đoán đúng tại từng thời điểm không?**
> **Không hẳn** — model chỉ xuất ra **MỘT điểm rủi ro (score)**, không phải một thời điểm. Phân tích sống còn kiểm tra con số đó **có ý nghĩa theo thời gian thế nào**, gồm 2 khía cạnh: **(1) Xếp hạng** đúng thứ tự nguy cơ — *C-index, Cox HR, Kaplan-Meier* (gộp trên toàn trục thời gian); **(2) Đúng "tại từng thời điểm"** — chỉ *time-dependent AUC* (phân biệt tại mốc 6/12/18 tháng) và *Brier score* (xác suất dự đoán khớp thực tế tại mỗi mốc) mới thật sự theo từng thời điểm. → **Ví dụ số cụ thể cho từng chỉ số: xem Phụ lục C.**

---

## Phụ lục B — Giải thích chi tiết 5 chỉ số sống còn

> Phần này giải thích **cách tính, công thức và ý nghĩa** của từng chỉ số dùng trong Slide KQ4 và 2 slide công thức (Slide 15–16 của bản có công thức). Triển khai thực tế bằng `lifelines` (C-index, Cox, Kaplan–Meier) và `scikit-survival` (time-dependent AUC, Brier score).

### Bối cảnh chung
- **Biến thời gian:** `pfs` = thời gian sống không tiến triển (tháng). `pfs_censor = 1` nếu **event** thật sự xảy ra (tiến triển/tử vong), `= 0` nếu **bị kiểm duyệt** (censored — mất theo dõi hoặc chưa có event đến thời điểm cắt).
- **Risk score** của DyAM: điểm **cao = nguy cơ cao = PFS ngắn**. Vì vậy khi đưa vào các hàm xếp hạng, score được **đảo dấu** (−score).
- Vì sao cần chỉ số riêng cho sống còn (không dùng AUC nhị phân)? Vì dữ liệu sống còn có **censoring** và có **trục thời gian** — AUC nhị phân thường bỏ qua cả hai.

---

### 1. Harrell's C-index (chỉ số đồng thuận) — *PHÂN BIỆT*
- **Công thức:**
  $$C = \frac{\#\{\text{cặp xếp hạng đúng}\}}{\#\{\text{cặp so sánh được}\}} = P\big(\hat r_i > \hat r_j \mid T_i < T_j\big)$$
- **Cách tính:** xét mọi **cặp bệnh nhân** "so sánh được" (biết chắc ai có event trước). Đếm tỉ lệ cặp mà model gán **điểm rủi ro đúng thứ tự** (người có event sớm hơn phải có risk cao hơn). Hàm `lifelines.concordance_index(pfs, −score, pfs_censor)`. CI 95% bằng **bootstrap 1000 lần** (percentile 2,5/97,5).
- **Ý nghĩa:** đúng là "AUC dành cho dữ liệu sống còn" — đo khả năng **xếp hạng đúng thứ tự nguy cơ** giữa các bệnh nhân. **0,5 = ngẫu nhiên; 1,0 = hoàn hảo.**
- **Trong bài:** cải thiện nhẹ và nhất quán với NLP ($\Delta C = +0{,}005$ ở cả 2 nhánh) nhưng **chưa có ý nghĩa** (Wilcoxon $p = 0{,}49$/$0{,}38$) — do **phương sai giữa các fold lớn** (C dao động 0,52–0,78) và mỗi fold chỉ ~25 BN.
- 🎙️ *Nói:* "C-index giống AUC nhưng cho sống còn: lấy mọi cặp bệnh nhân, đếm tỉ lệ model xếp đúng ai nguy cơ cao hơn. NLP cải thiện nhẹ nhưng phương sai giữa fold lớn nên chưa đủ ý nghĩa."

### 2. Time-dependent AUC (cumulative/dynamic) — *PHÂN BIỆT THEO THỜI GIAN*
- **Công thức (tại mốc $t$):**
  $$\mathrm{AUC}(t) = P\big(\hat r_i > \hat r_j \mid T_i \le t < T_j\big)$$
- **Cách tính:** tại mỗi mốc **t = 6, 12, 18 tháng**, chia bệnh nhân thành "**case**" (đã có event trước/đến t) và "**control**" (còn sống sau t), rồi tính AUC của ROC động. Dùng `sksurv.metrics.cumulative_dynamic_auc`, có **hiệu chỉnh kiểm duyệt bằng trọng số IPCW** (inverse-probability-of-censoring weighting).
- **Ý nghĩa:** cho biết model phân biệt tốt ở **ngắn hạn hay dài hạn** — điều mà một con số C-index gộp không thấy được.
- **Trong bài:** lợi thế của NLP **tập trung ở 12–18 tháng** ($+0{,}024$ và $+0{,}020$ ở IHC-A), gần như không ở 6 tháng ($+0{,}002$) → gợi ý NLP mã hóa thông tin **tiên lượng dài hạn**.
- 🎙️ *Nói:* "AUC phụ thuộc thời gian tách C-index ra theo từng mốc. NLP chỉ có lợi thế rõ ở 12–18 tháng chứ không ở 6 tháng — đúng với giả thuyết nó mang thông tin tiên lượng dài hạn."

### 3. Cox proportional hazards / Hazard Ratio (HR) — *HỒI QUY NGUY CƠ*
- **Mô hình & công thức:**
  $$h(t\mid\mathbf{x}) = h_0(t)\,\exp(\beta_1 x_1 + \dots + \beta_p x_p),\qquad \mathrm{HR}_{\text{score}} = e^{\beta_{\text{score}}}$$
- **Cách tính:** hồi quy Cox **đa biến** với `lifelines.CoxPHFitter(penalizer=0.1)` — biến quan tâm là **risk score** (đã scale về [0,1] để HR so sánh được giữa các model), **hiệu chỉnh** cho 5 yếu tố lâm sàng: tuổi, ECOG, albumin, dNLR, di căn gan. `penalizer=0.1` là **regularization L2** chống quá khớp khi các biến cộng tuyến.
- **Ý nghĩa:** $h_0(t)$ là nguy cơ nền; HR = **nguy cơ tương đối** khi score đi từ 0 → 1, **giữ các biến khác cố định**. **HR > 1 ⇒ score cao thì nguy cơ tiến triển/tử vong cao hơn.** p-value (kiểm định Wald) đo việc $\beta_{\text{score}} \neq 0$ — tức score có giá trị tiên lượng **độc lập** với 5 biến lâm sàng.
- **Trong bài:** HR cao nhất luôn thuộc NLP — **6,07** (NLP-PCA16, IHC-A) vs 5,30 (không lâm sàng); **4,97** (NLP raw, IHC-G) vs 4,74; **số thô KHÔNG cải thiện HR**. Tất cả $p < 0{,}001$.
- 🎙️ *Nói:* "Cox cho biết nguy cơ tức thời. HR bằng 6,07 nghĩa là nhóm risk score cao có nguy cơ tiến triển gấp khoảng 6 lần, ngay cả khi đã hiệu chỉnh cho 5 biến lâm sàng — chứng tỏ điểm số mang thông tin độc lập. Đáng chú ý: chỉ NLP nâng được HR, số thô thì không."

### 4. Integrated Brier Score (IBS) — *HIỆU CHUẨN (calibration)*
- **Công thức:**
  $$\mathrm{BS}(t) = \frac{1}{N}\sum_{i} w_i\big(\hat S(t\mid\mathbf{x}_i) - \mathbf{1}\{T_i > t\}\big)^2,\qquad
  \mathrm{IBS} = \frac{1}{t_{\max}-t_{\min}}\int_{t_{\min}}^{t_{\max}} \mathrm{BS}(t)\,dt$$
- **Cách tính:** dùng Cox đơn biến chuyển risk score → **xác suất sống** $\hat S(t\mid\mathbf{x})$, rồi đo **sai số bình phương** giữa xác suất dự đoán và outcome thực $\mathbf{1}\{T_i>t\}$, có **trọng số IPCW** $w_i$ cho censoring. Lấy **tích phân** (trung bình) trên dải thời gian → IBS. Dùng `sksurv.metrics.brier_score` + `integrated_brier_score`.
- **Ý nghĩa:** vừa đo **độ phân biệt** vừa đo **độ hiệu chuẩn** (xác suất dự đoán có "khớp" với thực tế không). **Càng thấp càng tốt; 0,25 = mô hình ngẫu nhiên.**
- **Trong bài:** **0,172–0,175** cho mọi model (đều dưới 0,25 → hiệu chuẩn tốt), thấp nhất là NLP-PCA16 (IHC-A) = 0,1716.
- 🎙️ *Nói:* "Brier score là sai số bình phương giữa xác suất sống dự đoán và thực tế — càng thấp càng tốt, 0,25 là mức ngẫu nhiên. Mọi model của chúng tôi đều quanh 0,17, tức hiệu chuẩn tốt."

### 5. Kaplan–Meier + kiểm định log-rank — *PHÂN TẦNG NGUY CƠ*
- **Công thức:**
  $$\hat S(t) = \prod_{t_i \le t}\Big(1 - \frac{d_i}{n_i}\Big),\qquad
  \chi^2_{\text{log-rank}} = \frac{\big(\sum_k (O_k - E_k)\big)^2}{\sum_k \mathrm{Var}(O_k)}$$
- **Cách tính:** chia bệnh nhân thành **2 nhóm** tại ngưỡng **score = 0** (nguy cơ cao vs thấp). KM ước lượng đường sống còn: ở mỗi thời điểm có event, nhân thêm thừa số $(1 - d_i/n_i)$ với $d_i$ = số event, $n_i$ = số còn theo dõi. **Log-rank** so sánh **số event quan sát $O_k$ với kỳ vọng $E_k$** nếu 2 nhóm giống nhau → ra thống kê $\chi^2$. Dùng `lifelines.KaplanMeierFitter` + `logrank_test`.
- **Ý nghĩa:** minh chứng **trực quan + thống kê** rằng risk score chia được **2 nhóm nguy cơ khác biệt có ý nghĩa**. $\chi^2$ càng lớn / p càng nhỏ ⇒ tách nhóm càng rõ.
- **Trong bài:** **mọi** phân tầng đạt $p < 0{,}005$; $\chi^2$ cao nhất = **28,92** (IHC-G + NLP raw).
- 🎙️ *Nói:* "Kaplan–Meier vẽ đường sống còn cho 2 nhóm nguy cơ cao/thấp chia tại score = 0; log-rank kiểm định hai đường có thật sự khác nhau. Tất cả đều p nhỏ hơn 0,005, cao nhất là cấu hình NLP raw."

---

### Bảng tóm tắt nhanh (để trả lời hội đồng)

| Chỉ số | Đo cái gì | Tốt khi | Con số trong bài |
|---|---|---|---|
| **C-index** | Phân biệt (xếp hạng nguy cơ) | → 1,0 | +0,005 (chưa có ý nghĩa) |
| **td-AUC(t)** | Phân biệt tại từng mốc t | → 1,0 | +0,024 @12m, +0,020 @18m |
| **Cox HR** | Nguy cơ tương đối, độc lập | > 1 & p nhỏ | **6,07** vs 5,30; p<0,001 |
| **IBS** | Hiệu chuẩn + chính xác | → 0 (<0,25) | 0,172–0,175 |
| **KM + log-rank** | Tách nhóm nguy cơ | χ² lớn, p nhỏ | mọi p<0,005; χ²=28,92 |

> **Thông điệp xuyên suốt (nhấn mạnh khi bị hỏi):** AUC nhị phân của NLP cải thiện nhưng chưa có ý nghĩa (thiếu power ở n=247); **nhưng phân tích sống còn — đặc biệt Cox HR và td-AUC dài hạn — cho tín hiệu mạnh và nhất quán nhất**, cho thấy NLP encoding mang giá trị **tiên lượng (prognostic)** rõ hơn giá trị **dự đoán đáp ứng (predictive)**.

---

## Phụ lục C — Ví dụ số: mỗi chỉ số nói gì về model

> Dùng **chung một bộ 5 bệnh nhân** chạy qua cả 5 chỉ số, để thấy rõ **từ mỗi chỉ số ta kết luận được gì về model**. Quy ước: model xuất **risk score**, **score cao = nguy cơ cao = sống ngắn**.

**Bộ dữ liệu ví dụ:**

| BN | Risk score (model) | PFS thực tế | Trạng thái |
|---|---|---|---|
| P1 | 0,9 | 2 tháng | tiến triển (event=1) |
| P2 | 0,7 | 4 tháng | tiến triển (event=1) |
| P3 | 0,5 | 10 tháng | tiến triển (event=1) |
| P4 | 0,3 | 15 tháng | tiến triển (event=1) |
| P5 | 0,1 | 20 tháng | *censored* (còn ổn ở tháng 20) |

Ở ví dụ này model xếp hạng **hoàn hảo** (score càng cao, PFS càng ngắn).

---

### ① C-index — *model xếp hạng nguy cơ đúng cỡ nào?*

**Tính:** lấy **mọi cặp** bệnh nhân, đếm cặp "đúng" (score cao hơn → hỏng sớm hơn).
- (P1,P3): 0,9 > 0,5 **và** 2 < 10 → đúng ✓ · (P2,P4): 0,7 > 0,3 và 4 < 15 → đúng ✓ · … cả 10 cặp đều đúng → **C = 1,0**.
- *(P5 censored vẫn dùng được: ta biết P5 sống ≥ 20 tháng > mọi người khác, nên mọi người "hỏng trước P5"; P5 lại có score thấp nhất → các cặp với P5 đều đúng.)*
- **Model "dở" (score ngẫu nhiên):** nhiều cặp sai → C ≈ 0,5 (như tung đồng xu).

> 🔎 **Nhận định về model:** C cho biết **khả năng XẾP HẠNG tổng thể** (gộp mọi thời điểm). C=1 hoàn hảo, 0,5 vô dụng. Bài thực ≈ **0,63** → tốt hơn ngẫu nhiên rõ rệt. **Ứng dụng:** xếp ưu tiên bệnh nhân nào cần theo dõi sát hơn.

### ② Time-dependent AUC — *model mạnh tại MỐC thời gian nào?*

**Tính:** tại mỗi mốc, chia "case" (đã hỏng) vs "control" (còn ổn), rồi hỏi case có score cao hơn control không.
- **Tháng 6:** case = {P1, P2}; control = {P3, P4, P5}. Score case (0,9; 0,7) **>** control (0,5; 0,3; 0,1) → **AUC(6m) = 1,0**.
- **Tháng 12:** case = {P1, P2, **P3**}; control = {P4, P5}. (0,9; 0,7; 0,5) **>** (0,3; 0,1) → **AUC(12m) = 1,0**.

> 🔎 **Nhận định về model:** đây là chỉ số **theo TỪNG mốc**. Bài thực: ở 6 tháng các model ngang nhau (~0,72), nhưng 12–18 tháng NLP vượt lên → **NLP mạnh ở dài hạn**. **Ứng dụng:** tiên lượng tại mốc lâm sàng cụ thể (vd tại lần tái khám 12 tháng).

### ③ Cox Hazard Ratio — *model có mang thông tin tiên lượng MỚI không, mạnh cỡ nào?*

**Tính:** chạy hồi quy Cox với score (scale [0,1]) + các biến lâm sàng. Ra **HR**.
- HR = 6,07 nghĩa: bệnh nhân score 1,0 so với score 0,0 có **nguy cơ tiến triển tức thời gấp ~6 lần**, *sau khi đã trừ ảnh hưởng* của tuổi, ECOG, albumin, dNLR, di căn gan.
- **HR = 1** → score không liên quan nguy cơ (vô giá trị). **HR > 1 và p < 0,05** → yếu tố nguy cơ **độc lập**.

> 🔎 **Nhận định về model:** HR đo **giá trị tiên lượng ĐỘC LẬP** (ngoài các biến lâm sàng sẵn có) và độ mạnh. Bài thực: NLP **6,07 > 5,30** (không lâm sàng), còn số thô không tăng → **NLP thêm thông tin thật**, đáng đưa vào quyết định lâm sàng.

### ④ Integrated Brier Score — *xác suất model đưa ra có ĐÁNG TIN không?*

**Tính:** model chuyển score → **xác suất sống** S(t). Đo sai số bình phương giữa xác suất dự đoán và thực tế.
- Tốt: model nói "P5 có **85%** còn sống ở tháng 12", thực tế P5 còn sống → sai số (0,85 − 1)² = 0,02 (nhỏ).
- Tệ: model nói "P1 có **90%** còn sống ở tháng 12", nhưng P1 đã hỏng ở tháng 2 → sai số (0,90 − 0)² = 0,81 (lớn).
- IBS = trung bình sai số này qua các mốc. **0 = hoàn hảo; 0,25 = đoán bừa 50/50.**

> 🔎 **Nhận định về model:** Brier đo **HIỆU CHUẨN** — xác suất có khớp thực tế không (khác xếp hạng). Bài thực **0,17 < 0,25** → đáng tin. **Ứng dụng:** có thể nói với bệnh nhân "khoảng X% còn ổn ở 12 tháng" một cách có cơ sở.

### ⑤ Kaplan–Meier + log-rank — *model có CHIA NHÓM nguy cơ khác biệt rõ không?*

**Tính:** chia 2 nhóm theo ngưỡng score (vd nhóm cao = {P1, P2}; nhóm thấp = {P3, P4, P5}). Vẽ 2 đường sống còn.
- Nhóm cao tụt nhanh (median ~3 tháng); nhóm thấp tụt chậm (median ~15 tháng) → **2 đường tách xa**.
- **Log-rank** so số ca hỏng *quan sát* vs *kỳ vọng* nếu 2 nhóm giống nhau → χ² lớn, **p nhỏ** → khác biệt thật.
- **Model "dở":** 2 nhóm trộn lẫn PFS → 2 đường KM **chồng nhau** → p > 0,05.

> 🔎 **Nhận định về model:** KM/log-rank cho biết model có **PHÂN TẦNG** được bệnh nhân thành nhóm nguy cơ khác biệt không. Bài thực: mọi p < 0,005 → tách rõ. **Ứng dụng:** đặt ngưỡng score để chia nhóm "theo dõi sát" vs "theo dõi nhẹ".

---

### Bảng tổng hợp — mỗi chỉ số → kết luận gì về model → dùng làm gì

| Chỉ số | Trả lời câu hỏi | Kết luận về model | Ứng dụng lâm sàng |
|---|---|---|---|
| **C-index** | Xếp hạng nguy cơ đúng không? | Khả năng phân biệt **tổng thể** | Ưu tiên theo dõi |
| **td-AUC** | Mạnh tại **mốc nào**? | Ngắn hạn hay **dài hạn** | Tiên lượng tại mốc tái khám |
| **Cox HR** | Có thông tin **mới & độc lập** không? | Giá trị tiên lượng độc lập, độ mạnh | Đưa vào mô hình quyết định |
| **Brier/IBS** | Xác suất có **đáng tin** không? | **Hiệu chuẩn** (calibration) | Nói % cho bệnh nhân |
| **KM + log-rank** | Chia **nhóm** rõ không? | **Phân tầng** nguy cơ | Đặt ngưỡng nhóm điều trị |

> **Mẹo trả lời nhanh:** *C-index & td-AUC* = "phân biệt" (xếp ai trước ai); *Brier* = "đúng xác suất" (hiệu chuẩn); *Cox HR* = "độc lập & mạnh cỡ nào"; *KM* = "chia nhóm có ý nghĩa". Một model tốt cần **cả phân biệt lẫn hiệu chuẩn** — bài này đạt cả hai (C>0,5, IBS<0,25).

---

## Ghi chú trình bày (tips)

- **Phân bổ thời gian:** dành nhiều nhất cho Slide 7 (kiến trúc attention) và Slide 13–15 (động lực + kết quả + bảng sống còn) — đây là điểm nhấn học thuật.
- **Khi bị hỏi sâu về thống kê:** luôn quay về thông điệp "hướng cải thiện nhất quán + tín hiệu sống còn mạnh = hiệu ứng thật nhưng thiếu power".
- **Trung thực về hạn chế** (CI chồng lấn, cỡ mẫu) tạo ấn tượng tốt với hội đồng hơn là né tránh.
- Nếu thiếu thời gian, có thể gộp Slide 10 vào Slide 11, và rút gọn Slide 17 (Hạn chế).
- Số liệu chủ chốt nên thuộc lòng: **AUC tốt nhất 0,813** (NLP raw, IHC-G), **0,800** (OvO Rad+IHC-G+Gen+PDL1), **HR 6,07** (NLP-PCA16, IHC-A), **n = 247**.
