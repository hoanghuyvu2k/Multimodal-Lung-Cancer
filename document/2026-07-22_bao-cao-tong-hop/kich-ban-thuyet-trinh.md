# Kịch bản thuyết trình — Tổng hợp thực nghiệm 21–22/07

_Mỗi mục tương ứng một slide. Thời lượng gợi ý: ~45–60 giây/slide, tổng ~9–11 phút._

## Slide 1 — Bìa — thông điệp chính

Xin chào. Đây là tổng hợp hai ngày thực nghiệm cho bài toán dự đoán đáp ứng liệu pháp miễn dịch NSCLC từ dữ liệu đa nguồn. Câu hỏi lớn: các phương pháp phức tạp — attention học được, foundation model, embedding từ ảnh gốc — có thực sự hơn một baseline đơn giản không? Tôi nói luôn kết luận: sau sáu nhóm thí nghiệm, không phương pháp nào vượt được baseline uniform_avg. Trần hiệu năng nằm ở bản thân dữ liệu, không phải ở mô hình. Điểm dương duy nhất, và cũng là đóng góp mạnh nhất, là phát hiện về generalization ở cuối bài.

## Slide 2 — Bối cảnh & câu hỏi

Bài toán: kết hợp năm nguồn — radiomics từ CT, pathology từ nhuộm PD-L1, genomic, điểm PD-L1, và xét nghiệm lâm sàng. Mô hình gốc trong bài báo là DyAM, dynamic attention fusion. Tôi kiểm chứng ba câu hỏi: attention có hơn trung bình đơn giản; foundation embedding có phá được trần; và TabPFN có hơn hồi quy tuyến tính. Về phương pháp, mọi con số đều là trung bình năm seed mười-fold có hoán vị phân hoạch, kèm kiểm định bootstrap trên bệnh nhân — để chống việc chỉ chạy một lần rồi lấy số đẹp. Và tôi luôn kiểm trên cohort ngoài, vì AUC nội bộ đã kịch trần khoảng 0.78.

## Slide 3 — Mô hình baseline: uniform_avg

Đây là baseline uniform_avg. Nó có hai phần. Phần HỌC: mỗi nguồn có một head hồi quy nhỏ — rᵢ bằng tanh của Wᵢ nhân xᵢ — các trọng số W này được huấn luyện bình thường bằng gradient descent. Phần KHÔNG học: trọng số kết hợp aᵢ bằng mask chia tổng mask, tức chia đều cho các nguồn bệnh nhân có, nguồn nào vắng thì tự loại. Về bản chất đây chính là DyAM nhưng cơ chế attention bị vô hiệu hoá, cố định về chia đều. Nó học cách đọc từng nguồn rồi cộng đều, thay vì học tin nguồn nào hơn.

## Slide 4 — Mô hình attention: DyAM (A) + OvO (B)

Để so sánh, đây là hai biến thể attention HỌC được của DyAM. Panel A là cooperative attention: một ma trận N nhân N học tương tác giữa các nguồn. Panel B là competitive, gọi là OvO — mỗi nguồn cạnh tranh với trung bình các nguồn còn lại. Cả hai đều học trọng số aᵢ một cách có tham số. Câu hỏi rất tự nhiên: tầng attention học được này có đáng giá không? Slide sau trả lời.

## Slide 5 — A · Fusion attention vs uniform_avg

Câu trả lời là không. Tôi chạy đủ mọi tổ hợp nguồn, hơn bốn mươi cấu hình. uniform_avg luôn ngang hoặc hơn DyAM và OvO. Quan trọng hơn: khoảng cách nghiêng về uniform lại TĂNG khi thêm nguồn — ngược hẳn kỳ vọng 'nhiều nguồn thì attention thắng'. DyAM chỉ thắng có ý nghĩa ở hai trên hai mươi sáu tổ hợp, đều ở trường hợp ít nguồn nhất. Kết luận: attention học được không đóng góp gì, nó chỉ tái lập phép chia đều.

## Slide 6 — Tổng quan 3 mô hình fusion

Đây là cái nhìn tổng quan ba mô hình fusion trên bốn tổ hợp benchmark. Nhìn bảng: uniform_avg và OvO gần như trùng nhau, bám sát nhau trong khoảng năm phần nghìn, và cả hai đều ngang hoặc hơn DyAM. Chênh lệch giữa ba mô hình chỉ khoảng một phần trăm, phần lớn không đạt ý nghĩa thống kê — tức chúng tương đương trong nhiễu. Con số 0.79 đến 0.80 mà bài báo báo cáo là kết quả single-run của DyAM và OvO trên một phân hoạch thuận lợi. Thông điệp: không mô hình attention nào vượt được baseline đơn giản một cách bền vững, nên ta chọn baseline vì nó đơn giản và ổn định hơn.

[GIẢI THÍCH single-run vs 5-seed — dùng khi hội đồng hỏi 'sao số thấp hơn bài báo gốc']: Đánh giá bằng 10-fold cross-validation cần chia ngẫu nhiên bệnh nhân thành các nhóm; cách chia phụ thuộc một 'seed' ngẫu nhiên. SINGLE-RUN là chạy với MỘT cách chia; 5-SEED là trung bình của NĂM cách chia khác nhau. Vì cỡ mẫu nhỏ (~250 bệnh nhân), AUC dao động mạnh giữa các cách chia — riêng BM1 dải là 0.726 đến 0.781, tức chỉ đổi cách chia thôi đã lệch tới 0.055. Con số 0.79 của bài báo rơi vào một lần chia THUẬN LỢI, thậm chí cao hơn cả đỉnh của dải 5-seed; còn 0.76 là TRUNG BÌNH, đại diện và ổn định hơn. Bằng chứng cấu hình giống hệt: khi tái lập ĐÚNG kiểu single-run, số của em khớp bài báo tới bốn chữ số ở BM4 (0.6932 so với 0.6931). Vì vậy em báo cáo 5-seed để trung thực về phương sai và chống cherry-pick. Ví von: single-run như chấm một đề thi (có thể dễ/khó bất chợt); 5-seed như trung bình năm đề — công bằng hơn.

## Slide 7 — B·C · Stacked fusion & rà soát cấu hình

Hai kiểm chứng phụ. Bên trái: stacked fusion — cho một meta-model học trọng số nguồn từ dự đoán out-of-fold, kỳ vọng nó tự hạ radiomics. Kết quả ngược lại: nó đề cao radiomics và bỏ pathology, vì học từ tín hiệu nội bộ. Bên phải, câu hỏi quan trọng: vì sao bài báo báo 0.79 còn tôi 0.76? Cấu hình giống hệt — khác biệt thuần ở chỗ bài báo báo MỘT lần chạy, tôi trung bình NĂM phân hoạch. Tái lập đúng kiểu một-lần, số của tôi khớp bài báo tới bốn chữ số. Vậy 0.79 chỉ là một phân hoạch may mắn, không phải cấu hình tốt hơn.

## Slide 8 — D · Quy trình embedding ảnh gốc

Trước khi xem kết quả, đây là cách tôi biến ẢNH GỐC thành vector đặc trưng — pipeline cho cả hai loại ảnh. Với PATHOLOGY: từ slide WSI, tôi cắt thành các tile 256 pixel ở độ phóng 20x, chỉ trong vùng khối u do HALO khoanh; mỗi tile đưa qua Phikon — một Vision Transformer nền tảng — ra vector 768 chiều; rồi mean-pool trung bình mọi tile thành một vector cho mỗi slide. Với RADIOLOGY: từ file CT và mask tổn thương, tôi chọn đúng series theo kích thước, cắt cửa sổ mô mềm, crop quanh lesion, đưa qua ba encoder — BiomedCLIP, MedicalNet 3D và fmcib — rồi cũng mean-pool thành một vector cho mỗi ca. Hai điểm cốt lõi: encoder được ĐÓNG BĂNG, chỉ chạy inference chứ không fine-tune; và cách gộp là mean-pool — đơn giản nhất và cũng là thô nhất, chuẩn mực hơn là ABMIL học trọng số tile. Chính hai lựa chọn này vừa định hình vừa giới hạn kết quả ở slide sau.

## Slide 9 — D · Foundation embedding — kết quả

Đây là nỗ lực lớn nhất — thay đặc trưng thủ công bằng embedding foundation model chạy trực tiếp trên ảnh gốc, cả WSI pathology lẫn CT. Tôi thử bốn model. Với pathology, Phikon giàu tín hiệu hơn ở nội bộ nhưng không vượt được GLCM thủ công ngoài mẫu. Với radiology, ba model — BiomedCLIP, MedicalNet, và cả fmcib chuyên biệt cho ung thư CT mà tôi phải chạy trên Colab — đều gần như ngẫu nhiên ở nội bộ, quanh 0.51. Kể cả fmcib, model đúng nhất, cũng không mang tín hiệu. Ba model độc lập cùng hội tụ về ngẫu nhiên, nên đây không phải lỗi chọn model.

## Slide 10 — E · Embedding trong fusion

Câu hỏi tiếp: embedding đơn lẻ yếu, nhưng khi kết hợp với nguồn khác có bổ sung tín hiệu không? Câu trả lời còn tệ hơn — nó kéo tụt fusion. Càng thay nhiều imaging bằng embedding, AUC càng giảm: từ 0.775 của bản thủ công xuống 0.653 khi embed cả hai. Thứ tự mức hại khớp chính xác với độ nhiễu của từng embedding. Kể cả chỉ thêm fmcib vào tổ hợp tốt nhất cũng giảm ba phần trăm. Giả thuyết 'bù trừ trong fusion' bị bác bỏ dứt khoát.

## Slide 11 — F · TabPFN

Thí nghiệm cuối: TabPFN, một tabular foundation model chuyên dữ liệu nhỏ — đúng ràng buộc của chúng ta. Nhưng nó cũng không vượt hồi quy tuyến tính: thắng không-trên-năm ở per-modality, và một-trên-bốn ở fusion, chỉ ở BM4 là tổ hợp ít nguồn và sạch nhất. Kết quả khớp đúng literature 2025: các model tabular hiện đại thường chỉ ngang ML truyền thống. Nên tôi giữ TabPFN làm comparator hiện đại trong bài, không phải một cải tiến.

## Slide 12 — Generalization — điểm mạnh nhất

Đây là phát hiện xuyên suốt và là điểm mạnh nhất cho paper. Qua mọi thí nghiệm, một tín hiệu bền vững: pathology transfer được ra ngoài mẫu, đạt 0.74 đến 0.77; còn radiomics thì không, thường dưới 0.5 và rất bất ổn — nghĩa là nó học đặc tính máy chụp, không phải sinh học khối u. 'Modality nào generalize' đang là chủ đề nóng trong literature multimodal 2025. Ghép với đóng góp NLP-clinical đã có, ta có câu chuyện mạch lạc: nguồn nào đáng tin, và biểu diễn dữ liệu lâm sàng bằng ngôn ngữ.

## Slide 13 — Kết luận & hướng đi

Tổng kết: sau fusion attention, stacked, bốn foundation embedding và TabPFN, mọi con đường hội tụ về một thông điệp — không gì vượt được baseline đơn giản trên đặc trưng thủ công. Trần nằm ở dữ liệu, không ở mô hình hay feature. Ba đóng góp cho paper: NLP-clinical embedding đã có; phát hiện generalization; và một loạt ablation trung thực cho thấy model đơn giản cộng hiểu dữ liệu thắng việc chạy theo con số. Việc còn lại là chốt thực nghiệm và viết, với generalization và baseline đơn giản làm câu chuyện trung tâm. Xin cảm ơn.

