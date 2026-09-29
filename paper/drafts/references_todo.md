## Step F1 — References cần hoàn thiện / kiểm tra

Trạng thái: 15/18 entries trong `references.bib` đã có DOI đầy đủ và đang
được `\cite{}` trong text. Còn lại:

### A. Cần bổ sung DOI / số trang (đã có tên bài, chỉ thiếu metadata)

1. **Siegel2024** — Cancer Statistics, 2024 (CA Cancer J Clin, vol. 74)
   - Thiếu: `pages`, `doi`
   - Search string: `"Cancer Statistics 2024" Siegel CA Cancer J Clin doi`
   - Gợi ý: DOI thường có dạng `10.3322/caac.XXXXX`

2. **Vaidya2020** — "Novel, Clinically Translatable Lung Cancer Biomarkers
   from Multi-Scale Radiomic Profiling"
   - Thiếu: `journal`, `doi`
   - Search string: `Vaidya "Novel Clinically Translatable Lung Cancer
     Biomarkers" radiomic 2020`
   - Lưu ý: nếu không tìm thấy bài chính xác, có thể thay bằng một bài
     radiomics-NSCLC review khác đã có DOI (ví dụ Trebeschi 2019 hoặc
     Khorrami 2020) và đổi cite key tương ứng trong `methods.tex`/`results.tex`.

### B. Cần tìm citation hoàn toàn mới (placeholder TODO)

3. **DyAM2024** — bài báo gốc của model DyAM (cited trong `methods.tex`,
   `results.tex`, `discussion.tex` — đây là nền tảng phương pháp luận chính).
   - Đây có vẻ là **bài báo trước của nhóm tác giả/nhóm nghiên cứu** (dự án
     gốc trước khi thêm OvO + NLP). Cần xác nhận:
     - Đã publish chưa? Nếu chưa, dùng `\bibitem` dạng "manuscript in
       preparation" hoặc "unpublished" thay vì DOI giả.
     - Nếu đã có preprint (bioRxiv/medRxiv/arXiv), dùng DOI preprint.
   - **Action item cho người dùng**: cung cấp tên tác giả + tiêu đề + nơi
     công bố (hoặc xác nhận "chưa công bố — dùng dạng in-preparation").

### C. Entries trong `references.bib` KHÔNG được `\cite{}` ở đâu (orphan)

Các entry sau tồn tại trong `references.bib` nhưng chưa xuất hiện trong bất
kỳ `\cite{}` nào của `sections/*.tex`. Cần quyết định: (a) thêm citation phù
hợp vào text, hoặc (b) xoá khỏi `references.bib` để tránh entry thừa.

4. **Vaswani2017** ("Attention Is All You Need")
   - Đề xuất: cite trong `methods.tex` §Model Architecture, khi giới thiệu
     khái niệm attention nói chung (trước khi mô tả `AttentionMatrix`/OvO).

5. **Cheerla2019** (Deep Learning multimodal pancancer prognosis)
   - Đề xuất: cite trong `introduction.tex` đoạn 2, cùng nhóm với
     Bodalal2019/Dercle2020/Vaidya2020 (multimodal fusion literature).

6. **Ma2021** ("missing modality multimodal learning") — **toàn bộ field là
   TODO**, cần tìm bài thật.
   - Search string: `"missing modality" multimodal deep learning survey 2021`
     hoặc `Ma "SMIL" missing modality 2021` (paper gợi ý: Ma et al., "SMIL:
     Multimodal Learning with Severely Missing Modality", AAAI 2021).
   - Đề xuất: cite trong `methods.tex` §Data Modalities, đoạn nói về
     `modality_MASK` / xử lý dữ liệu thiếu.
   - Nếu không cần thiết, có thể xoá entry này khỏi `references.bib`.

### D. Tổng kết hành động

| # | Cite key | Việc cần làm | Mức độ ưu tiên |
|---|---|---|---|
| 1 | Siegel2024 | Bổ sung pages + DOI | Thấp (dễ tìm) |
| 2 | Vaidya2020 | Bổ sung journal + DOI hoặc thay bài | Trung bình |
| 3 | DyAM2024 | Cần thông tin từ người dùng (đã publish?) | **Cao — cần hỏi user** |
| 4 | Vaswani2017 | Thêm `\cite` vào methods.tex hoặc xoá | Thấp |
| 5 | Cheerla2019 | Thêm `\cite` vào introduction.tex hoặc xoá | Thấp |
| 6 | Ma2021 | Tìm bài thật (SMIL AAAI 2021?) hoặc xoá | Trung bình |

**Khuyến nghị thực hiện ở Step G6**: xử lý mục 1, 2 (DOI tra cứu được ngay),
4–5 (thêm cite inline, không cần tìm thêm), và 6 (thay bằng SMIL 2021 nếu
phù hợp). Mục 3 (DyAM2024) cần xác nhận từ người dùng trước khi G6 hoàn tất.
