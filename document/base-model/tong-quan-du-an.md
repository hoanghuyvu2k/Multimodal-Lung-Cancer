# Slide 1 - Tong quan project

## Ten project
- Du an mo hinh da phuong thuc cho bai toan du doan dap ung dieu tri ung thu phoi.
- Pipeline ket hop radiomics + genomics + pathology + clinical, xu ly du lieu thieu modality.

## Muc tieu su dung
- Du doan nhan nhi phan `label` (0/1) tu nhieu nguon du lieu y sinh.
- Danh gia kha nang phan biet bang AUC/ROC va khoang tin cay.
- Giai thich du doan theo tung modality (risk, attention, share).

---

# Slide 2 - Bai toan va nhan du doan

## Bai toan
- Phan loai nhan 0/1 tren benh nhan ung thu phoi.
- Mapping nhan trong code:
  - `SD`, `POD`, ... -> `label = 1`
  - `PR`, `CR` -> `label = 0`

## Gia tri thuc te
- Kich ban huan luyen thuong dung: KFold hoac ShuffleSplit.
- Co ho tro LOO (Leave-One-Out) khi can danh gia tren tap nho.

---

# Slide 3 - Bo du lieu su dung

## Cac file du lieu chinh
- `datasets/lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet`
- `datasets/genomic_data_v3.parquet`
- `datasets/lung_pathology_pdl1_glcm_v3.parquet`
- `datasets/pdl1_score.parquet`
- `datasets/final_cohort_listing.csv`

## Kich thuoc du lieu thuc te (rows x cols)
- Radiomics train: `3996 x 1690` (index unique ~`187` benh nhan)
- Radiomics validation: `1176 x 1690`
- Genomics: `247 x 11` (index unique `247` benh nhan)
- Pathology GLCM: `105 x 150` (index unique `105` benh nhan)
- PD-L1 score: `201 x 1` (index unique `201` benh nhan)
- Cohort listing: `368 x 2`

---

# Slide 4 - So chieu / kich thuoc dac trung

## Radiomics
- 1690 cot tong, trong do:
  - 2 cot metadata: `job_tag`, `lesion_index`
  - 1688 cot radiomics features
- Cac image transform: original, wavelet, LBP, gradient, logarithm, square...

## Cac modality khac
- Genomics: 11 features (vi du TMB va cac bien driver mutation).
- Pathology (PD-L1/GLCM): 150 features.
- Clinical/PD-L1 score: 1 feature trong file `pdl1_score.parquet`.

## Shape tensor trong mo hinh
- Moi modality: `[n_samples, n_features_i]`
- Mask modality: `[n_samples, n_modalities]`
- Label: `[n_samples]`

---

# Slide 5 - Mo hinh su dung

## Mo hinh chinh
- `MultiModalDynamicModel` (PyTorch):
  - Moi modality co 1 risk linear: `Linear(n_features_i -> 1)`
  - Attention matrix de hoc trong so dong giua cac modality
  - Masking de bo qua modality bi thieu

## Bien the trong project
- `MultiModalDynamicModelOvO`
- `MultiModalDynamicModelOvO_Softmax`
- `MultiModalDynamicModelGMU` (Gated Multimodal Unit)

## Training setup
- Loss: `BCEWithLogitsLoss` + regularization (`alpha`, `beta`)
- Optimizer: Adam
- Co feature selection L1 cho mot so modality

---

# Slide 6 - Diem manh cua project

## Tai sao project huu ich
- Hop nhat da nguon du lieu y hoc trong cung mot mo hinh.
- Xu ly duoc missing modality (thuc te benh vien rat hay gap).
- Co kha nang giai thich theo benh nhan:
  - risk tung modality
  - attention tung modality
  - share dong gop
- De so sanh nhieu kien truc (attention goc vs GMU) tren cung pipeline.

---

# Slide 7 - Demo mot slide ket luan ngan

## Ket luan
- Day la he thong du doan dap ung dieu tri ung thu phoi da phuong thuc.
- Du lieu gom radiomics, genomics, pathology va clinical voi kich thuoc khac nhau.
- Mo hinh trung tam la `MultiModalDynamicModel` va bien the GMU/OvO.
- Kien truc duoc thiet ke de:
  - ho tro du lieu thieu,
  - tang hieu nang phan loai,
  - va giai thich duoc quyet dinh.
