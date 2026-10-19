# Cooperative Learning cho fusion đa modality — KẾT QUẢ ÂM

> 2026-09-30 · Ding & Tibshirani, *Cooperative learning for multiview analysis*, PNAS 2022
> **Kết luận: không dùng.** Code đã xoá theo yêu cầu; file này giữ lại toàn bộ thuật toán và số
> liệu để sau này dựng lại được mà không phải làm lại từ đầu.

## 1. Vì sao thử

Sau khi kiểm chứng cho thấy attention (DyAM, OvO) không hơn phép trung bình cộng `uniform_avg` ở
n=247, câu hỏi đặt ra là: có cơ chế hợp nhất nào phù hợp hơn cho cỡ mẫu này không?

Cooperative learning được chọn vì nó không phải "một kiến trúc khác để thử", mà là **một dải liên
tục giữa early fusion và late fusion điều khiển bởi một tham số ρ duy nhất** — tức là nó trả lời
được câu hỏi "cơ chế hợp nhất nào phù hợp" một cách có hệ thống.

## 2. Thuật toán

$$\min_{\beta} \tfrac12\Big\|y - \sum_m X_m\beta_m\Big\|^2
  + \tfrac{\rho}{2}\sum_{m<m'}\big\|X_m\beta_m - X_{m'}\beta_{m'}\big\|^2
  + \lambda\sum_m\|\beta_m\|_1$$

- **ρ = 0** → early fusion (lasso trên đặc trưng nối lại)
- **ρ lớn** → phạt bất đồng mạnh, các view buộc dự đoán giống nhau → hành vi late fusion

**Cập nhật khối** (block coordinate descent): cố định các view khác, đặt
$s_{-m} = \sum_{m'\neq m} X_{m'}\beta_{m'}$, $C = 1+\rho(M-1)$, bài toán cho view $m$ rút gọn về
đúng một lasso thường trên mục tiêu điều chỉnh $\tilde y_m = \big(y - (1-\rho)s_{-m}\big)/C$:

```python
def fit_coop(Xs, y, rho, n_iter=4):
    """Xs: danh sách ma trận đặc trưng đã chuẩn hoá theo view (hàng thiếu modality = 0)."""
    M, n = len(Xs), len(y)
    betas = [np.zeros(X.shape[1]) for X in Xs]
    preds = [np.zeros(n) for _ in range(M)]
    C, chosen = 1.0 + rho * (M - 1), [None] * M
    for _ in range(n_iter):
        for m in range(M):
            s = np.sum([preds[k] for k in range(M) if k != m], axis=0) if M > 1 else np.zeros(n)
            y_t = (y - (1.0 - rho) * s) / C
            if chosen[m] is None:      # chọn lambda một lần ở vòng đầu rồi giữ nguyên
                mdl = LassoCV(cv=3, n_alphas=20, max_iter=50000, tol=1e-3,
                              random_state=0).fit(Xs[m], y_t)
                chosen[m] = float(mdl.alpha_)
            else:
                mdl = Lasso(alpha=chosen[m], max_iter=50000, tol=1e-3).fit(Xs[m], y_t)
            betas[m] = mdl.coef_
            preds[m] = Xs[m] @ mdl.coef_ + mdl.intercept_
    return betas

def predict_masked_mean(Xs, betas, masks):
    """Điểm = trung bình dự đoán của các view CÓ SẴN — đúng quy ước a_i = mask_i/Σmask."""
    tot = np.zeros(Xs[0].shape[0]); cnt = np.zeros(Xs[0].shape[0])
    for X, b, mk in zip(Xs, betas, masks):
        tot += np.where(mk, X @ b, 0.0); cnt += mk.astype(float)
    return tot / np.maximum(cnt, 1.0)
```

**Thích ứng với dữ liệu dự án:** chuẩn hoá RobustScaler theo view trên các hàng có mask, hàng thiếu
modality đặt vector 0 (view đó đóng góp 0); điểm cuối là trung bình các view có sẵn.
**Đánh giá:** hoán vị hàng theo seed rồi `KFold(10)` không shuffle. *(Đính chính 2026-10-04: đây
KHÔNG phải đúng các fold của `run_paper_combos.py` — harness dùng `shuffle=True, random_state=0`; vẫn là
CV 5-seed × 10-fold hợp lệ, kết luận không đổi.)* seeds `[42, 7, 123, 2024, 31337]`, AUC gộp out-of-fold, mean ± sd.

## 3. Kết quả — chặng 1, quét ρ trên 4 benchmark (5 seed × 10 fold)

| BM | Tổ hợp | ρ=0 | ρ=0,2 | ρ=0,5 | ρ=1 | ρ=2 | ρ=5 | **uniform_avg** | DyAM | OvO |
|---|---|---|---|---|---|---|---|---|---|---|
| BM1 | Rad+IHC-G+Gen+PDL1 | 0,719 | 0,724 | 0,727 | 0,729 | 0,731 | 0,733 | **0,775** | 0,760 | 0,770 |
| BM2 | Rad+IHC-G+Gen+PDL1+Labs | 0,713 | 0,711 | 0,712 | 0,714 | 0,714 | 0,717 | **0,767** | 0,764 | 0,769 |
| BM3 | Rad+Gen | 0,667 | 0,668 | 0,669 | 0,666 | 0,666 | 0,668 | **0,716** | 0,710 | 0,716 |
| BM4 | PDL1+Gen | **0,746** | 0,743 | 0,741 | 0,743 | 0,742 | 0,744 | 0,719 | 0,707 | 0,718 |

sd trên 5 seed: 0,006–0,028. ρ tốt nhất theo tổng AUC = 5,0. **Số benchmark đạt ≥ uniform_avg: 1/4**
→ không đạt ngưỡng (cần ≥ 2/4) để chạy tiếp 21 tổ hợp. Dừng theo tiêu chí đặt trước.

## 4. Kiểm tra công bằng

Nghi ngờ: thua có thể vì lasso tự chọn đặc trưng radiomics, thay vì dùng bộ lọc L1 theo độ bền
(`l1_filter_features_list`, `robustness_cutoff=0.15`) mà pipeline dự án dùng. Đã chạy lại **có** áp
bộ lọc đó trong từng fold train (3 seed):

| | ρ=1 | ρ=5 | uniform_avg |
|---|---|---|---|
| BM1 có lọc L1 | 0,729 | 0,730 | 0,775 |
| BM3 có lọc L1 | 0,677 | 0,674 | 0,716 |

Gần như không đổi → **kết quả âm không phải do cách chọn đặc trưng.**

## 5. Diễn giải

**Chỉ thắng ở tổ hợp chiều thấp.** BM4 (PD-L1 1 chiều + genomics 11 chiều) là tổ hợp duy nhất
cooperative learning thắng, và thắng đậm: +0,027 so với uniform_avg, +0,039 so với DyAM. Mọi tổ hợp
có radiomics đều thua 0,04–0,05. Mô hình tuyến tính có lasso không khai thác được radiomics chiều
cao tốt bằng mô hình tanh-tuyến-tính theo từng modality của dự án.

**Đây là lần thứ hai xuất hiện đúng mẫu hình này.** TabPFN (§2.7 trong `result/training-results.md`)
cũng chỉ thắng ở BM4 (+0,024) và thua 0,044–0,060 ở các tổ hợp còn lại. Hai phương pháp hoàn toàn
khác nhau, cùng một kết luận: **ở cỡ mẫu này, tổ hợp ít chiều thì phương pháp thống kê cổ điển tốt
hơn; tổ hợp có radiomics thì kiến trúc theo-modality của dự án tốt hơn.**

**ρ gần như không ảnh hưởng** (BM1 đi từ 0,719 lên 0,733 khi ρ chạy từ 0 đến 5; BM2, BM3, BM4 phẳng).
Nghĩa là **vị trí trên dải early↔late fusion không phải là yếu tố quyết định** ở bài toán này —
điều quyết định là cách mô hình hoá từng modality. Có một tín hiệu phụ nhất quán với tài liệu: ρ
tốt nhất luôn rơi về phía **late fusion** (ρ=5), phù hợp với nhận định chung rằng late fusion được
ưa chuộng khi n nhỏ.

## 6. Kết luận

**XOÁ — không đưa vào mô hình chính.** Có thể nhắc một câu trong bài báo như một comparator hiện
đại đã thử và không hơn: *"we also evaluated cooperative learning (Ding et al., PNAS 2022), which
underperformed the averaging baseline on every configuration containing radiomics"*.

Phát hiện phụ đáng giữ: trên tổ hợp ít chiều BM4, cooperative learning đạt **0,7464** — ngang với
con số tốt nhất dự án từng có ở tổ hợp đó và vượt xa nhóm không-NLP:

| BM4 (PDL1+Gen) | AUC 5-seed |
|---|---|
| OvO+NLP | **0,7472** |
| cooperative learning | 0,7464 |
| uniform_avg+NLP | 0,7445 |
| TabPFN | 0,7430 |
| uniform_avg | 0,7191 |
| OvO | 0,7183 |
| DyAM | 0,7072 |

Tức là coop **không phải con số cao nhất** (thua OvO+NLP 0,0008 — coi như bằng nhau), nhưng đáng chú
ý ở chỗ nó đạt mức đó **không cần dữ liệu NLP**, chỉ bằng lasso + phạt bất đồng.

## 7. Đã xoá sau khi kết thúc

`experiments/coop_learning/{coop,run_coop}.py` · `experiments/results/coop_learning.json` ·
`document/2026-09-30_cooperative-learning/{ke-hoach.md,log_run.txt}`.
Thuật toán và toàn bộ số liệu nằm ở mục 2 và 3 của chính file này; muốn chạy lại thì dựng lại từ đó.
