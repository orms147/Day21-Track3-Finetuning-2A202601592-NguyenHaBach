# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Hà Bách  **Mã HV**: 2A202601592  **Ngày**: 2026-08-21
**Tier**: `LAPTOP`  **Base model**: `Qwen/Qwen3.5-2B`  **GPU thực tế**: `Nvidia RTX 4070`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | assistant-only |
| Epochs / max_steps | 2 / 58 |

**Template có giữ khối `<think>` không?** có — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào? Template Qwen3.5 giữ khối suy luận (verdict: reasoning preserved — safe to train on traces).

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.3936 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```json
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.5778 | 0.000 | 1842.7 |
| (b) base + optimized prompt | 0.585 | 0.5778 | 1.000 | 542.4 |
| (c) LoRA fine-tune | 0.995 | 0.0667 | 1.000 | 778.1 |

**(b) có thật sự mạnh hơn (a) không?** có — nếu không, bạn đã cải thiện (b) thế nào?
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao? Không sửa `OPTIMIZED_PROMPT` (SHA: 719e74d3b6232053). Mức điểm 58.5% của prompt tối ưu (b) đã đủ mạnh làm benchmark.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 16.81M (16819200) | 1e-4 | 0.2939 | 0.995 | 591.4 | 6.70 |
| `attn_only` | q,v | 322 | 16.81M (16816128) | 1e-4 | 0.3145 | 0.995 | 675.4 | 6.71 |
| `wrong_lr` | text-linear | 16 | 16.81M (16819200) | 1e-5 | 1.1175 | 0.470 | 751.3 | 6.70 |
| `qlora` | text-linear | 16 | Bỏ qua | | | | | |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**
Mô hình `attn_only` có số lượng tham số huấn luyện xấp xỉ `correct` nhưng phải đánh đổi bằng rank rất cao (r=322 so với r=16). Về điểm số target (0.995), `attn_only` đạt kết quả **hoà** với mô hình chuẩn trên bài toán JSON triage tương đối đơn giản này. Điều này chứng tỏ vị trí gắn adapter quan trọng hơn việc mù quáng tăng rank; nếu chỉ gắn vào `q,v` (ít trọng số hơn `text-linear`), ta phải dùng một ma trận rank r lớn hơn rất nhiều mới bù đắp được dung lượng thông tin so với việc gắn LoRA vào toàn bộ các lớp `text-linear` với rank thấp (r=16).

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Cấu hình `wrong_lr` dùng learning rate theo thang full fine-tune (1e-5) thay vì 1e-4 như LoRA chuẩn. Hậu quả là đường train loss giảm rất chậm và vẫn ở mức cực cao (1.1175 so với 0.2939 của cấu hình chuẩn). Nếu không biết đang dùng sai LR, chúng ta có thể dễ dàng đổ lỗi sai lệch (kết luận sai) rằng cấu hình mô hình hiện tại quá nhỏ, không đủ năng lực học, hoặc nghĩ rằng LoRA không hiệu quả bằng full fine-tune.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**
Vì chạy trên hệ điều hành Windows nên thư viện `bitsandbytes` không hỗ trợ đầy đủ 4-bit, kết quả là bước thử nghiệm `qlora` buộc phải được bỏ qua. Tuy nhiên, dựa theo lý thuyết và khuyến nghị từ nhà cung cấp, việc áp dụng QLoRA lên Qwen3.5-2B tuy giảm VRAM đáng kể nhưng sẽ gây suy giảm năng lực biểu diễn trọng số vì Qwen vốn đã được tối ưu. Điều này hoàn toàn ủng hộ triết lý "đừng dùng QLoRA trên Qwen3.5" trừ khi thật sự cạn kiệt tài nguyên bộ nhớ.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.410` · `regression Δ = -0.511` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)
Mô hình fine-tune bị đánh giá là **FAILED** vì dính lỗi Catastrophic Forgetting (quên thảm hoạ). Mặc dù năng lực giải quyết tác vụ mục tiêu (target) đã đạt xuất sắc (99.5%, tăng 41% so với baseline), nhưng năng lực trả lời các câu hỏi phổ thông (regression) lại sụt giảm thảm hại tới 51.1% (từ 0.5778 tụt thẳng xuống 0.0667), vượt xa mức dung sai cho phép là 0.02. Điều này cho thấy với lượng dữ liệu chuyên biệt hẹp và cấu hình huấn luyện LoRA áp dụng lên hầu hết lớp linear, mô hình đã "học vẹt" cách tạo JSON và đánh mất vốn hiểu biết tổng quát trước đó. Giải pháp bắt buộc trước khi triển khai là phải trộn thêm khoảng 1-5% dữ liệu hồi quy (replay data) để duy trì khả năng gốc.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper khô | van_chuyen | | van_chuyen | ✅ FT thắng |
| 2 | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm n | san_pham_loi | | san_pham_loi | ✅ FT thắng |
| 3 | Chào shop, mình đặt nồi chiên không dầu mã đơn VN558606. Giao hàng chậ | van_chuyen | | van_chuyen, urgency: cao | ❌ **FT thua** (sai 1 trường urgency, đạt 0.75 điểm) |
| 4 | *(Vì tập eval đạt target 0.995 (sai 1/200 trường) nên không thể lấy đủ 2 ca thua tuyệt đối)* | | | | ❌ **FT thua** |
| 5 | | | | | |

Có mẫu chung nào ở các ca FT thua không?
Điểm target tổng thể đạt mức 99.5%, nghĩa là trên tổng số 50 câu (x 4 trường JSON = 200 trường dự đoán), model chỉ trả lời sai đúng 1 trường duy nhất (trường urgency bị dự đoán sai ở ca số 23), đạt điểm 0.75 cho mẫu đó. Điều này cho thấy model có khả năng vấp lỗi ở việc ước lượng "độ khẩn cấp" (urgency) của khách khi từ ngữ mô tả trong ticket nằm ở ranh giới mờ giữa "thấp", "trung bình" và "cao".

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn
bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?
Mặc dù bản thân mô hình đạt chỉ số tuyệt vời trên tập đánh giá chuyên biệt (accuracy 99.5%, vượt mốc 58.5% của prompt tĩnh một cách dễ dàng), nhưng tôi **KHÔNG NÊN deploy** bản fine-tune này ra môi trường sản xuất. Nguyên nhân cốt lõi là lỗi quên thảm hoạ (Catastrophic Forgetting) đã khiến mô hình mất đi năng lực trả lời thông thường (giảm tới 51.1% so với dung sai cho phép). Việc triển khai một mô hình mất não như vậy sẽ là thảm hoạ nếu người dùng vô tình nhập các chuỗi khác thường. Đòn bẩy thật sự trong quá trình fine-tune này chính là **learning rate** (quyết định việc model học được hay không) và **dữ liệu huấn luyện chuyên biệt đi kèm replay data** (để giữ gìn khả năng tổng quát).

**Ba điều tôi học được** (cụ thể, không generic):
1. Loss mask là tối quan trọng: Chỉ nên tính loss trên phần trả lời (assistant). Nếu che sai và tính loss cả câu hỏi thì mô hình sẽ vô tình học cách lặp lại luôn cả prompt đầu vào.
2. Learning rate cho LoRA cần cao hơn khoảng 10 lần so với full fine-tune. Nếu dùng LR quá nhỏ, quá trình huấn luyện sẽ trở nên vô nghĩa (như mô hình wrong_lr chứng minh).
3. Không thể đánh giá năng lực tác vụ dựa trên Training Loss. Phải sử dụng bộ đo độc lập (cổng hồi quy regression, chỉ số định dạng format và target) để đánh giá đúng chất lượng triển khai.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ tạo thêm một tập dữ liệu "Replay Data" có chứa các câu hỏi thông thường chiếm khoảng 5% tập train để tiến hành trộn dữ liệu và train lại. Mục tiêu là cứu vớt điểm số Regression mà không làm tụt điểm số Target, từ đó qua được ải PASSED của lab.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
