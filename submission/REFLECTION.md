# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Ngạc nhiên nhất là hiện tượng "Quên thảm hoạ" (Catastrophic Forgetting) diễn ra cực kỳ khốc liệt dù chỉ cập nhật trọng số bằng LoRA trên một tập dữ liệu siêu nhỏ (250 mẫu). Chỉ sau 2 epoch, mô hình tuy đạt 99.5% accuracy cho task mục tiêu nhưng điểm kiến thức phổ thông (regression) rớt thảm hại từ ~58% xuống chỉ còn ~6.6%.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Mất thời gian nhất lại nằm ở khâu setup môi trường (đặc biệt là việc lệnh `make` trong thư mục gốc bị lỗi cú pháp đường dẫn trên Windows, và sự cố xung đột thư viện giữa `torch 2.6` và `torchao`). Ban đầu, tôi dự đoán việc mất thời gian nhất sẽ là chờ GPU chạy huấn luyện NB3 và NB4.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước đây tôi từng tin rằng cứ tăng `rank` của LoRA lên cao (ví dụ r=64 hoặc r=322) thì mô hình sẽ tự động học tốt hơn. Giờ thì tôi hiểu ra rằng "vị trí gắn adapter" quan trọng hơn việc mù quáng tăng rank; gắn LoRA vào toàn bộ các lớp `text-linear` với rank thấp (r=16) vẫn hiệu quả và an toàn hơn gắn `q,v` với rank khổng lồ.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi sử dụng AI assistant để thực hiện toàn bộ các thao tác gõ lệnh (từ cài package, sửa `.env`, đến chạy các file notebook script) và tổng hợp số liệu JSON để điền vào Report. AI đã "hơi cứng nhắc" khi cố chạy `make setup` (chỉ phù hợp cho Linux/Mac) trên Windows PowerShell khiến quá trình bị kẹt lại lúc đầu, sau đó mới tự sửa lỗi bằng cách dùng trực tiếp `pip`.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Thay vì nhảy ngay vào viết code huấn luyện, bước đầu tiên tôi làm sẽ là ngồi tinh chỉnh prompt (prompt engineering) để đo mốc baseline một cách tử tế, sau đó tự tay chuẩn bị một tập dữ liệu đánh giá bao gồm cả các "Replay Data" (kiến thức nền) để đảm bảo mô hình không bị "ngốc đi" sau khi fine-tune xong.
