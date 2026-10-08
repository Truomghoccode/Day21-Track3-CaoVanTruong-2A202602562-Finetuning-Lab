# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Với cấu hình attn_only rank 283 có loss thấp nhưng all-linear rank 16 nhưng kết quả trên validation lại tốt hơn so với attn_only rank 283. Điều này cho thấy rank và loss thấp hơn không đồng nghĩa với chất lượng tốt hơn 

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Bước mất thời gian nhất là bước sinh câu trả lời và đánh giá model trên 50 mẫu. Nó không phải chỗ mà tôi dự đoán , bước mà tôi dự đoán mất thời gian nhất là bước train model

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này tôi nghĩa rằng rank LoRA càng to thì mô hình học càng tốt và training loss giảm sâu là mô hình fine-tune sẽ học và cho kết quả tốt hơn.Tuy nhiên sau bài lab này tôi hiểu rằng rank chỉ giúp mô hình học các đặc trưng nhanh và tốt hơn .Và training loss thấp không đồng nghĩa với chất lượng tốt hơn nó chỉ thể hiện nó học các dữ liệu từ tập train tốt thôi.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để tóm tắt giải thích bài Lab. Tuy nhiên AI assistant chưa tính đến sự khác biệt về ký tự xuống dòng giữa Linux và Windows. Git đã chuyển một số file từ LF sang CRLF, khiến mã kiểm tra trong checksums.json không còn khớp. Tôi phải chuẩn hóa file về LF để khắc phục.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Nếu phải làm cho khách hàng thật tôi sẽ xây dựng một bộ đánh giá cố định, gồm cả nhiệm vụ nghiệp vụ và các câu hỏi kiểm tra kiến thức chung. Sau đó, tôi sẽ thử base model với prompt và ví dụ few-shot được tối ưu. 
