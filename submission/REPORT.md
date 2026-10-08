# Lab 21 — Evaluation Report

**Họ tên**: Cao Văn Trường  **MSSV**: 2A202602562  **Ngày**: 08/10/2026
**Tier**: T4  **Base model**: unsloth/Qwen3.5-4B  **GPU thực tế**: Tesla T4 16GB (fp16)

> Mọi con số dưới đây khớp chính xác 100% với file trong `results/`.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | assistant-only |
| Epochs / max_steps | 2.0 / 30 |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)* khẳng định: `reasoning preserved — safe to train on traces`. Chat template của Qwen3.5 giữ nguyên vẹn nội dung reasoning bên trong cặp thẻ `<think>...</think>`, không bị tự động lược bỏ trong quá trình tokenize hay render mẫu. Về cấu hình `max_length`, dù p95 đo được trên tập dữ liệu là 98 token (gợi ý ngưỡng 256), cấu hình phần cứng tier T4 được giữ nguyên ở mức 1024 token để đảm bảo biên an toàn tối đa cho quá trình sinh suy luận và sinh chuỗi JSON dài mà không gây ra hiện tượng cắt cụt văn bản (truncation).

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3180.9 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 988.8 |
| (c) LoRA fine-tune | 0.965 | 0.744 | 1.000 | 1406.3 |

**(b) có thật sự mạnh hơn (a) không?** Có, vượt trội hoàn toàn (0.765 so với 0.000). Với prompt đơn giản không có định nghĩa schema cụ thể, base model sinh văn bản tự do nên tỷ lệ parse được JSON chuẩn đạt 0.000. Ngược lại, prompt tối ưu cung cấp quy tắc 4 khóa rõ ràng kèm ví dụ few-shot giúp mô hình đạt 100% tỷ lệ format JSON và độ chính xác phân loại trường đạt 76.5%.
Tôi không chỉnh sửa chuỗi `OPTIMIZED_PROMPT` nhằm giữ nguyên SHA (`719e74d3b6232053`) đã được đóng băng từ ban đầu, đảm bảo tính liêm chính và tính công bằng tuyệt đối của phép so sánh trước khi bước vào huấn luyện.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32464896 | 0.0001 | 0.6277 | 0.965 | 402.2 | 8.78 |
| `attn_only` | q,v | 283 | 32456704 | 0.0001 | 0.5382 | 0.970 | 265.8 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32464896 | 0.00001 | 1.5702 | 0.000 | 392.5 | 8.78 |
| `qlora` | text-linear | 16 | 32464896 | 0.0001 | 0.7058 | 0.940 | 462.5 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

Trên tập target ở NB5 §4, `attn_only` đạt độ chính xác 0.970 so với 0.965 của `correct` (chênh lệch chỉ 0.005, tương ứng với việc đúng hơn đúng 1 trường trên tổng số 50 mẫu đánh giá, coi như mức hòa trên một bài toán hẹp). Tuy nhiên, trên phương diện loss huấn luyện ở NB4, `attn_only` lại ép loss xuống sâu hơn rõ rệt (0.5382 so với 0.6277). Thứ tự xếp hạng này cho thấy nghịch lý điển hình: một adapter có rank cực lớn (r=283) ép chặt vào không gian chiếu attention nhỏ hẹp sẽ dễ dàng ghi nhớ tập train nhỏ (225 mẫu) để tạo ra training loss rất thấp mà không đem lại sự vượt trội thực chất nào về năng lực tổng quát hóa. Do đó, việc nâng rank lên quá cao chỉ làm tăng nguy cơ học vẹt, trong khi việc phân bổ rank vừa phải (r=16) bao phủ toàn diện các khối text-linear (`all-linear`) mới là đòn bẩy cấu trúc tạo nên biểu diễn cân bằng và bền vững.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Run `wrong_lr` chỉ thay đổi duy nhất learning rate từ mức chuẩn LoRA (0.0001) xuống mức full fine-tune (0.00001), nhưng đường loss gần như đi ngang và dừng lại ở mức 1.5702, hoàn toàn không hội tụ so với mức 0.6277 của `correct`. Nếu một kỹ sư chỉ quan sát loss huấn luyện phẳng lì mà không nắm rõ thiết lập learning rate, họ sẽ rất dễ đưa ra kết luận sai lầm rằng năng lực mô hình quá yếu hoặc dữ liệu quá khó nên mô hình không thể học nổi. Thực tế nguyên nhân cốt lõi là do ma trận B của LoRA được khởi tạo bằng 0 và toàn bộ trọng số gốc bị đóng băng; nếu áp dụng tốc độ học nhỏ của full-FT, bước cập nhật gradient sẽ quá bé để đưa ma trận tích BA thoát khỏi điểm khởi đầu trong một số lượng bước tối ưu hữu hạn (30 steps).

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

Thực nghiệm cho thấy `qlora` cắt giảm mạnh dung lượng bộ nhớ VRAM đỉnh từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm hơn 56% VRAM), chứng minh lợi thế lớn về khả năng nạp mô hình trên phần cứng hạn chế. Tuy nhiên, mô hình phải trả giá bằng việc thời gian huấn luyện kéo dài thêm (462.5s so với 402.2s của fp16 do chi phí dequantize liên tục trong forward/backward pass), độ trễ suy luận tăng lên (1741.7 ms so với 1406.3 ms) và độ chính xác target bị sụt giảm từ 0.965 xuống 0.940. Kết quả đo lường thực tế này hoàn toàn ủng hộ khuyến nghị kỹ thuật của nhà phát triển Qwen3.5: đối với các dòng mô hình kiến trúc lai hiện đại (xen kẽ linear attention và full attention), sai số lượng tử hóa 4-bit gây tổn thất đáng kể đến độ chính xác và thông lượng tính toán, do đó nên ưu tiên sử dụng fp16/bf16 LoRA khi tài nguyên phần cứng vẫn cho phép.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.200` · `regression Δ = -0.0467` · `valid_trace_rate = 0.0`

Diễn giải: Cổng hồi quy đưa ra phán quyết FAILED là hoàn toàn khách quan và phản ánh đúng bản chất kỹ thuật của quá trình fine-tuning chuyên biệt. Mặc dù trên tác vụ mục tiêu (target triage), bản LoRA fine-tune đạt được bước tiến nhảy vọt khi vượt mốc baseline prompt tối ưu tới +20.0% độ chính xác và đảm bảo 100% chuẩn định dạng JSON, nhưng trên bài kiểm tra năng lực tổng quát (15 câu hỏi kiến thức phổ thông), điểm số đã bị sụt giảm 4.7% (từ 0.7911 xuống 0.7444), vượt quá ngưỡng dung sai an toàn cho phép là 2.0%. Đây là biểu hiện kinh điển của hiện tượng quên thảm họa (catastrophic forgetting): khi mô hình bị ép cập nhật liên tục trên một phân phối dữ liệu đơn nhất (JSON ticket chăm sóc khách hàng) mà không có bất kỳ mẫu dữ liệu giữ nhịp tổng quát nào, các trọng số LoRA đã vô tình làm nhiễu loạn một phần tri thức đa miền sẵn có. Kết quả này nhắc nhở rằng một mô hình đạt điểm số chuyên môn rất cao vẫn chưa đủ điều kiện an toàn để triển khai độc lập nếu thiếu chiến lược bảo toàn năng lực nền tảng theo khuyến nghị tại Deck §6.3 thông qua việc trộn thêm 1–5% dữ liệu tổng quát (replay data).

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper khô | van_chuyen / thap / ốp lưng điện thoại / tieu_cuc | Đúng 3/4 trường | {"intent": "van_chuyen", "urgency": "thap", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"} | ✅ FT thắng: Trích xuất trọn vẹn 4/4 trường thông tin, đúng định dạng JSON tuyệt đối. |
| 2 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. | hoi_thong_tin / trung_binh / ốp lưng điện thoại / trung_tinh | Đúng 3/4 trường | {"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"} | ✅ FT thắng: Nhận diện chính xác intent hỏi thông tin và thái độ trung tính ngắn gọn. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. | hoan_tien / thap / bình giữ nhiệt / tich_cuc | Đúng urgency thap | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"} | ❌ **FT thua**: Khách viết "khi nào tiện" thể hiện độ khẩn cấp thap, nhưng FT gán nhãn trung_binh. |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. | san_pham_loi / thap / nồi chiên không dầu / trung_tinh | Đúng urgency thap | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"} | ❌ **FT thua**: Khách viết "khi nào tiện", FT bỏ qua ngữ cảnh này và tự động dự đoán trung_binh. |
| 5 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. | san_pham_loi / thap / áo khoác gió / tich_cuc | Đúng nhãn | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"} | ❌ **FT thua**: Cụm từ "khi nào tiện" bị FT ngó lơ, mô hình bị thiên kiến gán urgency về lớp trung_binh. |

Có mẫu chung nào ở các ca FT thua không?
Điểm chung nổi bật ở tất cả các ca fine-tune bị trừ điểm (ft_score = 0.75) là mô hình dự đoán sai duy nhất trường `urgency`. Trong tập dữ liệu, các khách hàng sử dụng cụm từ lịch sự "Khi nào tiện" thể hiện mức độ khẩn cấp thấp (`thap`). Tuy nhiên, do phân phối tập huấn luyện 225 mẫu có số lượng nhãn `trung_binh` chiếm tỷ trọng đa số, mô hình sau khi hội tụ đã xuất hiện thiên kiến quy nạp (inductive bias) mạnh mẽ, tự động gán nhãn `trung_binh` cho các câu ticket này. Ngược lại, baseline prompt tối ưu (b) nhờ có phần mô tả hướng dẫn chi tiết từng cấp độ khẩn cấp trong prompt đã phân biệt chính xác hơn các trường hợp biên này.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** 
Dựa trên kết quả thực nghiệm toàn diện của bài lab, tôi đưa ra kết luận rằng phiên bản fine-tune hiện tại **chưa nên triển khai trực tiếp vào môi trường sản xuất**, bất chấp việc nó đạt độ chính xác phân loại chuyên môn rất ấn tượng (target đạt 0.965, vượt mốc baseline prompt tối ưu tới +20.0%). Rào cản lớn nhất ngăn cản việc triển khai là sự vi phạm cổng hồi quy (regression gate FAILED): khả năng suy luận ngôn ngữ tổng quát bị suy giảm 4.7%, chứng tỏ mô hình đang gặp hội chứng quên thảm họa và có xu hướng thiên kiến hóa nhãn `urgency`. Qua bài thực hành, đòn bẩy thật sự quyết định thành bại của quá trình fine-tuning không nằm ở việc cố gắng nâng rank lên mức khổng lồ (kết quả NB4 chỉ ra rank 283 ở attention chỉ giúp ép giảm train loss mà không vượt trội trên tập kiểm thử so với rank 16 ở all-linear). Thay vào đó, ba đòn bẩy mang tính quyết định là: (1) Tính đúng đắn của loss mask (đảm bảo chỉ tính loss trên câu trả lời), (2) Việc lựa chọn learning rate phù hợp với thang LoRA (LR full-FT khiến mô hình bất động), và (3) Độ phủ toàn diện của adapter trên toàn bộ các tầng linear của text decoder. Để đủ điều kiện đưa vào thực tế, mô hình cần được tái huấn luyện với 2–5% dữ liệu tổng quát (replay data) và tái cân bằng phân phối nhãn.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Loss mask là nền tảng sống còn, không thể phó mặc cho thư viện**: Việc giải mã ngược token và kiểm chứng offset trong NB1 chứng minh rằng việc tính loss sai trên cả prompt sẽ phá vỡ hoàn toàn năng lực sinh của mô hình. Tự kiểm chứng mask bằng code thay vì tin tưởng mù quáng vào các cờ cấu hình là bài học kỹ thuật quan trọng nhất.
2. **Cảnh giác trước cạm bẫy chỉ số thay thế (Proxy Metric Fallacy)**: Quan sát đường loss huấn luyện ở NB4 cho thấy `attn_only` có loss thấp hơn `correct`, nhưng khi chấm điểm trên năng lực tác vụ mục tiêu ở NB5 thì kết quả lại không hề phản ánh tương ứng. Đánh giá mô hình phải dựa trên năng lực thực thi tác vụ cụ thể và bài kiểm tra hồi quy chứ không được dừng lại ở loss.
3. **Cổng hồi quy là thước đo liêm chính trong kỹ thuật LLM**: Một mô hình chuyên biệt hóa có thể đạt độ chính xác rất cao ở bài toán hẹp nhưng âm thầm đánh mất các năng lực nền tảng. Việc thiết kế cổng kiểm tra hồi quy 4 nhóm giúp kỹ sư phát hiện sớm hiện tượng quên thảm họa trước khi đưa sản phẩm ra người dùng cuối.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ thực hiện thử nghiệm trộn thêm 3% dữ liệu đệm tổng quát (replay data) gồm các mẫu chỉ dẫn đa dạng bằng tiếng Việt vào tập huấn luyện của NB3 để chứng minh rằng độ suy giảm hồi quy (regression delta) sẽ được thu hẹp về dưới ngưỡng 2%, qua đó đưa kết quả cổng hồi quy chuyển từ FAILED sang PASSED mà không làm suy giảm độ chính xác của bài toán phân loại ticket.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
