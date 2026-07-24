# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> 
> Khi temperature thấp như 0.0, phản hồi trực tiếp và đúng; lên 0.7 thì câu trả lời tự nhiên hơn. Ở 1.2 nội dung bắt đầu sáng tạo hơn, có thể thêm chi tiết bất ngờ; đến khoảng 1.8 phản hồi lan man, không nhất quán.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
>
> Với trợ lý soạn hợp đồng pháp lý, tôi sẽ đặt temperature là 0 để ưu tiên chính xác, tránh thông tin bịa. Với trợ lý viết slogan quảng cáo, tôi sẽ đặt khoảng 1 để có nhiều ý tưởng mới, cách diễn đạt đa dạng và giàu cảm xúc hơn. Khác biệt chính là tác vụ pháp lý cần đảm bảo chính xác tuyệt đối, còn slogan thì cần có nhiều phương án sáng tạo khác lạ.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> 
> Workload có 20.000 * 2 * 500 = 20.000.000 token đầu ra mỗi ngày. Theo bảng giá, gpt-4o tốn khoảng 20.000 * 0,010 = 200 USD/ngày cho output, còn gpt-4o-mini tốn khoảng 20.000 * 0,0006 = 12 USD/ngày; model lớn đắt hơn khoảng 188 USD/ngày, hơn 16 lần. Model lớn xứng đáng khi cần giải bài toán khó, tư vấn chất lượng cao hoặc các câu hỏi cần độ chính xác cao; model nhỏ phù hợp cho FAQ, phân loại, tóm tắt ngắn hoặc chatbot hỗ trợ cơ bản có thể kiểm soát bằng prompt.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> 
> Với persona nhà thơ, câu trả lời thường dễ hiểu ngắn gọn hơn, dùng hình ảnh, ít thuật ngữ và có thể ngắn gọn, cảm xúc hơn. Với persona kỹ sư senior, phản hồi có xu hướng định nghĩa rõ, nêu khái niệm dữ liệu và có thể đưa ví dụ. Điều này cho thấy system prompt điều khiển được vai trò, giọng văn, mức độ kỹ thuật, cấu trúc câu trả lời và loại ví dụ được ưu tiên. Tuy nhiên nó không đảm bảo tuyệt đối nội dung đúng,
### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> 
> Với tiếng Việt, số token từ `tiktoken` thường cao hơn ước lượng thô `số từ / 0.75` vì dấu tiếng Việt, từ ghép và cách mã hóa có thể làm một từ bị tách thành nhiều token. Nếu dùng ước lượng thô để dự toán ngân sách API tiếng Việt, tôi sẽ dự toán thiếu vì token thật thường nhiều hơn số suy ra từ số từ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> 
> Chatbot văn bản và trợ lý giọng nói hưởng lợi nhiều nhất từ streaming vì người dùng thấy phản hồi bắt đầu ngay, giảm cảm giác chờ và có thể đọc/nghe dần khi model còn đang sinh tiếp. Với trợ lý giọng nói, streaming giúp bắt đầu tổng hợp hoặc phát âm sớm hơn, tạo cảm giác hội thoại tự nhiên. Pipeline dịch tài liệu chạy ngầm ban đêm không cần streaming vì chỉ quan tâm kết quả cuối cùng, log tiến độ và độ tin cậy cao

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> 
> Exponential backoff giúp giảm áp lực lên API khi quá tải vì mỗi lần retry sau sẽ chờ lâu hơn, tránh việc nhiều client cùng gửi lại liên tục. Nếu dùng delay cố định, các client có thể tiếp tục request vào server cùng lúc và kéo dài tình trạng nghẽn. Jitter thêm một độ trễ ngẫu nhiên để tránh hiện tượng nhiều client retry đồng thời, giúp phân tán request đều hơn theo thời gian.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> 
> System prompt tôi dùng: “Bạn là trợ giảng của khóa học AI, trả lời ngắn gọn bằng tiếng Việt, giải thích theo từng bước khi tôi có câu hỏi kỹ thuật, và nói rõ khi bạn không chắc chắn.” Nếu xóa “trả lời ngắn gọn bằng tiếng Việt”, trợ lý có thể trả lời dài hơn hoặc chuyển sang tiếng Anh. Nếu xóa “nói rõ khi bạn không chắc chắn”, trợ lý dễ đưa ra câu trả lời nghe tự tin dù thông tin chưa được kiểm chứng.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> 
> Ví dụ người dùng ở đầu phiên nói “tôi đang làm app hỗ trợ cho khách doanh nghiệp, ưu tiên bảo mật và audit log”, sau đó trao đổi thêm hơn 4 lượt về UI, dữ liệu và lỗi API; đến lượt sau người dùng hỏi “vậy nên chọn cách lưu lịch sử nào?”, trợ lý có thể quên ràng buộc bảo mật/audit log ban đầu và trả lời quá chung chung. Cách khắc phục là giữ một bản tóm tắt ngắn của các lượt cũ chứa yêu cầu quan trọng, rồi gửi kèm summary đó cùng 4 lượt gần nhất. Có thể kết hợp thêm cơ chế chọn lọc: giữ lại các facts/ràng buộc quan trọng.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
