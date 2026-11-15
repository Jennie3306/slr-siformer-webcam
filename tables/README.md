# Bảng biên độ chuyển động cho bước hiệu chỉnh tư thế tay

Có hai bộ bảng:

- `full/`: đủ các dòng mô tả bên dưới (4 dòng dạng–khép, 15 dòng gập–duỗi).
- `dip_only/` (mặc định): chỉ giữ 4 dòng gập–duỗi khớp DIP của 4 ngón dài, bỏ hết dạng–khép.

Lý do có bộ `dip_only`: thống kê ngày 14/11 trên tập train MediaPipe (35.640 bàn tay) cho thấy bộ `full` sửa ít nhất một khớp ở 78% số bàn tay. Dữ liệu của tác giả cũng cho tỉ lệ tương tự (85%).

- Các dòng dạng–khép bắt 26–32% số bàn tay.
- Ngón cái MCP bắt 38%.
- MCP và PIP của 4 ngón dài bắt 7–16%.

Nguyên nhân: biên độ trong Bảng 1 là giới hạn của khớp thật trong không gian 3D, còn góc đo trên ảnh 2D bị chiếu. Khi nắm tay hoặc chĩa ngón về phía camera, góc trên ảnh vượt giới hạn dù tư thế hoàn toàn bình thường. Chỉ 4 dòng DIP bắt dưới 3% số bàn tay, đúng với kỳ vọng "lỗi hiếm" của MediaPipe.

Bộ nào dùng cho kết quả chính sẽ được chọn theo độ chính xác trên tập **val**: so sánh không hiệu chỉnh, `dip_only` và `full`.

Code hiệu chỉnh của tác giả Siformer (`external/Siformer/rectifications/`) cần hai file bảng này, nhưng repo không công bố.
Đồ án tự xây lại. Biên độ góc lấy từ Bảng 1 của bài. Bài không ghi mỗi góc đo giữa 3 điểm nào, nên phần đó là do đồ án tự chọn, theo thứ tự 21 điểm tay của MediaPipe (0 cổ tay, 1–4 ngón cái, 5–8 trỏ, 9–12 giữa, 13–16 áp út, 17–20 út).

## fe_active_motion.csv: gập–duỗi

`paired_joint_index = [refer, check, extra]`. Theo code của tác giả, góc được đo là góc giữa đốt `extra → check` và đốt `check → refer`, tức độ gập tại khớp `check`. Nếu góc vượt biên độ, đoạn `check → refer` được xoay về lại một phần α, và các điểm trong `associated_joint_index` dịch theo.

| Khớp | Bộ 3 điểm (refer, check, extra) | Biên độ trong bài | Ghi trong bảng |
|---|---|---|---|
| Ngón cái CMC | 2, 1, 0 | −20 … 45 | [−45, 20] |
| Ngón cái MCP | 3, 2, 1 | 0 … 80 | [−80, 0] |
| Ngón cái IP | 4, 3, 2 | −30 … 90 | [−90, 30] |
| MCP 4 ngón dài | b+1, b, 0 | −40 … 90 | [−40, 90] |
| PIP | b+2, b+1, b | 0 … 130 | [0, 130] |
| DIP | b+3, b+2, b+1 | −30 … 90 | [−30, 90] |

- **Ngón cái:** code của tác giả dùng biên độ có hướng. Trục y hướng lên; khi hướng gập là "clockwise", ngón cái gập cho góc âm. Vì vậy bảng ghi biên độ đổi dấu. Đã kiểm tra trên bàn tay giả, cả lòng bàn tay lẫn mu bàn tay: gập tự nhiên không bị sửa, duỗi ngược thì bị sửa (`tests/selftest.py`).
- **4 ngón dài:** code chỉ dùng `[-max, max]`, nên dấu không quan trọng. Trên ảnh 2D, góc này chỉ bắt được trường hợp gập quá mức.

## aa_active_motion.csv: dạng–khép

`paired_joint_index = [check, refer, centre]`: góc giữa `centre → check` và `centre → refer`.

- Chỉ lấy MCP của 4 ngón dài, biên độ ±15° như bài. Bộ 3 điểm là check = PIP, refer = MCP, centre = cổ tay. Như vậy góc đo được là độ lệch ngang của đốt gần so với xương bàn tay.
- **Không dùng dạng–khép của ngón cái**, vì trên ảnh 2D dấu của góc này đổi khi bàn tay quay mu ra trước, nên sẽ sửa nhầm.

## Kiểm tra trước khi dùng

Chạy `rectify_hands.py --stats` trên tập train. Xem cột "% bàn tay lỗi" của từng dòng. Lỗi của MediaPipe hiếm, nên dòng nào bắt lỗi quá nhiều bàn tay là dòng đó đo sai, phải bỏ hoặc sửa. Ghi lại quyết định này vào báo cáo.
