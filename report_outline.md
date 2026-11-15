# Dàn ý báo cáo và nguồn số liệu

Mỗi mục ghi rõ lấy số / hình từ đâu. Viết ngắn, mỗi bảng hoặc hình kèm 1–2 câu nhận xét.

## 1. Giới thiệu
- Bài toán: nhận diện từ đơn lẻ ASL qua webcam, chạy trên laptop. Phạm vi: 100 từ (WLASL100), không dùng khuôn mặt, không nhận diện câu.
- Đóng góp của đồ án (mục 2 file 05): đưa Siformer sang MediaPipe, đánh giá trên tập chia chính thức, tách ký hiệu từ webcam, đo tốc độ và thử thật.

## 2. Khảo sát và lựa chọn mô hình
- Tóm tắt các dòng mô hình, bảng so sánh, ma trận chấm điểm: lấy từ file tổng hợp so sánh (00_Tong_hop...).
- Lý do chọn Siformer, phương án dự phòng DSTA-SLR.

## 3. Phương pháp
- Kiến trúc Siformer: sơ đồ và bảng thành phần (file 05, mục 1).
- Hệ thống: sơ đồ huấn luyện và chạy thời gian thực (file 05, mục 1).
- Trích keypoint: MediaPipe → 54 điểm, bảng ánh xạ (file 04), cách xử lý mất tay, gán tay theo vị trí cổ tay.
- Tách ký hiệu: chế độ bấm phím, chế độ tự động và các tham số `SEG` cuối cùng (`src/config.py`).

## 4. Thực nghiệm
### 4.1 Dữ liệu
- Bảng số video tải được theo train/val/test: in ra từ `prepare_wlasl.py`.
- Số video bị bỏ khi trích: `data/csv/build_report.txt`.
### 4.2 Chạy lại kết quả gốc
- Top-1 trên dữ liệu của tác giả so với 86,50%: `repro_train.log` (Colab).
### 4.3 Kết quả chính
- Bảng SPOTER vs Siformer (PI/PC Top-1, Top-5, trung bình ± độ lệch chuẩn 3 seed): `results/siformer_summary.json`, `results/spoter_summary.json`.
- Đường học: `results/fig_curves_siformer.png`.
- Giải thích nếu thấp hơn 86,50%: đổi bộ trích pose, đổi cách chia test (bài dùng 8 video ngẫu nhiên mỗi lớp + SMOTE).
### 4.4 Phân tích lỗi
- Độ chính xác theo lớp: `results/siformer_s0/per_class_hist.png`, `per_class.csv`.
- Cặp nhầm nhiều nhất: `results/siformer_s0/confused_pairs.png`; nhận xét có phải các cặp ký hiệu giống nhau không.
### 4.5 Độ bền
- `results/fig_robust.png`: so Siformer và SPOTER khi mất tay / thân.
### 4.6 Tốc độ
- `results/fig_latency.png`, `results/benchmark_siformer.json`; ghi cấu hình laptop (CPU, RAM, hệ điều hành, số luồng).
- FPS thực tế khi chạy webcam: cột `mediapipe_ms_mean` trong `results/webcam_log.csv`.
### 4.7 Thử nghiệm thật
- `results/real_test_summary.csv`: Top-1, Top-3, độ trễ theo điều kiện; 5 từ khó nhất.

## 5. Thảo luận và hạn chế
- Khoảng cách giữa benchmark và dùng thật; nguyên nhân nhìn thấy được.
- Không dùng khuôn mặt; chỉ 100 từ; dữ liệu WLASL mất video; tách ký hiệu tự động còn nhạy.

## 6. Kết luận và hướng phát triển
- Tóm tắt con số chính. Hướng mở: thêm điểm khuôn mặt, mở rộng WLASL300, label smoothing theo nghĩa (Bài 2).

## Phụ lục
- Phiên bản thư viện (`results/env_laptop.txt`), lệnh huấn luyện, cấu hình `template_config.json`.
