"""Bước 3a: phân tích file csv gốc của SPOTER để biết định dạng cần ghi.

    python src/inspect_template.py --csv path/to/WLASL100_train.csv

Sinh data/template_config.json. Đọc kỹ phần in ra, sau đó chạy visualize_template.py để kiểm tra bằng mắt.
"""
import argparse
import json

from config import TEMPLATE_CFG
from template import inspect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="một file csv dữ liệu SPOTER (WLASL100)")
    ap.add_argument("--out", default=str(TEMPLATE_CFG))
    args = ap.parse_args()

    cfg = inspect(args.csv)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    st = cfg["stats"]
    print(f"Số keypoint: {st['n_keypoints']} (54 điểm mô hình dùng + {len(cfg['unknown_bases'])} cột phụ)")
    print(f"Cột nhãn: {cfg['label_col']}  | nhãn từ {cfg['label_min']} đến {cfg['label_max']}")
    print(f"Thang giá trị: {cfg['scale']} (giá trị lớn nhất thấy được {st['max_value_seen']:.3f})")
    print(f"Trục y hướng lên: {cfg['y_up']}  (y mũi {st['nose_y']:.3f}, y vai {st['shoulder_y']:.3f})")
    print(f"Đổi tên trái/phải so với MediaPipe: {cfg['swap_lr']}  "
          f"(x vai trái {st['leftShoulder_x']:.3f}, x vai phải {st['rightShoulder_x']:.3f})")
    print(f"Hậu tố tay -> bên (theo tên trong khuôn): {cfg['hand_suffix_side']}")
    print(f"Tỉ lệ giá trị 0 ở các điểm tay: {st['mean_hand_zero_ratio']:.2%} (mất tay được ghi bằng 0)")
    print(f"Số frame mỗi video min/TB/max: {st['frames_min_mean_max']}")
    print(f"Độ dài cố định: {cfg['fixed_len']}  (None = độ dài thay đổi; nếu có số thì dữ liệu của mình sẽ được đệm cho bằng)")
    if cfg["unknown_bases"]:
        print(f"Cột không thuộc 54 điểm mô hình dùng: {cfg['unknown_bases']} -> Dataset của tác giả không đọc, sẽ ghi 0 cho đủ cột")
    print(f"Đã lưu {args.out}")


if __name__ == "__main__":
    main()
