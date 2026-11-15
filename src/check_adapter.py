"""Kiểm tra adapter: độ chính xác tính qua adapter phải bằng độ chính xác mà train.py của tác giả in ra.

    python src/check_adapter.py --which siformer --ckpt path/to/checkpoint.pth --csv path/to/WLASL100_test.csv

Nếu lệch: xem lại forward(), Dataset hoặc cách nạp checkpoint trong model_adapter.py.
"""
import argparse

import numpy as np

from model_adapter import ModelAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["siformer", "spoter"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ad = ModelAdapter(args.which, args.ckpt)
    res = ad.predict_csv(args.csv, args.limit)
    y = np.array([r[0] for r in res])
    pred = np.array([int(np.argmax(r[1])) for r in res])
    print(f"Số mẫu: {len(res)} | số lớp đầu ra: {len(res[0][1])} | nhãn từ {y.min()} đến {y.max()}")
    print(f"Top-1 qua adapter: {(pred == y).mean():.4f}")
    print(f"Thời gian mô hình mỗi mẫu: {np.mean([r[2] for r in res]):.1f} ms")
    print("So với độ chính xác train.py in ra cho cùng checkpoint và cùng file csv: phải GẦN BẰNG (lệch khoảng 1 điểm là bình")
    print("thường, vì ProbSparse attention lấy mẫu ngẫu nhiên; adapter cố định seed nên chạy lại vẫn ra đúng số này).")


if __name__ == "__main__":
    main()
