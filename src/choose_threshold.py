"""Chọn ngưỡng tin cậy cho webcam từ dự đoán trên tập VAL (không dùng tập test).

    python src/choose_threshold.py --pred results/siformer_s0/pred_val.npz

Với mỗi ngưỡng: bao nhiêu % ký hiệu được trả lời (độ phủ) và độ chính xác trên phần được trả lời.
Chọn ngưỡng cân bằng hai cột này, rồi sửa CONF_THRESHOLD trong src/config.py.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()
    d = np.load(args.pred)
    z = d["logits"] - d["logits"].max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    conf, pred, y = p.max(1), p.argmax(1), d["y"]
    print("ngưỡng | độ phủ | độ chính xác trên phần được trả lời")
    for t in np.arange(0.0, 0.95, 0.05):
        keep = conf >= t
        acc = (pred[keep] == y[keep]).mean() if keep.any() else float("nan")
        print(f" {t:.2f}  | {keep.mean():6.1%} | {acc:6.1%}")


if __name__ == "__main__":
    main()
