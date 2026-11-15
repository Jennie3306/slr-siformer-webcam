"""Bước 6a: chạy mô hình trên một file csv, lưu dự đoán để phân tích.

    python src/predict_split.py --which siformer --ckpt results/siformer_s0/best.pth \
        --csv data/csv/test.csv --out results/siformer_s0/pred_test.npz

File .npz chứa: y (nhãn đúng), logits (N, số lớp), ms (thời gian mô hình mỗi mẫu).
"""
import argparse
from pathlib import Path

import numpy as np

from model_adapter import ModelAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["siformer", "spoter"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    ad = ModelAdapter(args.which, args.ckpt, threads=args.threads)
    res = ad.predict_csv(args.csv)
    y = np.array([r[0] for r in res])
    logits = np.stack([r[1] for r in res])
    ms = np.array([r[2] for r in res])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, y=y, logits=logits, ms=ms)
    print(f"{len(y)} mẫu, Top-1 = {(logits.argmax(1) == y).mean():.4f}. Đã lưu {args.out}")


if __name__ == "__main__":
    main()
