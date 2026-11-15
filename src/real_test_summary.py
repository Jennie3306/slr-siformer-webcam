"""Bước 7b: tổng hợp kết quả thử nghiệm thật từ results/webcam_log.csv.

    python src/real_test_summary.py [--log results/webcam_log.csv]

Chỉ tính các dòng có cột target (chạy webcam_app.py với --targets). Gộp theo điều kiện (--condition).
"""
import argparse

import pandas as pd

from config import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(RESULTS / "webcam_log.csv"))
    ap.add_argument("--out", default=str(RESULTS / "real_test_summary.csv"))
    args = ap.parse_args()
    df = pd.read_csv(args.log)
    df = df[df["target"].notna() & (df["target"].astype(str) != "")]
    if df.empty:
        raise SystemExit("Chưa có dòng nào có target. Chạy webcam_app.py với --targets.")
    df["top1_ok"] = df["top1"] == df["target"]
    df["top3_ok"] = (df[["top1", "top2", "top3"]].values == df[["target"]].values).any(axis=1)
    g = df.groupby(["condition", "mode"]).agg(
        n=("target", "size"), top1=("top1_ok", "mean"), top3=("top3_ok", "mean"),
        tre_sau_ky_ms=("end_to_result_ms", "mean"), mediapipe_ms=("mediapipe_ms_mean", "mean"))
    g["top1"] = (g["top1"] * 100).round(1)
    g["top3"] = (g["top3"] * 100).round(1)
    g.to_csv(args.out)
    print(g.to_string())
    worst = df.groupby("target")["top1_ok"].mean().sort_values().head(5)
    print("\nTừ khó nhất (Top-1):\n" + (worst * 100).round(0).to_string())
    print(f"\nĐã lưu {args.out}")


if __name__ == "__main__":
    main()
