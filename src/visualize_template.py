"""Vẽ keypoint của một mẫu trong csv (của SPOTER hoặc do mình tạo) để kiểm tra bằng mắt.

    python src/visualize_template.py --csv data/csv/train.csv --row 0 --frame 20 --out results/check.png

Nên vẽ cùng một video từ file SPOTER và file của mình: hai hình phải giống nhau về bố cục,
tay trái/phải cùng màu, đầu ở trên.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import TEMPLATE_CFG
from template import classify_base, load_cfg, read_csv_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--frame", type=int, default=-1, help="-1 = frame giữa video")
    ap.add_argument("--cfg", default=str(TEMPLATE_CFG))
    ap.add_argument("--out", default="results/check_keypoints.png")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    s = read_csv_samples(args.csv, cfg)[args.row]
    T = len(next(iter(s["points"].values())))
    body = [b for b in s["points"] if classify_base(b)[0] == "body"]
    real = [t for t in range(T) if any((s["points"][b][t] != cfg["missing_value"]).any() for b in body)]
    n_real = len(real)
    f = (real[n_real // 2] if real else 0) if args.frame < 0 else args.frame

    fig, ax = plt.subplots(figsize=(6, 6))
    for base, arr in s["points"].items():
        x, y = arr[f]
        if x == cfg["missing_value"] and y == cfg["missing_value"]:
            continue
        kind, a, b = classify_base(base)
        if kind == "unknown":
            continue
        if kind == "hand":
            side = cfg["hand_suffix_side"].get(b)
        else:
            side = a
        color = {"left": "tab:red", "right": "tab:blue"}.get(side, "black")
        ax.scatter(x, y, c=color, s=18)
        if kind == "body" or a in ("wrist", "thumbTip", "indexTip"):
            ax.annotate(base, (x, y), fontsize=7)
    if not cfg["y_up"]:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"nhãn {s['label']}, frame {f} (video thật dài {n_real}/{T} frame)\nđỏ = tên 'left', xanh = tên 'right'", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Đã lưu {args.out}")


if __name__ == "__main__":
    main()
