"""Bước 6c: tạo các bản tập test bị xóa keypoint để đo độ bền (giống mục 4.4 của bài).

    python src/robustness.py --inp data/csv/test.csv --outdir data/csv/robust

Sinh: test_drop_left_hand.csv, test_drop_right_hand.csv, test_drop_body.csv,
và test_drop_random{p}.csv (mỗi frame mất ngẫu nhiên một tay với xác suất p, mô phỏng MediaPipe mất tay).
Tên 'left/right' ở đây theo tên trong khuôn. Chạy predict_split.py trên từng file rồi so sánh.
"""
import argparse
from pathlib import Path

import numpy as np

from config import TEMPLATE_CFG
from template import classify_base, load_cfg, read_csv_samples, write_csv


def groups(cfg):
    g = {"left_hand": [], "right_hand": [], "body": []}
    for b in cfg["keypoint_bases"]:
        kind, a, s = classify_base(b)
        if kind == "hand":
            g[f"{cfg['hand_suffix_side'][s]}_hand"].append(b)
        elif kind == "body":
            g["body"].append(b)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cfg", default=str(TEMPLATE_CFG))
    ap.add_argument("--random_p", type=float, nargs="*", default=[0.2, 0.4])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    miss = cfg["missing_value"]
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    g = groups(cfg)
    base_samples = read_csv_samples(args.inp, cfg)

    for name, bases in g.items():
        samples = read_csv_samples(args.inp, cfg)
        for s in samples:
            for b in bases:
                s["points"][b][:] = miss
        write_csv(samples, cfg, out / f"test_drop_{name}.csv")
        print(f"test_drop_{name}.csv: xóa {len(bases)} điểm")

    rng = np.random.default_rng(args.seed)
    for p in args.random_p:
        samples = read_csv_samples(args.inp, cfg)
        for s in samples:
            T = len(s["points"][cfg["keypoint_bases"][0]])
            for hand in ("left_hand", "right_hand"):
                drop = rng.random(T) < p / 2
                for b in g[hand]:
                    s["points"][b][drop] = miss
        write_csv(samples, cfg, out / f"test_drop_random{int(p * 100)}.csv")
        print(f"test_drop_random{int(p * 100)}.csv: mỗi frame mất một tay với xác suất {p}")
    print(f"Tập gốc: {len(base_samples)} video")


if __name__ == "__main__":
    main()
