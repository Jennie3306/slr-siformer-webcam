"""Bước 6d: đo tốc độ trên laptop (CPU).

    python src/benchmark.py --video data/videos/<một video>.mp4 --ckpt results/siformer_s0/best.pth \
        --csv data/csv/test.csv --out results/benchmark_siformer.json [--threads 4]

Đo ba phần: MediaPipe mỗi frame; mô hình mỗi ký hiệu (chỉ forward); đường đi đầy đủ sau khi kết thúc ký hiệu
(đổi định dạng + Dataset của tác giả + forward), giống những gì webcam_app làm.
Bỏ 20 lần đầu để "làm nóng", báo trung bình, trung vị và P95.
"""
import argparse
import json
import os
import platform
import time

import numpy as np

from config import TEMPLATE_CFG
from template import load_cfg, read_csv_samples


def stats(ms):
    ms = np.asarray(ms)
    return {"n": int(len(ms)), "mean": float(ms.mean()), "median": float(np.median(ms)), "p95": float(np.percentile(ms, 95))}


def bench_mediapipe(video, n, warmup=20):
    import cv2
    from keypoints import Extractor
    cap = cv2.VideoCapture(video)
    ex = Extractor()
    ms, i = [], 0
    while len(ms) < n:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        t = time.perf_counter()
        ex(frame)
        if i >= warmup:
            ms.append((time.perf_counter() - t) * 1000)
        i += 1
    ex.close()
    return stats(ms), frame.shape[:2]


def bench_model(which, ckpt, csv, n, threads, warmup=20, rectify=False):
    from model_adapter import ModelAdapter
    cfg = load_cfg(TEMPLATE_CFG)
    ad = ModelAdapter(which, ckpt, threads=threads)
    ds = ad.dataset(csv)
    samples = read_csv_samples(csv, cfg)
    m = min(n + warmup, len(ds))
    if rectify:
        from rectify_hands import _aspect, load_tables, rectify_points
        from config import ROOT
        tabs = load_tables(ROOT / rectify)
    fwd, full = [], []
    for i in range(m):
        item = ds[i]
        t = time.perf_counter()
        ad.forward(item)
        a = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        pts = samples[i]["points"]
        if rectify:
            pts = rectify_points(pts, cfg, tabs, aspect=_aspect(samples[i]))
        ad.predict_points(pts, cfg)   # đường đi đầy đủ như webcam
        b = (time.perf_counter() - t) * 1000
        if i >= warmup:
            fwd.append(a); full.append(b)
    import torch
    return stats(fwd), stats(full), torch.get_num_threads()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--ckpt")
    ap.add_argument("--csv")
    ap.add_argument("--which", default="siformer", choices=["siformer", "spoter"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rectify", nargs="?", const="tables/dip_only", default=None, metavar="BANG",
                    help="tính cả thời gian hiệu chỉnh tư thế tay (bộ bảng, mặc định tables/dip_only)")
    args = ap.parse_args()

    res = {"machine": {"platform": platform.platform(), "processor": platform.processor(),
                       "cpu_count": os.cpu_count(), "python": platform.python_version()}}
    if args.video:
        res["mediapipe_per_frame_ms"], shape = bench_mediapipe(args.video, args.n)
        res["video_frame_hw"] = list(shape)
        print(f"MediaPipe mỗi frame: {res['mediapipe_per_frame_ms']}")
    if args.ckpt and args.csv:
        fwd, full, th = bench_model(args.which, args.ckpt, args.csv, args.n, args.threads, rectify=args.rectify)
        res["model_forward_ms"], res["after_sign_total_ms"], res["torch_threads"] = fwd, full, th
        print(f"Mô hình (forward): {fwd}\nSau khi kết thúc ký hiệu (đầy đủ): {full}")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Đã lưu {args.out}. Nhớ ghi thêm cấu hình RAM / CPU cụ thể của laptop vào báo cáo.")


if __name__ == "__main__":
    main()
