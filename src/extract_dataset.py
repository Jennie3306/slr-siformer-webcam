"""Bước 3b: trích keypoint MediaPipe cho toàn bộ video trong manifest, lưu mỗi video một file .npz.

    python src/extract_dataset.py --manifest data/manifest.csv --out data/keypoints
    # chạy song song trên nhiều cửa sổ terminal:
    python src/extract_dataset.py --part 0 --nparts 3   (cửa sổ 1)
    python src/extract_dataset.py --part 1 --nparts 3   (cửa sổ 2) ...

Chạy lại được: video nào đã có .npz thì bỏ qua.
--no_trim: dùng khi video đã được cắt sẵn theo từng ký hiệu (ví dụ bản đã xử lý trên Kaggle).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA
from keypoints import extract_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest.csv"))
    ap.add_argument("--out", default=str(DATA / "keypoints"))
    ap.add_argument("--no_trim", action="store_true")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--model_complexity", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest, dtype={"video_id": str})
    df = df[df.available].reset_index(drop=True)
    df = df.iloc[args.part :: args.nparts]
    log = []
    t0 = time.time()
    for k, r in enumerate(df.itertuples(), 1):
        dst = out / f"{r.video_id}.npz"
        if dst.exists():
            continue
        start, end = (0, None) if args.no_trim else (max(r.frame_start - 1, 0), r.frame_end if r.frame_end > 0 else None)
        note = ""
        try:
            d = extract_video(r.path, start, end, args.model_complexity)
            if len(d["pose"]) == 0 and start > 0:
                note = f"frame_start {r.frame_start} vượt quá độ dài video {d['total_frames']}: dùng cả video"
                d = extract_video(r.path, 0, None, args.model_complexity)
            np.savez_compressed(dst, pose=d["pose"], left=d["left"], right=d["right"],
                                w=d["w"], h=d["h"], fps=d["fps"])
            n = len(d["pose"])
            pose_ok = float((~np.isnan(d["pose"][:, 0, 0])).mean()) if n else 0.0
            hand_ok = float(((~np.isnan(d["left"][:, 0, 0])) | (~np.isnan(d["right"][:, 0, 0]))).mean()) if n else 0.0
            log.append({"video_id": r.video_id, "frames": n, "pose_ratio": pose_ok, "any_hand_ratio": hand_ok, "note": note})
        except Exception as e:  # ghi lại để xử lý sau, không dừng cả quá trình
            log.append({"video_id": r.video_id, "frames": 0, "pose_ratio": 0, "any_hand_ratio": 0, "note": f"LỖI: {e}"})
        if k % 50 == 0:
            el = time.time() - t0
            print(f"{k}/{len(df)} video, {el / 60:.1f} phút, còn khoảng {el / k * (len(df) - k) / 60:.1f} phút")
    if log:
        lp = DATA / f"extract_log_part{args.part}.csv"
        pd.DataFrame(log).to_csv(lp, index=False)
        bad = sum(1 for x in log if x["note"].startswith("LỖI"))
        print(f"Xong. {len(log)} video mới, {bad} lỗi. Log: {lp}")
    else:
        print("Không có video mới cần trích.")


if __name__ == "__main__":
    main()
