"""Bước 3c: gom các file .npz thành train.csv / val.csv / test.csv đúng định dạng khuôn.

    python src/build_csv.py --manifest data/manifest.csv --kp data/keypoints --out data/csv

Video bị bỏ: không có frame nào, hoặc không frame nào thấy người. Video có nhưng thiếu tay vẫn giữ
(mất tay được ghi bằng giá trị thiếu của khuôn, giống dữ liệu gốc).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA, TARGET_FPS, TEMPLATE_CFG
from keypoints import resample, to_template
from template import load_cfg, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest.csv"))
    ap.add_argument("--kp", default=str(DATA / "keypoints"))
    ap.add_argument("--cfg", default=str(TEMPLATE_CFG))
    ap.add_argument("--out", default=str(DATA / "csv"))
    ap.add_argument("--min_frames", type=int, default=5)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest, dtype={"video_id": str})
    df = df[df.available]
    report = []
    for split in ("train", "val", "test"):
        samples, dropped = [], []
        for r in df[df.split == split].itertuples():
            f = Path(args.kp) / f"{r.video_id}.npz"
            if not f.exists():
                dropped.append((r.video_id, "chưa trích"))
                continue
            d = np.load(f)
            pose, left, right = resample(d["pose"], d["left"], d["right"], float(d["fps"]), TARGET_FPS)
            if len(pose) < args.min_frames or np.isnan(pose[:, 0, 0]).all():
                dropped.append((r.video_id, "không thấy người / quá ngắn"))
                continue
            pts = to_template(pose, left, right, cfg, int(d["w"]), int(d["h"]))
            samples.append({"label": int(r.label), "points": pts, "w": int(d["w"]), "h": int(d["h"]),
                            "fps": TARGET_FPS, "video_id": r.video_id})
        write_csv(samples, cfg, out / f"{split}.csv")
        pd.DataFrame({"video_id": [s["video_id"] for s in samples]}).to_csv(out / f"{split}_ids.csv", index=False)
        n_cls = len({s["label"] for s in samples})
        line = f"{split}: {len(samples)} video, {n_cls} lớp, bỏ {len(dropped)}"
        print(line)
        report.append(line)
        report += [f"   bỏ {v}: {why}" for v, why in dropped]
    (out / "build_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"Chi tiết: {out / 'build_report.txt'}")


if __name__ == "__main__":
    main()
