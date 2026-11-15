"""Bước 2: lập danh sách video WLASL100 theo cách chia chính thức và kiểm tra video nào có.

    python src/prepare_wlasl.py --json data/WLASL_v0.3.json --videos data/videos [--nslt data/nslt_100.json]

- WLASL100 = 100 gloss đầu tiên trong WLASL_v0.3.json (file được xếp theo số video giảm dần).
- Nếu có nslt_100.json (trong repo WLASL, thư mục code/I3D/preprocess) thì dùng nó để lấy đúng danh sách
  video và tập train/val/test; nên dùng nếu có.
Sinh data/manifest.csv và data/label_map.json.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from config import DATA, LABEL_MAP, NUM_CLASSES

VIDEO_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def find_video(folder, vid):
    for ext in VIDEO_EXT:
        p = folder / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def fill_empty_splits(df, glosses, seed):
    """Giữ nguyên cách chia chính thức; lớp nào không còn video ở val hoặc test thì chuyển 1 video
    (chọn ngẫu nhiên, cố định theo seed) từ train của chính lớp đó sang, miễn là train còn ít nhất 2 video."""
    import numpy as np
    rng = np.random.default_rng(seed)
    moved = []
    for c in sorted(df.label.unique()):
        for sp in ("test", "val"):
            have = df[(df.label == c) & df.available & (df.split == sp)]
            train = df[(df.label == c) & df.available & (df.split == "train")]
            if len(have) == 0 and len(train) > 2:
                i = train.index[rng.integers(len(train))]
                df.loc[i, "split"] = sp
                moved.append((glosses[c], df.loc[i, "video_id"], sp))
    print(f"Bù {len(moved)} video từ train sang val/test cho các lớp bị trống "
          f"(cột official_split vẫn giữ cách chia gốc).")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--nslt", default=None)
    ap.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--out", default=str(DATA / "manifest.csv"))
    ap.add_argument("--no_fill", action="store_true",
                    help="không bù video cho lớp bị trống ở val/test (mặc định: có bù)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        wlasl = json.load(f)
    glosses = [e["gloss"] for e in wlasl[: args.num_classes]]
    label_of = {g: i for i, g in enumerate(glosses)}
    nslt = None
    if args.nslt:
        with open(args.nslt, encoding="utf-8") as f:
            nslt = json.load(f)

    folder = Path(args.videos)
    rows = []
    for entry in wlasl[: args.num_classes]:
        for inst in entry["instances"]:
            vid = str(inst["video_id"])
            split = inst["split"]
            if nslt is not None:
                if vid not in nslt:
                    continue
                split = nslt[vid]["subset"]
            p = find_video(folder, vid)
            rows.append({
                "video_id": vid, "gloss": entry["gloss"], "label": label_of[entry["gloss"]],
                "split": split, "path": str(p) if p else "",
                "frame_start": int(inst.get("frame_start", 1)), "frame_end": int(inst.get("frame_end", -1)),
                "signer_id": inst.get("signer_id", ""), "available": p is not None,
            })
    df = pd.DataFrame(rows)
    df["official_split"] = df["split"]
    if not args.no_fill:
        df = fill_empty_splits(df, glosses, args.seed)
    df.to_csv(args.out, index=False)
    with open(LABEL_MAP, "w", encoding="utf-8") as f:
        json.dump({i: g for i, g in enumerate(glosses)}, f, ensure_ascii=False, indent=1)

    print(f"Đã lưu {args.out} ({len(df)} video) và {LABEL_MAP}")
    tab = df.groupby("split")["available"].agg(["count", "sum"]).rename(columns={"count": "trong_json", "sum": "tai_duoc"})
    tab["ti_le"] = (tab["tai_duoc"] / tab["trong_json"]).map("{:.1%}".format)
    print(tab.to_string())
    per_class = df[df.available].groupby(["label", "split"]).size().unstack(fill_value=0)
    for sp in ("train", "val", "test"):
        if sp in per_class:
            empty = per_class.index[per_class[sp] == 0].tolist()
            if empty:
                print(f"[CẢNH BÁO] {len(empty)} lớp không còn video nào ở tập {sp}: {[glosses[i] for i in empty]}")
    lost = sorted(set(range(len(glosses))) - set(per_class.index))
    if lost:
        print(f"[CẢNH BÁO] {len(lost)} lớp mất hết video: {[glosses[i] for i in lost]}")


if __name__ == "__main__":
    main()
