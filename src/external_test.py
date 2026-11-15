"""Thử nghiệm trên người ký mới: video ASL Citizen (Microsoft, 2023), quay bằng webcam tại nhà.

ASL Citizen gồm 83.399 video, 2.731 ký hiệu, 52 người ký Điếc/khiếm thính, KHÔNG trùng video với WLASL.
Script chỉ lấy các từ có trong 100 từ của đồ án, ở tập test của ASL Citizen (11 người ký).

Ba bước:

    # 1) Lấy video. Mặc định đọc thẳng file zip trên máy chủ Microsoft, chỉ tải những video cần
    #    (cần: pip install remotezip). Nếu máy chủ không cho đọc từng phần, tải cả file zip 43 GB
    #    về rồi dùng --zip D:\\...\\ASL_Citizen.zip
    python src/external_test.py fetch --per_gloss 5

    # 2) Trích keypoint (chạy được song song: --part 0 --nparts 2 ở hai cửa sổ)
    python src/external_test.py extract

    # 3) Ghép csv: bản thô và bản đã hiệu chỉnh tay (dip_only), cùng thứ tự video
    python src/external_test.py build --mirror none
    python src/external_test.py build --mirror signer

Sau đó chấm bằng predict_split.py + evaluate.py như tập test thường.
--mirror signer: tính trên TẤT CẢ video của một người ký; nếu tay trái xuất hiện nhiều hơn tay phải
quá --hand_ratio lần (người thuận tay trái) thì lật ngang mọi video của người đó, để tay thuận luôn
ở phía tay phải như dữ liệu huấn luyện. Thuận tay là đặc điểm của người, nên quyết định theo người ổn
định hơn --mirror auto (quyết định riêng từng video, dễ sai ở ký hiệu hai tay đối xứng).
"""
import argparse
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA, LABEL_MAP, ROOT, TARGET_FPS, TEMPLATE_CFG

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
OUT = DATA / "asl_citizen"

# cặp điểm trái/phải của MediaPipe Pose (33 điểm), dùng khi lật ngang
POSE_LR = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
           (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def norm_gloss(g):
    """'DRINK1', 'drink_2', 'Drink' -> 'drink'. Giữ chữ cái, bỏ số và ký tự phụ ở cuối."""
    g = str(g).strip().lower()
    g = re.sub(r"[\s_\-]*\d+$", "", g)
    return re.sub(r"[^a-z]", "", g)


def open_zip(zip_path):
    if zip_path:
        return zipfile.ZipFile(zip_path)
    try:
        from remotezip import RemoteZip
    except ImportError:
        raise SystemExit("Thiếu gói remotezip: chạy  pip install remotezip  rồi chạy lại.")
    return RemoteZip(URL)


# ------------------------------------------------------------------ fetch
def cmd_fetch(a):
    labels = json.load(open(LABEL_MAP, encoding="utf-8"))
    g2l = {norm_gloss(v): int(k) for k, v in labels.items()}
    t0 = time.time()
    z = open_zip(a.zip)
    names = z.namelist()
    print(f"Mở zip: {len(names)} file ({time.time() - t0:.0f} s)")
    split_file = next((n for n in names if n.lower().endswith(f"splits/{a.split}.csv")), None)
    if split_file is None:
        raise SystemExit(f"Không thấy splits/{a.split}.csv trong zip. Các file csv có: "
                         + ", ".join(n for n in names if n.endswith(".csv")))
    df = pd.read_csv(io.BytesIO(z.read(split_file)))
    df.columns = [c.strip() for c in df.columns]
    col_vid = next(c for c in df.columns if "video" in c.lower())
    col_g = next(c for c in df.columns if c.lower() == "gloss")
    col_p = next(c for c in df.columns if "participant" in c.lower())
    df["norm"] = df[col_g].map(norm_gloss)
    df = df[df.norm.isin(g2l)].copy()
    df["label"] = df.norm.map(g2l)
    found = sorted(set(df.norm))
    print(f"Từ của đồ án có trong {a.split} của ASL Citizen: {len(found)}/100")
    missing = sorted(set(g2l) - set(found))
    if missing:
        print("  không có:", ", ".join(missing))
    variants = df.groupby("norm")[col_g].nunique()
    if (variants > 1).any():
        print("  từ có nhiều biến thể ký (giữ cả):", ", ".join(variants[variants > 1].index))
    # mỗi từ lấy tối đa per_gloss video, ưu tiên người ký khác nhau
    rng = np.random.default_rng(0)
    pick = []
    for g, sub in df.groupby("norm"):
        sub = sub.sample(frac=1, random_state=int(rng.integers(1e9)))
        sub = sub.drop_duplicates(col_p).head(a.per_gloss) if sub[col_p].nunique() >= a.per_gloss \
            else sub.head(a.per_gloss)
        pick.append(sub)
    sel = pd.concat(pick).reset_index(drop=True)
    vdir = OUT / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    zname = {Path(n).name: n for n in names if n.lower().endswith(".mp4")}
    rows = []
    for k, (_, r) in enumerate(sel.iterrows(), 1):
        fn = Path(str(r[col_vid])).name
        dst = vdir / fn
        if not dst.exists():
            if fn not in zname:
                print("  không tìm thấy trong zip:", fn)
                continue
            dst.write_bytes(z.read(zname[fn]))
        rows.append({"video": fn, "gloss": labels[str(int(r["label"]))], "label": int(r["label"]),
                     "participant": r[col_p], "source_gloss": r[col_g]})
        if k % 50 == 0:
            print(f"  {k}/{len(sel)} video ({time.time() - t0:.0f} s)")
    pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False)
    print(f"Đã lấy {len(rows)} video, {len({r['label'] for r in rows})} từ, "
          f"{len({r['participant'] for r in rows})} người ký -> {OUT / 'manifest.csv'}")


# ------------------------------------------------------------------ extract
def cmd_extract(a):
    from keypoints import extract_video
    man = pd.read_csv(OUT / "manifest.csv")
    kdir = OUT / "keypoints"
    kdir.mkdir(parents=True, exist_ok=True)
    man = man.iloc[a.part::a.nparts]
    t0 = time.time()
    for k, r in enumerate(man.itertuples(), 1):
        dst = kdir / (Path(r.video).stem + ".npz")
        if dst.exists():
            continue
        d = extract_video(OUT / "videos" / r.video)
        np.savez_compressed(dst, **d)
        if k % 20 == 0:
            el = time.time() - t0
            print(f"  {k}/{len(man)} video, {el / k:.1f} s/video, còn khoảng {el / k * (len(man) - k) / 60:.0f} phút")
    print("Xong.")


# ------------------------------------------------------------------ build
def mirror(pose, left, right):
    pose = pose.copy()
    pose[..., 0] = 1.0 - pose[..., 0]
    for i, j in POSE_LR:
        pose[:, [i, j]] = pose[:, [j, i]]
    L, R = right.copy(), left.copy()
    L[..., 0] = 1.0 - L[..., 0]
    R[..., 0] = 1.0 - R[..., 0]
    return pose, L, R


def motion(hand):
    w = hand[:, 0]                       # cổ tay
    ok = ~np.isnan(w).any(axis=1)
    if ok.sum() < 2:
        return 0.0
    w = w[ok]
    return float(np.abs(np.diff(w, axis=0)).sum()) + 0.01 * ok.sum()   # di chuyển + số frame thấy tay


def trim(pose, left, right, margin):
    """Bỏ đoạn đầu/cuối không thấy tay nào (người ký chưa giơ tay hoặc đã hạ tay)."""
    has = ~(np.isnan(left[:, 0, 0]) & np.isnan(right[:, 0, 0]))
    if not has.any():
        return None
    i0 = max(0, np.argmax(has) - margin)
    i1 = min(len(has), len(has) - np.argmax(has[::-1]) + margin)
    return pose[i0:i1], left[i0:i1], right[i0:i1]


def cmd_build(a):
    from keypoints import resample, to_template
    from rectify_hands import load_tables, rectify_points
    from template import load_cfg, write_csv
    cfg = load_cfg(TEMPLATE_CFG)
    tabs = load_tables(ROOT / a.tables)
    man = pd.read_csv(OUT / "manifest.csv")
    items = []
    for r in man.itertuples():
        f = OUT / "keypoints" / (Path(r.video).stem + ".npz")
        if not f.exists():
            continue
        d = np.load(f)
        t = trim(d["pose"], d["left"], d["right"], a.margin)
        if t is None or len(t[0]) < 5:
            continue
        items.append((r, d, t))

    # thuận tay theo người ký: tổng số frame thấy tay trái / tay phải trên mọi video của người đó
    vis = {}
    for r, d, (pose, left, right) in items:
        L, R = vis.get(r.participant, (0, 0))
        vis[r.participant] = (L + int((~np.isnan(left[:, 0, 0])).sum()), R + int((~np.isnan(right[:, 0, 0])).sum()))
    left_handed = {p for p, (L, R) in vis.items() if L > a.hand_ratio * max(R, 1)}
    if a.mirror == "signer":
        print("Người ký   frame tay trái / tay phải   tỉ lệ   lật?")
        for p in sorted(vis, key=lambda q: int(q[1:]) if q[1:].isdigit() else 0):
            L, R = vis[p]
            print(f"  {p:<8} {L:>6} / {R:<6}               {L / max(R, 1):.2f}    {'có' if p in left_handed else ''}")

    raw, rect, kept, n_mirror = [], [], [], 0
    for r, d, (pose, left, right) in items:
        flip = (a.mirror == "all"
                or (a.mirror == "auto" and motion(left) > motion(right))
                or (a.mirror == "signer" and r.participant in left_handed))
        if flip:
            pose, left, right = mirror(pose, left, right)
            n_mirror += 1
        pose, left, right = resample(pose, left, right, float(d["fps"]), TARGET_FPS)
        if len(pose) > cfg["fixed_len"]:                      # quá dài: lấy mẫu đều cho vừa 204 frame
            idx = np.linspace(0, len(pose) - 1, cfg["fixed_len"]).round().astype(int)
            pose, left, right = pose[idx], left[idx], right[idx]
        w, h = int(d["w"]), int(d["h"])
        pts = to_template(pose, left, right, cfg, w, h)
        base = {"label": int(r.label), "w": w, "h": h, "fps": TARGET_FPS}
        raw.append({**base, "points": pts})
        rect.append({**base, "points": rectify_points(pts, cfg, tabs, aspect=w / h if h else 1.0)})
        kept.append({"video": r.video, "label": r.label, "gloss": r.gloss, "participant": r.participant,
                     "frames": len(pose), "mirrored": int(flip)})
    tag = f"mirror_{a.mirror}"
    write_csv(raw, cfg, OUT / f"test_raw_{tag}.csv")
    write_csv(rect, cfg, OUT / f"test_rect_dip_{tag}.csv")
    pd.DataFrame(kept).to_csv(OUT / f"test_ids_{tag}.csv", index=False)
    print(f"{len(kept)}/{len(man)} video dùng được, {len({k['label'] for k in kept})} từ, "
          f"{len({k['participant'] for k in kept})} người ký; lật ngang {n_mirror} video")
    print(f"Đã ghi {OUT / f'test_raw_{tag}.csv'} và {OUT / f'test_rect_dip_{tag}.csv'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--zip", default=None, help="đường dẫn ASL_Citizen.zip nếu đã tải cả file")
    f.add_argument("--split", default="test")
    f.add_argument("--per_gloss", type=int, default=5)
    e = sub.add_parser("extract")
    e.add_argument("--part", type=int, default=0)
    e.add_argument("--nparts", type=int, default=1)
    b = sub.add_parser("build")
    b.add_argument("--mirror", choices=["none", "auto", "signer", "all"], default="none")
    b.add_argument("--hand_ratio", type=float, default=1.2,
                   help="--mirror signer: lật người ký nếu frame tay trái > hand_ratio x frame tay phải")
    b.add_argument("--tables", default="tables/dip_only")
    b.add_argument("--margin", type=int, default=5)
    a = ap.parse_args()
    {"fetch": cmd_fetch, "extract": cmd_extract, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
