"""Hiệu chỉnh tư thế tay (kinematic hand pose rectification) của Siformer, dùng lại code của tác giả.

Code tác giả (external/Siformer/rectifications/) có đủ thuật toán nhưng cần hai bảng biên độ chuyển động
mà repo không công bố. Đồ án tự xây hai bảng này ở thư mục tables/ từ Bảng 1 của bài và thứ tự 21 điểm tay
của MediaPipe (xem chú thích trong tables/README.md). Mọi phép tính góc / xoay khớp đều gọi hàm của tác giả.

    # 1) Chỉ thống kê: mỗi dòng bảng bắt lỗi bao nhiêu % bàn tay (không ghi file)
    python src/rectify_hands.py --inp data/csv/train.csv --stats results/rectify_stats_train.json

    # 2) Hiệu chỉnh cả 3 tập
    python src/rectify_hands.py --inp data/csv/train.csv --out data/csv_rect/train.csv
    python src/rectify_hands.py --inp data/csv/val.csv   --out data/csv_rect/val.csv
    python src/rectify_hands.py --inp data/csv/test.csv  --out data/csv_rect/test.csv

Mặc định alpha = 0,4 như bài (code tác giả để 0,5). Thứ tự: dạng–khép trước, gập–duỗi sau.
Webcam dùng cùng hàm rectify_points() để dữ liệu lúc chạy giống lúc huấn luyện.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import HAND_INDEX, ROOT, SIFORMER_DIR, TEMPLATE_CFG

TABLE_DIR = ROOT / "tables"
RECT_DIR = Path(SIFORMER_DIR) / "rectifications"
HAND_NAMES = sorted(HAND_INDEX, key=HAND_INDEX.get)      # 21 tên theo thứ tự MediaPipe 0..20 (= HAND_IDENTIFIERS)

_author = None


def author_code():
    """Nạp 3 file của tác giả, tắt các lệnh print từng frame của họ."""
    global _author
    if _author is None:
        if not RECT_DIR.is_dir():
            raise SystemExit(f"Không thấy {RECT_DIR}. Clone repo Siformer vào external/Siformer trước.")
        sys.path.insert(0, str(RECT_DIR))
        import rectification_utils as ru
        import abduction_adduction_rectification as aar
        import flexion_extension_rectification as fer
        for m in (ru, aar, fer):
            m.print = lambda *a, **k: None
        _author = (ru, aar, fer)
    return _author


def _lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


DEFAULT_TABLES = TABLE_DIR / "dip_only"


def load_tables(table_dir=DEFAULT_TABLES):
    aa = pd.read_csv(Path(table_dir) / "aa_active_motion.csv")
    fe = pd.read_csv(Path(table_dir) / "fe_active_motion.csv")
    t = {"aa": {k: [_lit(v) for v in aa[k]] for k in ("ranges", "paired_joint_index", "associated_joint_index")}}
    for part, sel in (("thumb", fe["fingers"] == "thumb"), ("other", fe["fingers"] != "thumb")):
        sub = fe[sel]
        t[part] = {k: [_lit(v) for v in sub[k]] for k in ("ranges", "paired_joint_index", "associated_joint_index")}
        t[part]["fingers"] = sub["fingers"].tolist()
    return t


def row_names(t):
    names = [f"AA {HAND_NAMES[p[1]]}" for p in t["aa"]["paired_joint_index"]]
    names += [f"FE {HAND_NAMES[p[1]]}" for p in t["thumb"]["paired_joint_index"]]
    names += [f"FE {HAND_NAMES[p[1]]}" for p in t["other"]["paired_joint_index"]]
    return names


def _safe_angle(ru, u, v):
    try:
        return ru.compute_angle_between(u, v)
    except ZeroDivisionError:
        return None


def angles_and_flags(hp, which, t):
    """Góc từng dòng bảng trên bàn tay gốc (cùng công thức, cùng biên độ hợp lệ như code tác giả)."""
    ru, aar, fer = author_code()
    out = []
    for rng, (c, r, ce) in zip(t["aa"]["ranges"], t["aa"]["paired_joint_index"]):
        a = _safe_angle(ru, (hp[c][0] - hp[ce][0], hp[c][1] - hp[ce][1]),
                        (hp[r][0] - hp[ce][0], hp[r][1] - hp[ce][1]))
        if a is None:
            out.append((float("nan"), False))
            continue
        vr = (-rng[1], -rng[0]) if which == "right" else rng
        out.append((a, ru.get_angle_to_rotate(a, vr) != 0))
    direction = fer.check_thumb_flexion_direction(hp)
    for part in ("thumb", "other"):
        for rng, (r, c, e) in zip(t[part]["ranges"], t[part]["paired_joint_index"]):
            cp = [hp[c][0] - hp[e][0], hp[c][1] - hp[e][1]]
            rp = [hp[r][0] - hp[c][0], hp[r][1] - hp[c][1]]
            a = _safe_angle(ru, cp, rp)
            if a is None:
                out.append((float("nan"), False))
                continue
            if part == "thumb":
                vr = rng if direction == "clockwise" else (-rng[1], -rng[0])
                a_anat = -a if direction == "clockwise" else a       # gập ngón cái = số dương
            else:
                vr = (-max(rng), max(rng))
                a_anat = abs(a)
            out.append((a_anat, ru.get_angle_to_rotate(a, vr) != 0))
    return out


def rectify_hand(hp, which, t, alpha, parts=("aa", "fe")):
    """hp: list 21 điểm [x, y] (đã nhân tỉ lệ khung hình). Trả về (bàn tay mới, số khớp đã sửa)."""
    ru, aar, fer = author_code()
    hp = [list(p) for p in hp]
    n = 0
    if "aa" in parts:
        hp, k = aar.rectify_each_hand_abduction_and_addiction(
            hp, which, t["aa"]["ranges"], t["aa"]["paired_joint_index"], t["aa"]["associated_joint_index"], alpha)
        n += k
    if "fe" in parts:
        d = fer.check_thumb_flexion_direction(hp)
        hp, k1 = fer.rectify_thumb_flexion_and_extension(
            hp, d, t["thumb"]["ranges"], t["thumb"]["paired_joint_index"], t["thumb"]["associated_joint_index"], alpha)
        hp, k2 = fer.rectify_other_flexion_and_extension(
            hp, t["other"]["ranges"], t["other"]["paired_joint_index"], t["other"]["associated_joint_index"], alpha)
        n += k1 + k2
    return [list(p) for p in hp], n


def _hand_cols(side, suffix_of):
    return [f"{name}{suffix_of[side]}" for name in HAND_NAMES]


def rectify_points(points, cfg, t=None, alpha=0.4, parts=("aa", "fe"), aspect=1.0, stats=None):
    """points: dict base -> (T,2) theo khuôn của tác giả. Trả về dict mới (không sửa bản gốc).

    aspect = rộng/cao của video: tọa độ kiểu 'unit' bị chia theo rộng và cao khác nhau nên phải nhân lại
    trước khi đo góc, xong chia trả về. stats: dict để cộng dồn thống kê (tùy chọn)."""
    t = t or load_tables()
    miss = cfg["missing_value"]
    suffix_of = {side: suf for suf, side in cfg["hand_suffix_side"].items()}
    out = {k: np.array(v, dtype=float, copy=True) for k, v in points.items()}
    names = row_names(t)
    for which in ("right", "left"):
        cols = _hand_cols(which, suffix_of)
        H = np.stack([out[c] for c in cols], axis=1)            # (T, 21, 2)
        for f in range(H.shape[0]):
            P = H[f]
            if np.all(P == miss) or not np.isfinite(P).all():
                continue
            hp = [[float(x) * aspect, float(y)] for x, y in P]   # float Python: chia 0 -> ZeroDivisionError như tác giả mong đợi
            if stats is not None:
                fl = angles_and_flags(hp, which, t)
                stats["hands"] += 1
                stats["any"] += int(any(x[1] for x in fl))
                for name, (a, bad) in zip(names, fl):
                    stats["rows"].setdefault(name, {"flag": 0, "angles": []})
                    stats["rows"][name]["flag"] += int(bad)
                    stats["rows"][name]["angles"].append(a)
            new, n = rectify_hand(hp, which, t, alpha, parts)
            new = np.array([[x / aspect, y] for x, y in new], dtype=float)
            if not np.isfinite(new).all():          # phòng xa: không bao giờ ghi NaN vào dữ liệu
                continue
            if stats is not None:
                stats["keypoints"] += n
            H[f] = new
        for j, c in enumerate(cols):
            out[c] = H[:, j]
    return out


def new_stats():
    return {"hands": 0, "any": 0, "keypoints": 0, "rows": {}}


def summarize(stats):
    h = max(stats["hands"], 1)
    rows = {}
    for name, r in stats["rows"].items():
        a = np.asarray(r["angles"], dtype=float)
        a = a[np.isfinite(a)] if np.isfinite(a).any() else np.zeros(1)
        rows[name] = {"pct_flag": round(100 * r["flag"] / h, 2),
                      "p1": round(float(np.percentile(a, 1)), 1), "p50": round(float(np.percentile(a, 50)), 1),
                      "p99": round(float(np.percentile(a, 99)), 1)}
    return {"hands": stats["hands"], "pct_hands_rectified": round(100 * stats["any"] / h, 2),
            "rectified_joints": stats["keypoints"], "rows": rows}


def _aspect(s):
    try:
        w, h = float(s.get("w", 0)), float(s.get("h", 0))
        return w / h if w > 0 and h > 0 else 1.0
    except (TypeError, ValueError):
        return 1.0


def main():
    from template import load_cfg, read_csv_samples, write_csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", help="csv sau hiệu chỉnh; bỏ trống = chỉ thống kê")
    ap.add_argument("--stats", help="file json ghi thống kê")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--parts", default="aa,fe", help="aa, fe hoặc aa,fe")
    ap.add_argument("--tables", default=str(TABLE_DIR / "dip_only"),
                    help="thư mục chứa 2 bảng: tables/dip_only (mặc định, bảo thủ) hoặc tables/full")
    ap.add_argument("--cfg", default=str(TEMPLATE_CFG))
    ap.add_argument("--limit", type=int, default=None, help="chỉ xử lý n mẫu đầu (để thử nhanh)")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    t = load_tables(args.tables)
    parts = tuple(p.strip() for p in args.parts.split(",") if p.strip())
    samples = read_csv_samples(args.inp, cfg)[: args.limit]
    st = new_stats()
    for i, s in enumerate(samples):
        s["points"] = rectify_points(s["points"], cfg, t, args.alpha, parts, _aspect(s), st)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(samples)} mẫu")
    summ = summarize(st)
    summ.update({"inp": args.inp, "tables": args.tables, "alpha": args.alpha, "parts": list(parts), "samples": len(samples)})
    print(f"Bàn tay có dữ liệu: {summ['hands']} | bị sửa ít nhất 1 khớp: {summ['pct_hands_rectified']}% "
          f"| tổng số lần sửa khớp: {summ['rectified_joints']}")
    print(f"{'dòng bảng':<16}{'% bàn tay lỗi':>14}{'góc p1':>9}{'p50':>8}{'p99':>8}")
    for name, r in summ["rows"].items():
        print(f"{name:<16}{r['pct_flag']:>14}{r['p1']:>9}{r['p50']:>8}{r['p99']:>8}")
    if args.stats:
        Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats).write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
        print("Đã ghi", args.stats)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_csv(samples, cfg, args.out)
        print("Đã ghi", args.out)


if __name__ == "__main__":
    main()
