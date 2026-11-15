"""Tự kiểm tra các phần không cần mediapipe/torch bằng dữ liệu giả.

    python tests/selftest.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

from config import HAND_INDEX, MP_POSE  # noqa: E402
from keypoints import assign_hands, resample, to_template  # noqa: E402
from segmenter import AutoSegmenter, ManualRecorder  # noqa: E402
from template import inspect, read_csv_samples, write_csv  # noqa: E402

BODY = ["nose", "neck", "rightEye", "leftEye", "rightEar", "leftEar", "rightShoulder", "leftShoulder",
        "rightElbow", "leftElbow", "rightWrist", "leftWrist"]
HAND = list(HAND_INDEX)
rng = np.random.default_rng(0)

# Vị trí "thật" của người ký trong ảnh không lật (y hướng xuống, left = tay trái người ký -> x lớn hơn)
TRUE = {"nose": (0.5, 0.30), "neck": (0.5, 0.50), "leftEye": (0.53, 0.28), "rightEye": (0.47, 0.28),
        "leftEar": (0.56, 0.30), "rightEar": (0.44, 0.30), "leftShoulder": (0.60, 0.50), "rightShoulder": (0.40, 0.50),
        "leftElbow": (0.66, 0.65), "rightElbow": (0.34, 0.65), "leftWrist": (0.65, 0.45), "rightWrist": (0.35, 0.42)}


def hand_pts(wrist):
    return np.array([[wrist[0] + 0.004 * i, wrist[1] - 0.004 * i] for i in range(21)])


def make_template(path, n=30, T=20):
    """Khuôn giả theo kiểu Apple Vision: y hướng lên, tay '_0' là tay PHẢI, nhãn từ 1."""
    cols = [f"{b}_{a}" for b in BODY + ["root"] for a in "XY"]
    cols += [f"{h}_{s}_{a}" for s in ("right", "left") for h in HAND for a in "XY"]
    rows = []
    for i in range(n):
        row = {"labels": i % 5 + 1, "video_fps": 25, "video_size_width": 640, "video_size_height": 480}
        row["root_X"], row["root_Y"] = str([0.0] * T), str([0.0] * T)
        for b in BODY:
            x, y = TRUE[b]
            row[f"{b}_X"] = str([float(x)] * T)
            row[f"{b}_Y"] = str([float(1 - y)] * T)
        for s, side in (("right", "right"), ("left", "left")):
            hp = hand_pts(TRUE[f"{side}Wrist"])
            for j, h in enumerate(HAND):
                xs = [float(hp[j, 0]) if t % 4 else 0.0 for t in range(T)]
                ys = [float(1 - hp[j, 1]) if t % 4 else 0.0 for t in range(T)]
                row[f"{h}_{s}_X"], row[f"{h}_{s}_Y"] = str(xs), str(ys)
        rows.append(row)
    pd.DataFrame(rows, columns=["labels"] + cols + ["video_fps", "video_size_width", "video_size_height"]).to_csv(path, index=False)


def make_mediapipe(T=20):
    pose = np.full((T, 33, 2), np.nan)
    for name, idx in MP_POSE.items():
        side, part = (name.split("_") + [""])[:2] if "_" in name else ("", name)
        key = name if not side else side + part.capitalize()
        pose[:, idx] = TRUE[key]
    left = np.repeat(hand_pts(TRUE["leftWrist"])[None], T, 0)
    right = np.repeat(hand_pts(TRUE["rightWrist"])[None], T, 0)
    right[::5] = np.nan   # mất tay phải ở vài frame
    return pose, left, right


def check(cond, msg):
    print(("OK   " if cond else "LỖI ") + msg)
    if not cond:
        global FAILED
        FAILED = True


FAILED = False
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    tpl = d / "tpl.csv"
    make_template(tpl)
    cfg = inspect(tpl)
    check(cfg["y_up"] is True, "phát hiện trục y hướng lên")
    check(cfg["swap_lr"] is False, "tên trái/phải theo người ký")
    check(cfg["hand_suffix_side"] == {"_left": "left", "_right": "right"}, f"hậu tố tay kiểu Siformer: {cfg['hand_suffix_side']}")
    check(cfg["fixed_len"] == 20, f"phát hiện độ dài cố định: {cfg['fixed_len']}")
    check(cfg["label_min"] == 1 and cfg["scale"] == "unit", "nhãn bắt đầu từ 1, thang [0,1]")
    check(len(cfg["keypoint_bases"]) == 55 and cfg["unknown_bases"] == ["root"], "54 keypoint + cột phụ 'root'")

    pose, left, right = make_mediapipe()
    # gán tay: đưa nhãn bị đảo, hàm phải sửa lại theo vị trí cổ tay
    l2, r2 = assign_hands(pose[0], right[1], left[1])
    check(np.allclose(l2, left[1]) and np.allclose(r2, right[1]), "gán lại tay trái/phải theo cổ tay")
    pts = to_template(pose, left, right, cfg)
    check(np.allclose(pts["leftShoulder"][0], [0.60, 0.50]), "vai trái đúng vị trí, đã lật y")
    check(np.allclose(pts["wrist_right"][1], [0.35, 1 - 0.42]), "tay '_right' lấy từ tay phải người ký")
    check((pts["wrist_right"][0] == 0).all() and (pts["wrist_left"][0] != 0).all(), "frame mất tay ghi bằng 0")
    check(np.allclose(pts["neck"][0], [0.5, 0.5]), "cổ = trung điểm hai vai")
    check((pts["root"] == 0).all(), "cột phụ 'root' ghi 0")

    # so với khuôn: giá trị phải trùng với dữ liệu "thật" của khuôn
    tpl_s = read_csv_samples(tpl, cfg)[0]
    for b in ("nose", "leftElbow", "indexTip_left", "thumbTip_right"):
        t = tpl_s["points"][b][1]
        check(np.allclose(t, pts[b][1], atol=1e-5), f"{b} khớp với khuôn")

    out = d / "ours.csv"
    samples = [{"label": i % 5, "points": to_template(*make_mediapipe(15 + i), cfg), "w": 640, "h": 480, "fps": 25} for i in range(12)]
    samples += [{"label": 4, "points": to_template(*make_mediapipe(10), cfg), "w": 640, "h": 480, "fps": 25}] * 6
    write_csv(samples, cfg, out)
    df = pd.read_csv(out)
    check(list(df.columns) == list(pd.read_csv(tpl, nrows=1).columns), "cột giống hệt khuôn")
    check(df["labels"].min() == 1, "nhãn ghi theo quy ước của khuôn (từ 1)")
    back = read_csv_samples(out, cfg)
    check(back[3]["label"] == 3 and np.allclose(back[3]["points"]["leftShoulder"][:18], samples[3]["points"]["leftShoulder"], atol=1e-5),
          "đọc lại đúng")
    check(all(len(b["points"]["nose"]) == 20 for b in back) and (back[0]["points"]["nose"][15:] == 0).all(),
          "mọi mẫu được đệm 0 cho đúng độ dài cố định")

    cfgp = d / "cfg.json"
    json.dump(cfg, open(cfgp, "w"))
    r = subprocess.run([sys.executable, str(SRC / "oversample.py"), "--inp", str(out), "--out", str(d / "sm.csv"), "--cfg", str(cfgp)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    sm = pd.read_csv(d / "sm.csv")
    check(r.returncode == 0 and sm["labels"].value_counts().nunique() == 1, f"SMOTE cân bằng lớp: {r.stdout.strip() or r.stderr[-300:]}")
    r = subprocess.run([sys.executable, str(SRC / "robustness.py"), "--inp", str(out), "--outdir", str(d / "rob"), "--cfg", str(cfgp)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    rb = read_csv_samples(d / "rob" / "test_drop_left_hand.csv", cfg)[0]
    check(r.returncode == 0 and (rb["points"]["wrist_left"] == 0).all() and (rb["points"]["wrist_right"] != 0).any(),
          "tạo tập test mất tay trái")

    # evaluate
    y = rng.integers(0, 10, 200)
    logits = rng.normal(size=(200, 10))
    logits[np.arange(200), y] += 2.0
    np.savez(d / "pred.npz", y=y, logits=logits, ms=rng.random(200))
    r = subprocess.run([sys.executable, str(SRC / "evaluate.py"), "--pred", str(d / "pred.npz"), "--out", str(d / "ev")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    m = json.load(open(d / "ev" / "metrics.json")) if r.returncode == 0 else {}
    check(r.returncode == 0 and 0 < m["pi_top1"] < 1 and m["pi_top5"] >= m["pi_top1"], f"evaluate chạy: {m or r.stderr[-300:]}")

# đổi tốc độ khung hình 30 -> 25
P, L, R = make_mediapipe(30)
P2, L2, R2 = resample(P, L, R, 30, 25)
check(len(P2) == 25 and len(L2) == 25, f"đổi 30 frame/giây -> 25: {len(P)} -> {len(P2)} frame")

# chọn checkpoint theo val
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "ck").mkdir()
    for k in range(3):
        (d / "ck" / f"checkpoint_v_{k}.pth").write_text(str(k))
    log = ["[1] TRAIN  loss: 1 acc: 0.1", "Save checkpoint for [1] as out-checkpoints/x/checkpoint_v_0.pth", "[1] VALIDATION  acc: 0.2",
           "[5] TRAIN  loss: 1 acc: 0.5", "Save checkpoint for [5] as out-checkpoints/x/checkpoint_v_1.pth", "[5] VALIDATION  acc: 0.7",
           "[15] VALIDATION  acc: 0.6", "Save checkpoint for [15] as out-checkpoints/x/checkpoint_v_2.pth", "[15] VALIDATION  Top 5 acc: 0.6"]
    (d / "train.log").write_text("\n".join(log), encoding="utf-8")
    r = subprocess.run([sys.executable, str(SRC / "select_checkpoint.py"), "--log", str(d / "train.log"), "--ckpt_dir", str(d / "ck")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    check(r.returncode == 0 and (d / "best.pth").read_text() == "1", f"chọn checkpoint theo val: {r.stdout.strip() or r.stderr[-200:]}")

# segmenter
pose, left, right = make_mediapipe(1)
down = np.full((21, 2), np.nan)
seq = [(pose[0], down, down)] * 10 + [(pose[0], left[0], right[0])] * 30 + [(pose[0], down, down)] * 15
seg = AutoSegmenter()
outs = [o for o in (seg.update(*f) for f in seq) if o is not None]
check(len(outs) == 1 and 28 <= len(outs[0][0]) <= 32, f"tự động tách đúng 1 ký hiệu, dài {len(outs[0][0]) if outs else 0} frame")
mr = ManualRecorder()
mr.toggle()
for f in seq[:20]:
    mr.update(*f)
o = mr.toggle()
check(o is not None and len(o[0]) == 20, "chế độ bấm phím ghi đúng 20 frame")

# hiệu chỉnh tư thế tay (cần code tác giả trong external/Siformer/rectifications)
from config import SIFORMER_DIR  # noqa: E402
if (Path(SIFORMER_DIR) / "rectifications").is_dir():
    from rectify_hands import HAND_NAMES, load_tables, new_stats, rectify_points, summarize  # noqa: E402

    def _hand(side, thumb=(0, 0, 0), pip=0.0, back=False):
        s = -1 if side == "left" else 1
        if back:
            s = -s
        H = np.zeros((21, 2))
        H[1] = [0.1 * s, 0.1]
        ang, cur = 35 * s, H[1].copy()
        for j, Ln in enumerate((0.18, 0.14, 0.12)):
            ang += -s * thumb[j]
            r = np.radians(ang)
            cur = cur + Ln * np.array([np.sin(r), np.cos(r)])
            H[2 + j] = cur
        for k, b in enumerate((5, 9, 13, 17)):
            H[b] = [s * [0.12, 0.04, -0.04, -0.12][k], 0.6]
            ang, cur = 0.0, H[b].copy()
            for j, (Ln, a) in enumerate(zip((0.25, 0.15, 0.1), (0, pip, 0))):
                ang += -s * a
                r = np.radians(ang)
                cur = cur + Ln * np.array([np.sin(r), np.cos(r)])
                H[b + 1 + j] = cur
        return H

    tcfg = {"missing_value": 0.0, "hand_suffix_side": {"_left": "left", "_right": "right"}}
    from rectify_hands import TABLE_DIR  # noqa: E402
    tabs = load_tables(TABLE_DIR / "full")

    def _run(**kw):
        pts = {}
        for i, n in enumerate(HAND_NAMES):
            pts[f"{n}_left"] = _hand("left", **kw)[None, i]
            pts[f"{n}_right"] = _hand("right", **kw)[None, i]
        st = new_stats()
        new = rectify_points(pts, tcfg, tabs, 0.4, ("aa", "fe"), 1.0, st)
        return summarize(st), new

    ok_natural = all(_run(**kw)[0]["rectified_joints"] == 0 for kw in
                     ({}, {"thumb": (30, 60, 70)}, {"thumb": (30, 60, 70), "back": True}))
    check(ok_natural, "hiệu chỉnh: tay tự nhiên (thẳng, ngón cái gập, mu bàn tay) không bị sửa")
    s1, new = _run(pip=160)
    s2, _ = _run(thumb=(0, -40, 0))
    check(s1["rows"]["FE indexPIP"]["pct_flag"] == 100 and s2["rows"]["FE thumbMP"]["pct_flag"] == 100,
          "hiệu chỉnh: PIP gập 160° và ngón cái duỗi ngược 40° bị phát hiện")
    st = new_stats()
    rectify_points(new, tcfg, tabs, 0.4, ("fe",), 1.0, st)
    check(abs(summarize(st)["rows"]["FE indexPIP"]["p50"] - 148) < 0.5, "hiệu chỉnh: alpha 0,4 kéo PIP 160° về 148°")
else:
    print("BỎ QUA kiểm tra hiệu chỉnh tay: chưa có external/Siformer/rectifications")

print("\nKẾT QUẢ:", "CÓ LỖI" if FAILED else "tất cả đều đạt")
sys.exit(1 if FAILED else 0)
