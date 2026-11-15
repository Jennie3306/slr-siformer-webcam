"""Đọc / kiểm tra / ghi file csv theo đúng định dạng dữ liệu SPOTER mà Siformer dùng.

Ý tưởng: không đoán định dạng. Lấy một file csv gốc của SPOTER làm "khuôn" (template),
đọc danh sách cột, quy ước tọa độ và cách đặt tên tay, rồi ghi dữ liệu MediaPipe
ra đúng y hệt khuôn đó.
"""
import json
import re

import numpy as np
import pandas as pd

from config import BODY_PARTS, HAND_INDEX

AXES = ("_X", "_Y")
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_cell(v):
    """Ô csv chứa một danh sách số theo frame, ví dụ "[0.1, 0.2]"."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    if isinstance(v, str):
        s = re.sub(r"np\.float\d+\(", "(", v)          # phòng khi dữ liệu được ghi từ numpy 2
        return [float(x) for x in _NUM.findall(s)]
    return []


def split_col(col):
    """'indexTip_0_X' -> ('indexTip_0', 'X'); cột không phải keypoint -> (None, None)."""
    for ax in AXES:
        if col.endswith(ax):
            return col[: -len(ax)], ax[1:]
    return None, None


def classify_base(base):
    """Phân loại tên điểm.

    Trả về ('hand', tên_điểm, hậu_tố) | ('body', bên, phần) | ('unknown', base, None)
    bên ∈ {'left','right',None}; phần là khóa trong config.MP_POSE bỏ bên (eye, wrist...) hoặc 'nose'/'neck'.
    """
    m = re.match(r"^([A-Za-z]+)(_.+)$", base)
    if m and m.group(1) in HAND_INDEX:
        return "hand", m.group(1), m.group(2)
    if base in ("nose", "neck"):
        return "body", None, base
    for side in ("left", "right"):
        if base.startswith(side) and base[len(side):] in BODY_PARTS:
            return "body", side, BODY_PARTS[base[len(side):]]
    return "unknown", base, None


def keypoint_bases(columns):
    """Danh sách tên điểm theo đúng thứ tự xuất hiện trong csv (mỗi điểm có _X và _Y)."""
    bases = []
    for c in columns:
        b, ax = split_col(c)
        if b is not None and ax == "X":
            bases.append(b)
    return bases


def _mean_xy(df, base, axis):
    vals = []
    for v in df[f"{base}_{axis}"]:
        arr = np.asarray(parse_cell(v), dtype=float)
        vals.extend(arr[arr != 0].tolist())
    return float(np.mean(vals)) if vals else float("nan")


def _pair_dist(df, a, b):
    """Khoảng cách trung bình giữa điểm a và b qua mọi frame mà cả hai đều có giá trị khác 0."""
    d = []
    for _, r in df.iterrows():
        ax, ay = np.asarray(parse_cell(r[f"{a}_X"]), float), np.asarray(parse_cell(r[f"{a}_Y"]), float)
        bx, by = np.asarray(parse_cell(r[f"{b}_X"]), float), np.asarray(parse_cell(r[f"{b}_Y"]), float)
        n = min(len(ax), len(bx))
        ok = (ax[:n] != 0) & (bx[:n] != 0)
        if ok.any():
            d.extend(np.hypot(ax[:n][ok] - bx[:n][ok], ay[:n][ok] - by[:n][ok]).tolist())
    return float(np.mean(d)) if d else float("inf")


def inspect(csv_path, n_rows=300):
    """Phân tích file csv khuôn của SPOTER, trả về dict cấu hình để ghi dữ liệu MediaPipe."""
    df = pd.read_csv(csv_path, nrows=n_rows)
    cols = list(df.columns)
    bases = keypoint_bases(cols)
    kinds = {b: classify_base(b) for b in bases}
    unknown = [b for b, k in kinds.items() if k[0] == "unknown"]
    hand_suffixes = sorted({k[2] for k in kinds.values() if k[0] == "hand"})

    label_col = "labels" if "labels" in cols else None
    if label_col is None:
        ints = [c for c in cols if split_col(c)[0] is None and pd.api.types.is_integer_dtype(df[c])]
        label_col = ints[0] if ints else None
    labels = df[label_col] if label_col else pd.Series([], dtype=int)

    # Thang giá trị: [0,1] hay pixel
    sample_vals = []
    for b in bases[:20]:
        for ax in ("X", "Y"):
            for v in df[f"{b}_{ax}"].head(50):
                a = np.asarray(parse_cell(v), float)
                sample_vals.extend(a[a != 0].tolist())
    vmax = float(np.max(sample_vals)) if sample_vals else 1.0
    scale = "unit" if vmax <= 1.5 else "pixel"

    # Trục y: mũi nằm trên vai. Nếu y(mũi) > y(vai) thì gốc tọa độ ở dưới (y hướng lên).
    nose_y = _mean_xy(df, "nose", "Y") if "nose" in bases else float("nan")
    sh_y = np.nanmean([_mean_xy(df, f"{s}Shoulder", "Y") for s in ("left", "right") if f"{s}Shoulder" in bases])
    y_up = bool(nose_y > sh_y) if not np.isnan(nose_y) else False

    # Trái/phải của thân: MediaPipe (ảnh không lật) có vai trái của người ký ở x lớn hơn.
    lx = _mean_xy(df, "leftShoulder", "X") if "leftShoulder" in bases else float("nan")
    rx = _mean_xy(df, "rightShoulder", "X") if "rightShoulder" in bases else float("nan")
    swap_lr = bool(lx < rx) if not (np.isnan(lx) or np.isnan(rx)) else False

    # Hậu tố tay nào ứng với bên nào: so cổ tay của bàn tay với cổ tay của thân.
    hand_side = {}
    for s in hand_suffixes:
        w = f"wrist{s}"
        if w in bases and "leftWrist" in bases and "rightWrist" in bases:
            dl, dr = _pair_dist(df, w, "leftWrist"), _pair_dist(df, w, "rightWrist")
            hand_side[s] = "left" if dl < dr else "right"
    if len(set(hand_side.values())) < len(hand_side):
        print("[CẢNH BÁO] Không tách được bên của hai bàn tay; hãy kiểm tra bằng visualize_template.py")

    # Giá trị khi mất tay
    hand_zero = []
    for b in bases:
        if kinds[b][0] == "hand":
            for v in df[f"{b}_X"].head(100):
                a = np.asarray(parse_cell(v), float)
                hand_zero.append(float((a == 0).mean()) if len(a) else 0.0)
    lengths = [len(parse_cell(v)) for v in df[f"{bases[0]}_X"]]

    cfg = {
        "template_csv": str(csv_path),
        "columns": cols,
        "keypoint_bases": bases,
        "label_col": label_col,
        "label_min": int(labels.min()) if len(labels) else 0,
        "label_max": int(labels.max()) if len(labels) else 0,
        "scale": scale,
        "y_up": y_up,
        "swap_lr": swap_lr,
        "hand_suffix_side": hand_side,
        "missing_value": 0.0,
        "fixed_len": int(np.max(lengths)) if int(np.min(lengths)) == int(np.max(lengths)) else None,
        "unknown_bases": unknown,
        "stats": {
            "n_rows_checked": len(df),
            "n_keypoints": len(bases),
            "max_value_seen": vmax,
            "mean_hand_zero_ratio": float(np.mean(hand_zero)) if hand_zero else 0.0,
            "frames_min_mean_max": [int(np.min(lengths)), float(np.mean(lengths)), int(np.max(lengths))],
            "nose_y": nose_y, "shoulder_y": float(sh_y), "leftShoulder_x": lx, "rightShoulder_x": rx,
        },
    }
    return cfg


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fit_length(arr, n, missing):
    """Đệm giá trị thiếu ở cuối (hoặc cắt bớt) để chuỗi dài đúng n frame."""
    if len(arr) >= n:
        return arr[:n]
    pad = np.full((n - len(arr), arr.shape[1]), missing, dtype=float)
    return np.concatenate([arr, pad], axis=0)


def write_csv(samples, cfg, out_path):
    """samples: list[dict] với khóa: label (int, 0-based), points (dict base -> mảng (T,2)), w, h, fps.

    Nhãn được cộng label_min của khuôn (nếu khuôn đánh số từ 1 thì ghi từ 1).
    """
    rows, warned = [], set()
    for s in samples:
        row = {}
        for c in cfg["columns"]:
            base, ax = split_col(c)
            if base is not None:
                pts = s["points"][base]
                if cfg.get("fixed_len"):
                    pts = fit_length(pts, cfg["fixed_len"], cfg["missing_value"])
                arr = pts[:, 0 if ax == "X" else 1]
                row[c] = "[" + ", ".join(f"{v:.5f}" for v in arr) + "]"
            elif c == cfg["label_col"]:
                row[c] = int(s["label"]) + cfg["label_min"]
            elif c == "video_size_width":
                row[c] = s.get("w", "")
            elif c == "video_size_height":
                row[c] = s.get("h", "")
            elif c == "video_fps":
                row[c] = s.get("fps", "")
            else:
                if c not in warned and not c.startswith("Unnamed"):
                    print(f"[CẢNH BÁO] Cột '{c}' không biết điền gì, để trống")
                    warned.add(c)
                row[c] = ""
        rows.append(row)
    pd.DataFrame(rows, columns=cfg["columns"]).to_csv(out_path, index=False)


def read_csv_samples(csv_path, cfg):
    """Đọc csv định dạng khuôn -> list dict(label 0-based, points base->(T,2))."""
    df = pd.read_csv(csv_path)
    out = []
    for _, r in df.iterrows():
        pts = {}
        for b in cfg["keypoint_bases"]:
            x = np.asarray(parse_cell(r[f"{b}_X"]), float)
            y = np.asarray(parse_cell(r[f"{b}_Y"]), float)
            pts[b] = np.stack([x, y], axis=1) if len(x) else np.zeros((0, 2))
        s = {"label": int(r[cfg["label_col"]]) - cfg["label_min"], "points": pts}
        for k in ("video_size_width", "video_size_height", "video_fps"):
            if k in df.columns:
                s[{"video_size_width": "w", "video_size_height": "h", "video_fps": "fps"}[k]] = r[k]
        out.append(s)
    return out
