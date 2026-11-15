"""Bước 4 (tùy chọn): cân bằng lớp cho tập TRAIN bằng SMOTE, giống mục 3.1 của bài Siformer.

    python src/oversample.py --inp data/csv/train.csv --out data/csv/train_smote.csv

CHỈ dùng cho tập train, sau khi đã chia. Trước khi dùng, kiểm tra train.py của Siformer:
nếu code đã tự làm SMOTE thì KHÔNG chạy file này (tránh cân bằng hai lần).

Cách làm (giống bài): thêm số 0 cho mọi chuỗi bằng độ dài lớn nhất, coi mỗi video là một vector;
với lớp ít mẫu, chọn ngẫu nhiên một mẫu a và một trong k láng giềng gần nhất b cùng lớp,
tạo mẫu mới a + λ(b − a), λ ~ U(0,1), cho tới khi lớp đủ số mẫu của lớp nhiều nhất.
"""
import argparse

import numpy as np

from config import TEMPLATE_CFG
from template import load_cfg, read_csv_samples, write_csv


def to_matrix(samples, bases):
    T = max(len(s["points"][bases[0]]) for s in samples)
    X = np.zeros((len(samples), T, len(bases), 2), dtype=np.float32)
    for i, s in enumerate(samples):
        for j, b in enumerate(bases):
            a = s["points"][b]
            X[i, : len(a), j] = a
    return X


def trim(seq, missing):
    """Bỏ các frame cuối mà mọi giá trị đều bằng giá trị thiếu (phần đệm)."""
    flat = seq.reshape(seq.shape[0], -1)
    keep = np.where(~np.all(flat == missing, axis=1))[0]
    return seq[: keep[-1] + 1] if len(keep) else seq[:1]


def smote(X, y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    flat = X.reshape(len(X), -1)
    counts = np.bincount(y)
    target = counts.max()
    new_X, new_y = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        need = target - len(idx)
        if need <= 0:
            continue
        if len(idx) == 1:  # chỉ một mẫu: nhân bản
            new_X += [X[idx[0]]] * need
            new_y += [c] * need
            continue
        F = flat[idx].astype(np.float64)
        sq = (F ** 2).sum(1)
        d = sq[:, None] + sq[None, :] - 2 * F @ F.T   # khoảng cách bình phương, tiết kiệm bộ nhớ
        np.fill_diagonal(d, np.inf)
        kk = min(k, len(idx) - 1)
        nn = np.argsort(d, axis=1)[:, :kk]
        for _ in range(need):
            a = rng.integers(len(idx))
            b = nn[a, rng.integers(kk)]
            lam = rng.random()
            new_X.append(X[idx[a]] + lam * (X[idx[b]] - X[idx[a]]))
            new_y.append(c)
    return new_X, new_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=str(TEMPLATE_CFG))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    bases = cfg["keypoint_bases"]
    samples = read_csv_samples(args.inp, cfg)
    X = to_matrix(samples, bases)
    y = np.array([s["label"] for s in samples])
    new_X, new_y = smote(X, y, args.k, args.seed)
    out = list(samples)
    for seq, c in zip(new_X, new_y):
        seq = trim(seq, cfg["missing_value"])
        out.append({"label": int(c), "points": {b: seq[:, j] for j, b in enumerate(bases)},
                    "w": samples[0].get("w", ""), "h": samples[0].get("h", ""), "fps": samples[0].get("fps", "")})
    write_csv(out, cfg, args.out)
    print(f"{len(samples)} mẫu gốc + {len(new_X)} mẫu tổng hợp = {len(out)}; mỗi lớp {np.bincount(y).max()} mẫu")


if __name__ == "__main__":
    main()
