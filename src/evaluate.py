"""Bước 6b: tính chỉ số và vẽ hình từ file dự đoán.

Một lần chạy:
    python src/evaluate.py --pred results/siformer_s0/pred_test.npz --out results/siformer_s0
Gộp nhiều seed (trung bình ± độ lệch chuẩn):
    python src/evaluate.py --aggregate results/siformer_s*/metrics.json --out results/siformer_summary.json

Sinh: metrics.json, per_class.csv, confused_pairs.csv, per_class_hist.png, confused_pairs.png
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import LABEL_MAP


def topk_acc(logits, y, k):
    top = np.argsort(-logits, axis=1)[:, :k]
    return float((top == y[:, None]).any(1).mean())


def metrics(logits, y):
    pred = logits.argmax(1)
    classes = np.unique(y)
    per_cls = {int(c): float((pred[y == c] == c).mean()) for c in classes}
    top5 = np.argsort(-logits, axis=1)[:, :5]
    per_cls5 = {int(c): float((top5[y == c] == c).any(1).mean()) for c in classes}
    return {
        "n": int(len(y)), "n_classes_in_test": int(len(classes)),
        "pi_top1": float((pred == y).mean()), "pi_top5": topk_acc(logits, y, 5),
        "pc_top1": float(np.mean(list(per_cls.values()))), "pc_top5": float(np.mean(list(per_cls5.values()))),
    }, per_cls, pred


def load_names():
    try:
        with open(LABEL_MAP, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        return {}


def single(args):
    d = np.load(args.pred)
    y, logits = d["y"], d["logits"]
    # nếu nhãn trong csv đánh số từ 1 còn đầu ra mô hình từ 0, Dataset của tác giả thường đã trừ 1;
    # nếu Top-1 gần 0 bất thường, kiểm tra lệch nhãn ở đây
    m, per_cls, pred = metrics(logits, y)
    if "ms" in d:
        ms = d["ms"]
        m["model_ms_mean"], m["model_ms_p95"] = float(ms.mean()), float(np.percentile(ms, 95))
    names = load_names()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

    counts = pd.Series(y).value_counts()
    pc = pd.DataFrame([{"label": c, "gloss": names.get(c, str(c)), "n_test": int(counts[c]), "acc": a}
                       for c, a in per_cls.items()]).sort_values("acc")
    pc.to_csv(out / "per_class.csv", index=False)

    pairs = pd.Series([(int(t), int(p)) for t, p in zip(y, pred) if t != p]).value_counts().head(15)
    cp = pd.DataFrame([{"true": names.get(t, t), "pred": names.get(p, p), "count": int(n)} for (t, p), n in pairs.items()])
    cp.to_csv(out / "confused_pairs.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(pc["acc"] * 100, bins=np.arange(0, 105, 10), color="#2a78d6", edgecolor="white")
    ax.set_xlabel("Độ chính xác của lớp (%)")
    ax.set_ylabel("Số lớp")
    ax.set_title("Phân bố độ chính xác theo lớp")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(out / "per_class_hist.png", dpi=160); plt.close()

    if len(cp):
        fig, ax = plt.subplots(figsize=(6, 0.3 * len(cp) + 1))
        lab = [f"{t} → {p}" for t, p in zip(cp["true"], cp["pred"])][::-1]
        ax.barh(lab, cp["count"][::-1], color="#b5b5b5")
        ax.set_xlabel("Số lần nhầm (đúng → đoán)")
        ax.set_title("Các cặp nhầm nhiều nhất")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        plt.tight_layout(); plt.savefig(out / "confused_pairs.png", dpi=160); plt.close()

    print(json.dumps(m, indent=2))
    print(f"5 lớp yếu nhất:\n{pc.head(5).to_string(index=False)}")


def aggregate(args):
    files = sorted({f for pat in args.aggregate for f in glob.glob(pat)})
    ms = [json.load(open(f)) for f in files]
    keys = [k for k in ms[0] if isinstance(ms[0][k], float)]
    summ = {k: {"mean": float(np.mean([m[k] for m in ms])), "std": float(np.std([m[k] for m in ms], ddof=1)) if len(ms) > 1 else 0.0}
            for k in keys}
    summ["files"] = files
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summ, f, indent=2)
    for k in keys:
        scale = 100 if k.startswith(("pi_", "pc_")) else 1
        print(f"{k}: {summ[k]['mean'] * scale:.2f} ± {summ[k]['std'] * scale:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--aggregate", nargs="*")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    aggregate(args) if args.aggregate else single(args)


if __name__ == "__main__":
    main()
